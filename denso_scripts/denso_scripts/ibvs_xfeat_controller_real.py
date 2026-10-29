#!/usr/bin/env python3
"""
IBVS ponderado para pontos XFeat + KLT (6 ativos com peso w).

Recebe em /ibvs/xfeat_features (extractor com formato_ponderado:=true):
  [N, detectado, stamp_sec, stamp_nanosec,  id1, x1, y1, z1, x1*, y1*, z1*, w1,  ...]

Lei de controle:  v = -lambda * (W L)^+ W e   (DLS normalizado pela soma dos pesos)
O callback calcula a velocidade-ALVO; o cmd_loop (50 Hz) aproxima o comando do alvo
com um filtro de primeira ordem, para a velocidade não mudar em degrau a cada frame.
"""
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped
import numpy as np


CMD_VEL_TOPIC = '/servo_node/delta_twist_cmds'

# converte uma velocidade do frame óptico para o frame físico da câmera
R_OPT2PHYS = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

COLS = 8   # id, x, y, z, x*, y*, z*, w


class IBVSPonderado(Node):
    def __init__(self):
        super().__init__('ibvs_controller')

        self.declare_parameter('features_topic', '/ibvs/xfeat_features')
        self.declare_parameter('camera_frame', 'basic_camera_link')
        self.declare_parameter('lambda_gain', 1.2)
        self.declare_parameter('convergence_threshold', 0.015)   # RMS ponderado por ponto (coord. normalizada)
        self.declare_parameter('convergence_frames', 5)
        self.declare_parameter('max_lin_vel', 0.08)    # m/s   (conservador para o 1º teste)
        self.declare_parameter('max_ang_vel', 0.30)    # rad/s (conservador para o 1º teste)
        self.declare_parameter('max_msg_age', 0.30)    # s
        self.declare_parameter('hold_time', 0.4)       # s sem medida nova -> para
        self.declare_parameter('cmd_rate', 50.0)       # Hz enviados ao Servo
        self.declare_parameter('tau_filtro', 0.15)     # s: constante de tempo do filtro de velocidade
        self.declare_parameter('interaction_mode', 'mean')  # current | desired | mean
        self.declare_parameter('damping', 0.05)        # mu do DLS
        self.declare_parameter('min_pontos', 4)
        self.declare_parameter('min_soma_w', 4.0)
        self.declare_parameter('log_every', 5)

        gp = lambda n: self.get_parameter(n).value
        self.camera_frame = gp('camera_frame')
        self.lam = gp('lambda_gain')
        self.threshold = gp('convergence_threshold')
        self.conv_frames = gp('convergence_frames')
        self.max_l = gp('max_lin_vel')
        self.max_a = gp('max_ang_vel')
        self.max_age = gp('max_msg_age')
        self.hold_time = gp('hold_time')
        self.cmd_rate = gp('cmd_rate')
        self.tau = gp('tau_filtro')
        self.mode = gp('interaction_mode')
        self.mu = gp('damping')
        self.min_pontos = gp('min_pontos')
        self.min_soma_w = gp('min_soma_w')
        self.log_every = max(int(gp('log_every')), 1)

        # alfa do filtro por ciclo do cmd_loop (filtro de 1ª ordem com constante tau)
        self.alpha = 1.0 - np.exp(-(1.0 / self.cmd_rate) / max(self.tau, 1e-3))

        self.v_alvo_lin = np.zeros(3)
        self.v_alvo_ang = np.zeros(3)
        self.v_cmd_lin = np.zeros(3)
        self.v_cmd_ang = np.zeros(3)
        self.t_alvo = None
        self._converged = False
        self._below = 0
        self._n_msg = 0

        self.create_subscription(Float64MultiArray, gp('features_topic'), self.features_cb, 10)
        self.pub_error = self.create_publisher(Float64MultiArray, '/ibvs/error', 10)
        self.pub_twist = self.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)
        self.create_timer(1.0 / self.cmd_rate, self.cmd_loop)

        self.get_logger().info(
            f'IBVS PONDERADO iniciado | lambda={self.lam} | thr={self.threshold} | '
            f'vmax=({self.max_l} m/s, {self.max_a} rad/s) | L={self.mode} | mu={self.mu} | '
            f'tau={self.tau}s (alpha/ciclo={self.alpha:.3f}) | hold={self.hold_time}s')

    # ------------------------------------------------------------------
    def build_L(self, s, Z):
        n = len(Z)
        L = np.zeros((2 * n, 6))
        for i in range(n):
            x, y, z = s[2 * i], s[2 * i + 1], Z[i]
            L[2 * i]     = [-1 / z, 0, x / z, x * y, -(1 + x ** 2), y]
            L[2 * i + 1] = [0, -1 / z, y / z, 1 + y ** 2, -x * y, -x]
        return L

    def dls(self, Lw, ew, wsum):
        """DLS normalizado pela soma dos pesos: o amortecimento não muda com o nº de pontos."""
        A = (Lw.T @ Lw) / wsum + (self.mu ** 2) * np.eye(6)
        return np.linalg.solve(A, (Lw.T @ ew) / wsum)

    def saturate(self, v_lin, v_ang):
        """Escala o twist inteiro por um único fator (preserva a direção 6D)."""
        nl, na = np.linalg.norm(v_lin), np.linalg.norm(v_ang)
        k = 1.0
        if nl > self.max_l:
            k = min(k, self.max_l / nl)
        if na > self.max_a:
            k = min(k, self.max_a / na)
        return v_lin * k, v_ang * k, k

    def publish_twist(self, v_lin, v_ang):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = map(float, v_lin)
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = map(float, v_ang)
        self.pub_twist.publish(msg)

    def parar(self):
        """Zera alvo e comando imediatamente (sem filtro)."""
        self.v_alvo_lin[:] = 0.0
        self.v_alvo_ang[:] = 0.0
        self.v_cmd_lin[:] = 0.0
        self.v_cmd_ang[:] = 0.0
        self.t_alvo = self.get_clock().now()
        self.publish_twist(self.v_cmd_lin, self.v_cmd_ang)

    # ------------------------------------------------------------------
    def cmd_loop(self):
        """50 Hz: aproxima o comando do alvo com filtro de 1ª ordem; para se a medida ficou velha."""
        if self.t_alvo is None:
            return
        if self._converged:
            self.publish_twist(np.zeros(3), np.zeros(3))
            return
        dt = (self.get_clock().now() - self.t_alvo).nanoseconds * 1e-9
        if dt > self.hold_time:
            self.v_cmd_lin[:] = 0.0
            self.v_cmd_ang[:] = 0.0
            self.publish_twist(self.v_cmd_lin, self.v_cmd_ang)
            self.get_logger().warn(f'Sem medida nova ha {dt:.2f}s -> parado', throttle_duration_sec=1.0)
            return
        a = self.alpha
        self.v_cmd_lin += a * (self.v_alvo_lin - self.v_cmd_lin)
        self.v_cmd_ang += a * (self.v_alvo_ang - self.v_cmd_ang)
        self.publish_twist(self.v_cmd_lin, self.v_cmd_ang)

    # ------------------------------------------------------------------
    def features_cb(self, msg):
        if self._converged:
            return
        d = np.asarray(msg.data)
        if d.size < 4:
            return

        n = int(d[0])
        valido = bool(d[1])
        stamp = Time(seconds=int(d[2]), nanoseconds=int(d[3]), clock_type=self.get_clock().clock_type)

        if not valido or n < self.min_pontos:
            self.parar()
            self.get_logger().warn(f'Sem pontos validos (N={n}, detectado={int(valido)}) -> parado',
                                   throttle_duration_sec=2.0)
            return
        if d.size < 4 + COLS * n:
            self.parar()
            self.get_logger().error(
                f'Mensagem com {d.size} campos, esperado {4 + COLS * n}. '
                f'O extractor esta com formato_ponderado:=true?', throttle_duration_sec=2.0)
            return

        age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
        if age > self.max_age or age < -0.05:
            self.parar()
            self.get_logger().warn(f'Medida velha (idade={age:.3f}s) -> parado', throttle_duration_sec=2.0)
            return

        P = d[4:4 + COLS * n].reshape(n, COLS)
        ids = P[:, 0].astype(int)
        s = P[:, 1:3].ravel()
        Z = P[:, 3]
        s_star = P[:, 4:6].ravel()
        Zs = P[:, 6]
        w = P[:, 7]

        if np.any(Z <= 0) or np.any(Zs <= 0):
            self.parar()
            self.get_logger().warn('Z ou Z* invalido em algum ponto -> parado', throttle_duration_sec=2.0)
            return

        wsum = float(w.sum())
        if wsum < self.min_soma_w:
            self.parar()
            self.get_logger().warn(f'Soma dos pesos {wsum:.2f} < {self.min_soma_w} -> parado',
                                   throttle_duration_sec=2.0)
            return

        # ---- erro ponderado ----
        e = s - s_star
        e2 = e.reshape(n, 2)
        rms = float(np.sqrt(np.sum(w * np.sum(e2 ** 2, axis=1)) / wsum))

        err_msg = Float64MultiArray()
        err_msg.data = [rms, float(n), wsum]
        self.pub_error.publish(err_msg)

        # ---- convergência (só com todos os pontos inteiros) ----
        if rms < self.threshold and np.all(w > 0.99):
            self._below += 1
            if self._below >= self.conv_frames:
                self.parar()
                self._converged = True
                self.get_logger().info(f'CONVERGIU! rms={rms:.4f} | ativos {ids.tolist()}')
                return
        else:
            self._below = 0

        # ---- matriz de interação ----
        if self.mode == 'desired':
            L = self.build_L(s_star, Zs)
        elif self.mode == 'mean':
            L = 0.5 * (self.build_L(s, Z) + self.build_L(s_star, Zs))
        else:
            L = self.build_L(s, Z)

        # ---- lei ponderada ----
        W = np.repeat(w, 2)
        Lw = W[:, None] * L
        ew = W * e
        v = -self.lam * self.dls(Lw, ew, wsum)

        v_lin, v_ang, k = self.saturate(R_OPT2PHYS @ v[0:3], R_OPT2PHYS @ v[3:6])
        self.v_alvo_lin = v_lin
        self.v_alvo_ang = v_ang
        self.t_alvo = self.get_clock().now()

        self._n_msg += 1
        if self._n_msg % self.log_every == 0:
            pesos = ' '.join(f'{i}:{wi:.2f}' for i, wi in zip(ids, w))
            self.get_logger().info(
                f'rms={rms:.4f} | N={n} soma_w={wsum:.2f} [{pesos}] | cond(Lw)={np.linalg.cond(Lw):.1f} | '
                f'idade={age * 1000:.0f}ms k={k:.2f} | '
                f'alvo v_lin=({v_lin[0]:+.3f},{v_lin[1]:+.3f},{v_lin[2]:+.3f}) '
                f'v_ang=({v_ang[0]:+.3f},{v_ang[1]:+.3f},{v_ang[2]:+.3f})')


def main(args=None):
    rclpy.init(args=args)
    node = IBVSPonderado()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.parar()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()