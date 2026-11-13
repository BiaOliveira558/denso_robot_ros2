#!/usr/bin/env python3
"""
IBVS ponderado para pontos XFeat + KLT (ativos com peso w) — versão de COLETA.

Recebe em /ibvs/xfeat_features (extractor com formato_ponderado:=true):
  [N, detectado, stamp_sec, stamp_nanosec,  id1, x1, y1, z1, x1*, y1*, z1*, w1,  ...]

Lei de controle:  v = -lambda * (W L)^+ W e   (DLS normalizado pela soma dos pesos)
O callback calcula a velocidade-ALVO; o cmd_loop (50 Hz) aproxima o comando do alvo
com um filtro de primeira ordem.

Publica para o logger:
  /ibvs/converged (Bool, latched)
  /ibvs/debug (Float64MultiArray), 22 valores + 6 por ponto:
    [status, rms, err_norm, N, soma_w, cond_L, sat_k, idade_s, t_ctrl_ms, below,
     alvo vx vy vz wx wy wz, cmd vx vy vz wx wy wz,
     (id, ex, ey, w, Z, Z*) x N]
"""
import time
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, Bool
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

# ---- códigos de status publicados em /ibvs/debug ----
ST_OK, ST_LOST, ST_STALE, ST_OUTLIER, ST_CONVERGED, ST_BAD_MSG, ST_LOW_W, ST_BAD_Z, ST_WAIT = range(9)


class IBVSPonderado(Node):
    def __init__(self):
        super().__init__('ibvs_xfeat_controller')

        self.declare_parameter('features_topic', '/ibvs/xfeat_features')
        self.declare_parameter('camera_frame', 'basic_camera_link')
        self.declare_parameter('lambda_gain', 1.2)
        self.declare_parameter('convergence_threshold', 0.025)   # RMS ponderado por ponto (coord. normalizada)
        self.declare_parameter('convergence_frames', 5)
        self.declare_parameter('max_lin_vel', 0.08)    # m/s
        self.declare_parameter('max_ang_vel', 0.30)    # rad/s
        self.declare_parameter('max_msg_age', 0.30)    # s
        self.declare_parameter('hold_time', 0.8)       # s sem medida nova -> para
        self.declare_parameter('cmd_rate', 50.0)       # Hz enviados ao Servo
        self.declare_parameter('tau_filtro', 0.15)     # s: constante de tempo do filtro de velocidade
        self.declare_parameter('interaction_mode', 'mean')  # current | desired | mean
        self.declare_parameter('damping', 0.01)        # mu do DLS
        self.declare_parameter('min_pontos', 4)
        self.declare_parameter('min_soma_w', 4.0)
        self.declare_parameter('start_delay', 2.0)     # s esperando antes de agir (logger grava a pose inicial)
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
        self.start_delay = gp('start_delay')
        self.log_every = max(int(gp('log_every')), 1)

        self.alpha = 1.0 - np.exp(-(1.0 / self.cmd_rate) / max(self.tau, 1e-3))

        self.v_alvo_lin = np.zeros(3)
        self.v_alvo_ang = np.zeros(3)
        self.v_cmd_lin = np.zeros(3)
        self.v_cmd_ang = np.zeros(3)
        self.t_alvo = None
        self._converged = False
        self._below = 0
        self._n_msg = 0
        self._t_start = self.get_clock().now()
        self._started = self.start_delay <= 0.0

        self.create_subscription(Float64MultiArray, gp('features_topic'), self.features_cb, 10)
        self.pub_error = self.create_publisher(Float64MultiArray, '/ibvs/error', 10)
        self.pub_debug = self.create_publisher(Float64MultiArray, '/ibvs/debug', 50)
        self.pub_twist = self.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_conv = self.create_publisher(Bool, '/ibvs/converged', latched)
        self.create_timer(1.0 / self.cmd_rate, self.cmd_loop)

        self.get_logger().info(
            f'IBVS PONDERADO (coleta) | lambda={self.lam} | thr={self.threshold} ({self.conv_frames}) | '
            f'vmax=({self.max_l} m/s, {self.max_a} rad/s) | L={self.mode} | mu={self.mu} | '
            f'tau={self.tau}s | hold={self.hold_time}s | start_delay={self.start_delay}s')

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
        A = (Lw.T @ Lw) / wsum + (self.mu ** 2) * np.eye(6)
        return np.linalg.solve(A, (Lw.T @ ew) / wsum)

    def saturate(self, v_lin, v_ang):
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
        self.v_alvo_lin[:] = 0.0
        self.v_alvo_ang[:] = 0.0
        self.v_cmd_lin[:] = 0.0
        self.v_cmd_ang[:] = 0.0
        self.t_alvo = self.get_clock().now()
        self.publish_twist(self.v_cmd_lin, self.v_cmd_ang)

    def publish_debug(self, status, t0, rms=np.nan, en=np.nan, n=0, wsum=np.nan,
                      cond=np.nan, k=np.nan, age=np.nan, pts=None):
        t_ctrl = (time.perf_counter() - t0) * 1000.0
        dados = ([float(status), float(rms), float(en), float(n), float(wsum), float(cond),
                  float(k), float(age), float(t_ctrl), float(self._below)]
                 + [float(v) for v in self.v_alvo_lin] + [float(v) for v in self.v_alvo_ang]
                 + [float(v) for v in self.v_cmd_lin] + [float(v) for v in self.v_cmd_ang])
        if pts is not None:
            dados += [float(v) for v in np.asarray(pts).ravel()]
        msg = Float64MultiArray()
        msg.data = dados
        self.pub_debug.publish(msg)

    # ------------------------------------------------------------------
    def cmd_loop(self):
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
        t0 = time.perf_counter()
        if self._converged:
            return

        # ---- espera inicial: o logger grava a pose inicial com o robô parado ----
        if not self._started:
            if (self.get_clock().now() - self._t_start).nanoseconds * 1e-9 < self.start_delay:
                self.publish_debug(ST_WAIT, t0)
                return
            self._started = True
            self.get_logger().info('start_delay concluido — IBVS ativo')

        d = np.asarray(msg.data)
        if d.size < 4:
            self.publish_debug(ST_BAD_MSG, t0)
            return

        n = int(d[0])
        valido = bool(d[1])
        stamp = Time(seconds=int(d[2]), nanoseconds=int(d[3]), clock_type=self.get_clock().clock_type)
        age = (self.get_clock().now() - stamp).nanoseconds * 1e-9

        if not valido or n < self.min_pontos:
            self.parar()
            self.publish_debug(ST_LOST, t0, n=n, age=age)
            self.get_logger().warn(f'Sem pontos validos (N={n}, detectado={int(valido)}) -> parado',
                                   throttle_duration_sec=2.0)
            return
        if d.size < 4 + COLS * n:
            self.parar()
            self.publish_debug(ST_BAD_MSG, t0, n=n, age=age)
            self.get_logger().error(
                f'Mensagem com {d.size} campos, esperado {4 + COLS * n}. '
                f'O extractor esta com formato_ponderado:=true?', throttle_duration_sec=2.0)
            return
        if age > self.max_age or age < -0.05:
            self.parar()
            self.publish_debug(ST_STALE, t0, n=n, age=age)
            self.get_logger().warn(f'Medida velha (idade={age:.3f}s) -> parado', throttle_duration_sec=2.0)
            return

        P = d[4:4 + COLS * n].reshape(n, COLS)
        ids = P[:, 0].astype(int)
        s = P[:, 1:3].ravel()
        Z = P[:, 3]
        s_star = P[:, 4:6].ravel()
        Zs = P[:, 6]
        w = P[:, 7]

        e = s - s_star
        e2 = e.reshape(n, 2)
        pts = np.column_stack([ids, e2[:, 0], e2[:, 1], w, Z, Zs])   # para o logger

        if np.any(Z <= 0) or np.any(Zs <= 0):
            self.parar()
            self.publish_debug(ST_BAD_Z, t0, n=n, age=age, pts=pts)
            self.get_logger().warn('Z ou Z* invalido em algum ponto -> parado', throttle_duration_sec=2.0)
            return

        wsum = float(w.sum())
        if wsum < self.min_soma_w:
            self.parar()
            self.publish_debug(ST_LOW_W, t0, n=n, wsum=wsum, age=age, pts=pts)
            self.get_logger().warn(f'Soma dos pesos {wsum:.2f} < {self.min_soma_w} -> parado',
                                   throttle_duration_sec=2.0)
            return

        # ---- erro ponderado ----
        rms = float(np.sqrt(np.sum(w * np.sum(e2 ** 2, axis=1)) / wsum))
        err_norm = float(np.linalg.norm(e))

        err_msg = Float64MultiArray()
        err_msg.data = [rms, float(n), wsum]
        self.pub_error.publish(err_msg)

        # ---- convergência (só com todos os pontos inteiros) ----
        if rms < self.threshold and np.all(w > 0.99):
            self._below += 1
            if self._below >= self.conv_frames:
                self.parar()
                self._converged = True
                self.publish_debug(ST_CONVERGED, t0, rms, err_norm, n, wsum, age=age, pts=pts)
                self.pub_conv.publish(Bool(data=True))
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
        cond = float(np.linalg.cond(Lw))

        v_lin, v_ang, k = self.saturate(R_OPT2PHYS @ v[0:3], R_OPT2PHYS @ v[3:6])
        self.v_alvo_lin = v_lin
        self.v_alvo_ang = v_ang
        self.t_alvo = self.get_clock().now()
        self.publish_debug(ST_OK, t0, rms, err_norm, n, wsum, cond, k, age, pts)

        self._n_msg += 1
        if self._n_msg % self.log_every == 0:
            pesos = ' '.join(f'{i}:{wi:.2f}' for i, wi in zip(ids, w))
            self.get_logger().info(
                f'rms={rms:.4f} ||e||={err_norm:.4f} | N={n} soma_w={wsum:.2f} [{pesos}] | '
                f'cond={cond:.1f} | idade={age * 1000:.0f}ms k={k:.2f} | '
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