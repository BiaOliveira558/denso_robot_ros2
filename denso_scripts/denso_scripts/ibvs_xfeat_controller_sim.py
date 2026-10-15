#!/usr/bin/env python3
"""
Controlador IBVS com features do XFeat (simulação).

Assina /ibvs/xfeat_features (Float64MultiArray):
  [N, detectado, stamp_sec, stamp_nanosec,
   x1, y1, z1, x1*, y1*, z1*,  ...  (N blocos)]

O comando é calculado a cada mensagem do extrator e REPETIDO a cmd_rate Hz
(laço contínuo para o Servo), até hold_time sem mensagens novas.
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped


CAMERA_FRAME = 'basic_camera_link'
CMD_VEL_TOPIC = '/servo_node/delta_twist_cmds'

R_OPT2PHYS = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)


def build_L(x, y, z):
    """Matriz de interação 2N x 6, linhas intercaladas [x1, y1, x2, y2, ...]."""
    N = len(x)
    L = np.zeros((2 * N, 6))
    L[0::2, 0] = -1 / z
    L[0::2, 2] = x / z
    L[0::2, 3] = x * y
    L[0::2, 4] = -(1 + x**2)
    L[0::2, 5] = y
    L[1::2, 1] = -1 / z
    L[1::2, 2] = y / z
    L[1::2, 3] = 1 + y**2
    L[1::2, 4] = -x * y
    L[1::2, 5] = -x
    return L


class IBVSXFeatController(Node):
    def __init__(self):
        super().__init__('ibvs_xfeat_controller')
        self.declare_parameter('features_topic', '/ibvs/xfeat_features')
        self.declare_parameter('lambda_gain', 0.25)
        self.declare_parameter('convergence_rms', 0.009)
        self.declare_parameter('convergence_frames', 5)
        self.declare_parameter('interaction_mode', 'mean')   # current | desired | mean
        self.declare_parameter('damping', 0.0)
        self.declare_parameter('min_pontos', 6)
        self.declare_parameter('z_aprox', 0.3)
        self.declare_parameter('max_lin_vel', 0.15)
        self.declare_parameter('max_ang_vel', 0.4)
        self.declare_parameter('max_msg_age', 0.0)           # 0 = não verifica (sim: relógios diferentes)
        self.declare_parameter('dry_run', False)             # False = malha fechada
        self.declare_parameter('cmd_rate', 50.0)
        self.declare_parameter('hold_time', 0.5)
        self.declare_parameter('cmd_filter_alpha', 1.0)
        self.declare_parameter('log_csv',
                               time.strftime('/root/ibvs_%Y%m%d_%H%M%S.csv'))  # '' = não grava
        gp = lambda n: self.get_parameter(n).value

        self.lam = gp('lambda_gain')
        self.thr = gp('convergence_rms')
        self.conv_frames = gp('convergence_frames')
        self.mode = gp('interaction_mode')
        self.mu = gp('damping')
        self.min_pontos = max(gp('min_pontos'), 4)
        self.z_aprox = gp('z_aprox')
        self.max_l = gp('max_lin_vel')
        self.max_a = gp('max_ang_vel')
        self.max_age = gp('max_msg_age')
        self.dry_run = gp('dry_run')
        self.hold_time = gp('hold_time')
        self.alpha = gp('cmd_filter_alpha')

        self._converged = False
        self._below = 0
        self._log_counter = 0
        self._log_every_n = 5
        self._cmd_lin = np.zeros(3)
        self._cmd_ang = np.zeros(3)
        self._cmd_time = None

        self.csv = None
        self.t0 = time.time()
        if gp('log_csv'):
            self.csv = open(gp('log_csv'), 'w')
            self.csv.write('t,N,norma_e,rms,cond_L,k,vx,vy,vz,wx,wy,wz,convergiu\n')
            self.get_logger().info(f'Gravando em {gp("log_csv")}')

        self.sub = self.create_subscription(
            Float64MultiArray, gp('features_topic'), self.features_callback, 10)
        self.pub_error = self.create_publisher(Float64MultiArray, '/ibvs/error', 10)
        self.pub_twist = self.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)
        self.create_timer(1.0 / gp('cmd_rate'), self.cmd_loop)

        self.get_logger().info(
            f'IBVS XFeat iniciado (lambda={self.lam}, rms_thr={self.thr}, L={self.mode}, '
            f'cmd_rate={gp("cmd_rate")} Hz, hold={self.hold_time}s, alpha={self.alpha}, '
            f'{"DRY RUN (robo nao se move)" if self.dry_run else "MALHA FECHADA"})')

    # ------------------------------------------------------------------ comando
    def publish_twist(self, v_lin, v_ang):
        if self.dry_run:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = CAMERA_FRAME
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = map(float, v_lin)
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = map(float, v_ang)
        self.pub_twist.publish(msg)

    def set_cmd(self, v_lin, v_ang, filtered=True):
        """Guarda o comando (o cmd_loop é quem publica)."""
        if filtered and self._cmd_time is not None:
            a = self.alpha
            v_lin = a * v_lin + (1 - a) * self._cmd_lin
            v_ang = a * v_ang + (1 - a) * self._cmd_ang
        self._cmd_lin, self._cmd_ang = v_lin, v_ang
        self._cmd_time = time.time()

    def stop_robot(self):
        self.set_cmd(np.zeros(3), np.zeros(3), filtered=False)
        self.publish_twist(np.zeros(3), np.zeros(3))

    def cmd_loop(self):
        """Publica a taxa constante; zera se o último comando ficou velho."""
        if self._cmd_time is None:
            return
        if self._converged or (time.time() - self._cmd_time) > self.hold_time:
            self.publish_twist(np.zeros(3), np.zeros(3))
        else:
            self.publish_twist(self._cmd_lin, self._cmd_ang)

    # ------------------------------------------------------------------ auxiliares
    def saturate(self, v_lin, v_ang):
        nl, na = np.linalg.norm(v_lin), np.linalg.norm(v_ang)
        k = 1.0
        if nl > self.max_l:
            k = min(k, self.max_l / nl)
        if na > self.max_a:
            k = min(k, self.max_a / na)
        return v_lin * k, v_ang * k, k

    def resolver(self, L, e):
        if self.mu > 0:
            A = L.T @ L + (self.mu ** 2) * np.eye(6)
            return np.linalg.solve(A, L.T @ e)
        return np.linalg.pinv(L) @ e

    def gravar(self, N, err_norm, rms, cond_L, k, v_lin, v_ang, convergiu):
        if self.csv is None:
            return
        self.csv.write(
            f'{time.time() - self.t0:.3f},{N},{err_norm:.6f},{rms:.6f},{cond_L:.2f},{k:.3f},'
            f'{v_lin[0]:.5f},{v_lin[1]:.5f},{v_lin[2]:.5f},'
            f'{v_ang[0]:.5f},{v_ang[1]:.5f},{v_ang[2]:.5f},{int(convergiu)}\n')
        self.csv.flush()

    # ------------------------------------------------------------------ callback
    def features_callback(self, msg):
        if self._converged:
            return

        data = np.array(msg.data)
        if data.size < 4:
            return
        N = int(data[0])
        detectado = bool(data[1])

        if self.max_age > 0:
            stamp = Time(seconds=int(data[2]), nanoseconds=int(data[3]),
                         clock_type=self.get_clock().clock_type)
            age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
            if age > self.max_age or age < -0.05:
                self.stop_robot()
                self.get_logger().warn(f'Medida velha (idade={age:.3f}s) — parado',
                                       throttle_duration_sec=2.0)
                return

        if not detectado or N < self.min_pontos or data.size < 4 + 6 * N:
            self.stop_robot()
            self.get_logger().warn(f'Poucos pontos (N={N}) — parado', throttle_duration_sec=2.0)
            return

        b = data[4:4 + 6 * N].reshape(N, 6)
        x, y, z, xs, ys, zs = b.T
        z_ok, zs_ok = z > 0, zs > 0
        z = np.where(z_ok, z, np.where(zs_ok, zs, self.z_aprox))
        zs = np.where(zs_ok, zs, z)

        e = np.column_stack([x - xs, y - ys]).flatten()
        err_norm = np.linalg.norm(e)
        rms = err_norm / np.sqrt(N)

        err_msg = Float64MultiArray()
        err_msg.data = [float(N), float(rms)] + e.tolist()
        self.pub_error.publish(err_msg)

        if self.mode == 'desired':
            L = build_L(xs, ys, zs)
        elif self.mode == 'mean':
            L = 0.5 * (build_L(x, y, z) + build_L(xs, ys, zs))
        else:
            L = build_L(x, y, z)
        cond_L = np.linalg.cond(L)

        if rms < self.thr:
            self._below += 1
            if self._below >= self.conv_frames:
                self.stop_robot()
                self._converged = True
                self.gravar(N, err_norm, rms, cond_L, 1.0, np.zeros(3), np.zeros(3), True)
                self.get_logger().info(f'✅ CONVERGIU! rms={rms:.4f} (N={N})')
                return
        else:
            self._below = 0

        v = -self.lam * self.resolver(L, e)
        v_lin, v_ang, k = self.saturate(R_OPT2PHYS @ v[0:3], R_OPT2PHYS @ v[3:6])
        self.set_cmd(v_lin, v_ang)
        self.gravar(N, err_norm, rms, cond_L, k, self._cmd_lin, self._cmd_ang, False)

        self._log_counter += 1
        if self._log_counter % self._log_every_n == 0:
            cl, ca = self._cmd_lin, self._cmd_ang
            self.get_logger().info(
                f'{"[DRY] " if self.dry_run else ""}N={N} ||e||={err_norm:.4f} rms={rms:.4f} '
                f'cond(L)={cond_L:.1f} k={k:.2f} '
                f'v_lin=({cl[0]:+.3f},{cl[1]:+.3f},{cl[2]:+.3f}) '
                f'v_ang=({ca[0]:+.3f},{ca[1]:+.3f},{ca[2]:+.3f})')


def main(args=None):
    rclpy.init(args=args)
    node = IBVSXFeatController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_robot()
        except Exception:
            pass
        if node.csv is not None:
            node.csv.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()