#!/usr/bin/env python3
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

# ---- códigos de status publicados em /ibvs/debug ----
ST_OK, ST_LOST, ST_STALE, ST_OUTLIER, ST_CONVERGED, ST_BAD_MSG = 0, 1, 2, 3, 4, 5
Z_MEAS, Z_HOLD, Z_STAR = 0, 1, 2
# layout /ibvs/debug (21 valores):
# [status, err_norm, e(8), cond_L, sat_k, age_s, z_source, cmd(vx vy vz wx wy wz), below_count]


class IBVSController(Node):
    def __init__(self):
        super().__init__('ibvs_controller')

        # ---- s* capturado na pose alvo com a D405 REAL ----
        # ordem: TL, TR, BR, BL (coords normalizadas x, y)
        self.s_star = np.array([
            -0.29045673378065917, -0.15744387095374116,
             0.12530107198348406, -0.14686016263010895,
             0.13266376643673053,  0.2872386092862659,
            -0.3422863926517179,   0.27362307682651793,
        ])
        self.z_star = np.array([0.094, 0.094, 0.084, 0.084])

        self.declare_parameter('camera_frame', 'basic_camera_link')
        self.declare_parameter('lambda_gain', 1.2)
        self.declare_parameter('convergence_threshold', 0.02)
        self.declare_parameter('convergence_frames', 10)
        self.declare_parameter('max_lin_vel', 0.12)     # m/s
        self.declare_parameter('max_ang_vel', 0.45)     # rad/s
        self.declare_parameter('max_msg_age', 0.25)     # s
        self.declare_parameter('hold_time', 0.7)        # s — mantém último cmd entre frames
        self.declare_parameter('cmd_rate', 50.0)        # Hz — taxa enviada ao Servo
        self.declare_parameter('z_hold_time', 1.0)      # s — reusa último Z válido
        self.declare_parameter('start_delay', 0.0)      # s — espera antes de começar a agir

        self.declare_parameter('interaction_mode', 'mean')  # current | desired | mean
        self.declare_parameter('damping', 0.05)
        self.declare_parameter('cmd_filter_alpha', 0.4)
        self.declare_parameter('max_jump', 0.1)

        gp = lambda n: self.get_parameter(n).value
        self.camera_frame = gp('camera_frame')
        self.lam = gp('lambda_gain')
        self.threshold = gp('convergence_threshold')
        self.conv_frames = gp('convergence_frames')
        self.max_l = gp('max_lin_vel')
        self.max_a = gp('max_ang_vel')
        self.max_age = gp('max_msg_age')
        self.hold_time = gp('hold_time')
        self.z_hold_time = gp('z_hold_time')
        self.start_delay = gp('start_delay')
        self.mode = gp('interaction_mode')
        self.mu = gp('damping')
        self.alpha = gp('cmd_filter_alpha')
        self.max_jump = gp('max_jump')

        self._t_start = self.get_clock().now()
        self._started = self.start_delay <= 0.0
        self._last_s = None
        self._converged = False
        self._below_count = 0
        self._cmd_lin = np.zeros(3)
        self._cmd_ang = np.zeros(3)
        self._cmd_time = None
        self._last_z = None
        self._last_z_time = None
        self._log_counter = 0
        self._log_every_n = 5

        self.L_star = self.build_interaction_matrix(self.s_star, self.z_star)

        self.sub_features = self.create_subscription(
            Float64MultiArray, '/ibvs/features', self.features_callback, 10)
        self.pub_error = self.create_publisher(Float64MultiArray, '/ibvs/error', 10)
        self.pub_debug = self.create_publisher(Float64MultiArray, '/ibvs/debug', 50)
        self.pub_twist = self.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_conv = self.create_publisher(Bool, '/ibvs/converged', latched)

        self.create_timer(1.0 / gp('cmd_rate'), self.cmd_loop)

        self.get_logger().info(
            f'IBVS REAL iniciado (lambda={self.lam}, thr={self.threshold}, '
            f'vmax=({self.max_l} m/s, {self.max_a} rad/s), L_s={self.mode}, '
            f'mu={self.mu}, alpha={self.alpha}, cond(L*)={np.linalg.cond(self.L_star):.1f}, '
            f'start_delay={self.start_delay}s)')

    # ------------------------------------------------------------------
    def build_interaction_matrix(self, s, z_list):
        L = np.zeros((8, 6))
        for i in range(4):
            x, y, Z = s[2 * i], s[2 * i + 1], z_list[i]
            L[2 * i]     = [-1 / Z, 0, x / Z, x * y, -(1 + x**2), y]
            L[2 * i + 1] = [0, -1 / Z, y / Z, 1 + y**2, -x * y, -x]
        return L

    def dls_solve(self, L, e):
        A = L.T @ L + (self.mu ** 2) * np.eye(6)
        return np.linalg.solve(A, L.T @ e)

    def publish_twist(self, v_lin, v_ang):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = map(float, v_lin)
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = map(float, v_ang)
        self.pub_twist.publish(msg)

    def publish_debug(self, status, e=None, cond_L=np.nan, k=np.nan,
                      age=np.nan, z_src=np.nan):
        if e is None:
            e = np.full(8, np.nan)
            en = np.nan
        else:
            en = float(np.linalg.norm(e))
        msg = Float64MultiArray()
        msg.data = ([float(status), en] + [float(x) for x in e]
                    + [float(cond_L), float(k), float(age), float(z_src)]
                    + [float(x) for x in self._cmd_lin]
                    + [float(x) for x in self._cmd_ang]
                    + [float(self._below_count)])
        self.pub_debug.publish(msg)

    def set_cmd(self, v_lin, v_ang, filtered=True):
        if filtered and self._cmd_time is not None:
            a = self.alpha
            v_lin = a * v_lin + (1 - a) * self._cmd_lin
            v_ang = a * v_ang + (1 - a) * self._cmd_ang
        self._cmd_lin = v_lin
        self._cmd_ang = v_ang
        self._cmd_time = self.get_clock().now()

    def stop_robot(self):
        self.set_cmd(np.zeros(3), np.zeros(3), filtered=False)
        self.publish_twist(np.zeros(3), np.zeros(3))

    def cmd_loop(self):
        if self._cmd_time is None:
            return
        if self._converged:
            self.publish_twist(np.zeros(3), np.zeros(3))
            return
        dt = (self.get_clock().now() - self._cmd_time).nanoseconds * 1e-9
        if dt > self.hold_time:
            self.publish_twist(np.zeros(3), np.zeros(3))
            self.get_logger().warn(
                f'Sem features há {dt:.2f}s — parado', throttle_duration_sec=2.0)
        else:
            self.publish_twist(self._cmd_lin, self._cmd_ang)

    def saturate(self, v_lin, v_ang):
        nl, na = np.linalg.norm(v_lin), np.linalg.norm(v_ang)
        k = 1.0
        if nl > self.max_l:
            k = min(k, self.max_l / nl)
        if na > self.max_a:
            k = min(k, self.max_a / na)
        return v_lin * k, v_ang * k, k

    # ------------------------------------------------------------------
    def features_callback(self, msg):
        if self._converged:
            return

        # ---- espera inicial (logger preparar TF/juntas) ----
        if not self._started:
            if (self.get_clock().now() - self._t_start).nanoseconds * 1e-9 < self.start_delay:
                return
            self._started = True
            self.get_logger().info('▶ start_delay concluído — IBVS ativo')

        data = np.array(msg.data)
        if data.size < 16:
            self.publish_debug(ST_BAD_MSG)
            self.get_logger().error(f'Mensagem com {data.size} campos (esperado 16)',
                                    throttle_duration_sec=2.0)
            return

        s = data[:8]
        z_list = data[8:12]
        marker_detected = bool(data[12])
        depth_valid = bool(data[13])
        stamp = Time(seconds=int(data[14]), nanoseconds=int(data[15]),
                     clock_type=self.get_clock().clock_type)

        now = self.get_clock().now()
        age = (now - stamp).nanoseconds * 1e-9

        if not marker_detected:
            self.stop_robot()
            self.publish_debug(ST_LOST, age=age)
            self.get_logger().warn('Marcador não detectado — parado',
                                   throttle_duration_sec=2.0)
            return

        if age > self.max_age or age < -0.05:
            self.stop_robot()
            self.publish_debug(ST_STALE, age=age)
            self.get_logger().warn(f'Medida velha (idade={age:.3f}s)',
                                   throttle_duration_sec=2.0)
            return

        if depth_valid and np.all(z_list > 0.05) and np.all(z_list < 1.0):
            self._last_z = z_list.copy()
            self._last_z_time = now
            z_src = Z_MEAS
        elif self._last_z is not None and \
                (now - self._last_z_time).nanoseconds * 1e-9 < self.z_hold_time:
            z_list = self._last_z
            z_src = Z_HOLD
            self.get_logger().warn('Depth inválido — usando último Z válido',
                                   throttle_duration_sec=2.0)
        else:
            z_list = self.z_star
            z_src = Z_STAR
            self.get_logger().warn('Depth inválido há muito tempo — usando Z*',
                                   throttle_duration_sec=2.0)

        e = s - self.s_star
        err_norm = np.linalg.norm(e)

        # ---- rejeita outlier: salto brusco entre frames consecutivos ----
        if self._last_s is not None and np.linalg.norm(s - self._last_s) > self.max_jump:
            self.publish_debug(ST_OUTLIER, e, age=age, z_src=z_src)
            self.get_logger().warn(
                f'Salto de {np.linalg.norm(s - self._last_s):.3f} entre frames — descartado',
                throttle_duration_sec=1.0)
            return
        self._last_s = s.copy()

        err_msg = Float64MultiArray()
        err_msg.data = e.tolist()
        self.pub_error.publish(err_msg)

        if err_norm < self.threshold:
            self._below_count += 1
            if self._below_count >= self.conv_frames:
                self.stop_robot()
                self._converged = True
                self.publish_debug(ST_CONVERGED, e, age=age, z_src=z_src)
                self.pub_conv.publish(Bool(data=True))
                self.get_logger().info(f'✅ CONVERGIU! ||e||={err_norm:.4f}')
                return
        else:
            self._below_count = 0

        if self.mode == 'desired':
            L = self.L_star
        elif self.mode == 'mean':
            L = 0.5 * (self.build_interaction_matrix(s, z_list) + self.L_star)
        else:
            L = self.build_interaction_matrix(s, z_list)

        cond_L = np.linalg.cond(L)
        # ----------- A lei de controle --------------------
        v = -self.lam * self.dls_solve(L, e)

        v_lin, v_ang, k = self.saturate(R_OPT2PHYS @ v[0:3], R_OPT2PHYS @ v[3:6])
        self.set_cmd(v_lin, v_ang)
        self.publish_debug(ST_OK, e, cond_L, k, age, z_src)

        self._log_counter += 1
        if self._log_counter % self._log_every_n == 0:
            cl, ca = self._cmd_lin, self._cmd_ang
            self.get_logger().info(
                f'||e||={err_norm:.4f} cond(L)={cond_L:.1f} idade={age*1000:.0f}ms k={k:.2f} '
                f'v_lin=({cl[0]:+.3f},{cl[1]:+.3f},{cl[2]:+.3f}) '
                f'v_ang=({ca[0]:+.3f},{ca[1]:+.3f},{ca[2]:+.3f})')


def main(args=None):
    rclpy.init(args=args)
    node = IBVSController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_robot()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()