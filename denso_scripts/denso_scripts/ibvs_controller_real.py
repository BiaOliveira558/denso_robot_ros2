#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped
import numpy as np


CAMERA_FRAME = 'camera_link'
CMD_VEL_TOPIC = '/servo_node/delta_twist_cmds'

MAX_L_VEL = 0.03   # m/s — início conservador
MAX_A_VEL = 0.1

# transformação óptico → físico (mesma usada no aruco_pose_estimator.py)
R_OPT2PHYS = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)


class IBVSController(Node):
    def __init__(self):
        super().__init__('ibvs_controller')

        # ---- s* capturado na pose alvo (fixo) ----
        self.s_star = np.array([
            0.11583916393110873, -0.198758746710332,
            0.422322932261142, -0.21734974579427083,
            0.42464477899091496, 0.0847539893197354,
            0.13209209103951958, 0.10334498840367426
        ])

        self.declare_parameter('lambda_gain', 0.25)
        self.declare_parameter('convergence_threshold', 0.05)
        self.lam = self.get_parameter('lambda_gain').value
        self.threshold = self.get_parameter('convergence_threshold').value

        self._converged = False
        self._log_counter = 0
        self._log_every_n = 5

        self.sub_features = self.create_subscription(
            Float64MultiArray, '/ibvs/features', self.features_callback, 10)

        self.pub_error = self.create_publisher(
            Float64MultiArray, '/ibvs/error', 10)

        self.pub_twist = self.create_publisher(
            TwistStamped, CMD_VEL_TOPIC, 10)

        self.get_logger().info(
            f'IBVSController iniciado (lambda={self.lam}, threshold={self.threshold}) '
            f'[TESTE DE ISOLAMENTO: v_ang zerado antes de publicar]')

    def build_interaction_matrix(self, s, z_list):
        L = np.zeros((8, 6))
        for i in range(4):
            x = s[2 * i]
            y = s[2 * i + 1]
            Z = z_list[i]

            L[2 * i]     = [-1 / Z, 0, x / Z, x * y, -(1 + x**2), y]
            L[2 * i + 1] = [0, -1 / Z, y / Z, 1 + y**2, -x * y, -x]

        return L

    def publish_twist(self, v_lin, v_ang):
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = CAMERA_FRAME
        twist_msg.twist.linear.x = float(v_lin[0])
        twist_msg.twist.linear.y = float(v_lin[1])
        twist_msg.twist.linear.z = float(v_lin[2])
        twist_msg.twist.angular.x = float(v_ang[0])
        twist_msg.twist.angular.y = float(v_ang[1])
        twist_msg.twist.angular.z = float(v_ang[2])
        self.pub_twist.publish(twist_msg)

    def stop_robot(self):
        self.publish_twist(np.zeros(3), np.zeros(3))

    def features_callback(self, msg):
        if self._converged:
            return

        data = np.array(msg.data)
        s = data[:8]
        z_list = data[8:12]
        marker_detected = bool(data[12])
        depth_valid = bool(data[13])

        e = s - self.s_star
        err_norm = np.linalg.norm(e)

        error_msg = Float64MultiArray()
        error_msg.data = e.tolist()
        self.pub_error.publish(error_msg)

        # ---- gate: só segue se detecção e profundidade forem confiáveis ----
        if not (marker_detected and depth_valid):
            self.get_logger().warn(
                f'Dados não confiáveis (detected={marker_detected}, depth_valid={depth_valid}) — pulando ciclo de controle',
                throttle_duration_sec=2.0)
            return

        # ---- checagem de convergência ----
        if err_norm < self.threshold:
            self.stop_robot()
            self._converged = True
            self.get_logger().info(
                f'✅ CONVERGIU! ||e||={err_norm:.4f} < {self.threshold}')
            return

        L_s = self.build_interaction_matrix(s, z_list)
        cond_L_s = np.linalg.cond(L_s)
        L_s_pinv = np.linalg.pinv(L_s)

        v = -self.lam * (L_s_pinv @ e)   # frame óptico da câmera
        v_lin_opt = v[0:3]
        v_ang_opt = v[3:6]

        v_lin_phys = R_OPT2PHYS @ v_lin_opt
        v_ang_phys = R_OPT2PHYS @ v_ang_opt

        v_lin_phys = np.clip(v_lin_phys, -MAX_L_VEL, MAX_L_VEL)
        v_ang_phys = np.clip(v_ang_phys, -MAX_A_VEL, MAX_A_VEL)

        # ---- TESTE DE ISOLAMENTO: publica só a parte linear, angular zerada ----
        self.publish_twist(v_lin_phys, np.zeros(3))

        self._log_counter += 1
        if self._log_counter % self._log_every_n == 0:
            self.get_logger().info(
                f'||e||={err_norm:.4f}  cond(L_s)={cond_L_s:.2f}  '
                f'v_lin=({v_lin_phys[0]:+.3f},{v_lin_phys[1]:+.3f},{v_lin_phys[2]:+.3f})  '
                f'v_ang_calc=({v_ang_phys[0]:+.3f},{v_ang_phys[1]:+.3f},{v_ang_phys[2]:+.3f}) [NÃO enviado]'
            )


def main(args=None):
    rclpy.init(args=args)
    node = IBVSController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()