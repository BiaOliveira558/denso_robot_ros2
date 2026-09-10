#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped
import numpy as np
from denso_scripts.csv_logger import CsvLogger
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8


CSV_COLUMNS = [
    'timestamp', 't_relative',
    'e_0', 'e_1', 'e_2', 'e_3', 'e_4', 'e_5', 'e_6', 'e_7',
    'z_0', 'z_1', 'z_2', 'z_3',
    'err_norm', 'cond_L_s',
    'marker_detected', 'depth_valid',
    'v_lin_pre_clip_x', 'v_lin_pre_clip_y', 'v_lin_pre_clip_z',
    'v_ang_pre_clip_x', 'v_ang_pre_clip_y', 'v_ang_pre_clip_z',
    'v_lin_phys_x', 'v_lin_phys_y', 'v_lin_phys_z',
    'v_ang_phys_x', 'v_ang_phys_y', 'v_ang_phys_z',
    'joint_pos_0', 'joint_pos_1', 'joint_pos_2', 'joint_pos_3', 'joint_pos_4', 'joint_pos_5',
    'joint_vel_0', 'joint_vel_1', 'joint_vel_2', 'joint_vel_3', 'joint_vel_4', 'joint_vel_5',
    'joint_eff_0', 'joint_eff_1', 'joint_eff_2', 'joint_eff_3', 'joint_eff_4', 'joint_eff_5',
    'joint_state_age_s',
    'servo_status', 'servo_status_age_s',
    'feature_age_s', 'converged',
    'image_stamp_sec', 'image_stamp_nanosec',
]

CAMERA_FRAME = 'basic_camera_link'
CMD_VEL_TOPIC = '/servo_node/delta_twist_cmds'

MAX_L_VEL = 0.15   # m/s, mesmo limite do PBVS
MAX_A_VEL = 0.4    # rad/s, mesmo limite do PBVS

# mesma transformação óptico → físico usada no aruco_pose_estimator.py
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
            -0.18662198846537845, 0.7014412669905606,
            -0.1480105425759898, 0.20914533190085524,
            0.2187981933732023, 0.2413215368086791,
            0.27993298269806766, 0.6917884055182134
        ])

        self.declare_parameter('lambda_gain', 0.25)
        self.declare_parameter('convergence_threshold', 0.05)
        self.lam = self.get_parameter('lambda_gain').value
        self.threshold = self.get_parameter('convergence_threshold').value
        self.last_features_time = None  # timestamp da última mensagem em /ibvs/features

        self._converged = False

        self.sub_features = self.create_subscription(
            Float64MultiArray, '/ibvs/features', self.features_callback, 10)

        self.pub_error = self.create_publisher(
            Float64MultiArray, '/ibvs/error', 10)

        self.pub_twist = self.create_publisher(
            TwistStamped, CMD_VEL_TOPIC, 10)

        self.last_servo_status = None
        self.last_servo_status_time = None

        self.sub_servo_status = self.create_subscription(
            Int8, '/servo_node/status', self._servo_status_cb, 10)

        self._log_counter = 0
        self._log_every_n = 5

        self.csv_logger = CsvLogger(
            output_dir='/root/denso_ws/logs/ibvs',
            prefix='ibvs_run',
            columns=CSV_COLUMNS,
        )
        self._start_time = self.get_clock().now()

        # "último estado conhecido" — usado tanto pelo callback normal quanto pelo health check
        self._last_row = {col: '' for col in CSV_COLUMNS}
        self.last_features_time = None
        self.last_joint_state = None
        self.last_joint_state_time = None

        self.sub_joint_state = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10)

        self.health_check_timer = self.create_timer(0.5, self.health_check_callback)

        self.get_logger().info(
            f'IBVSController iniciado (lambda={self.lam}, threshold={self.threshold})')

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

    def get_joint_state_fields(self, now):
        if self.last_joint_state is None:
            return {
                'joint_pos_0': '', 'joint_pos_1': '', 'joint_pos_2': '',
                'joint_pos_3': '', 'joint_pos_4': '', 'joint_pos_5': '',
                'joint_vel_0': '', 'joint_vel_1': '', 'joint_vel_2': '',
                'joint_vel_3': '', 'joint_vel_4': '', 'joint_vel_5': '',
                'joint_eff_0': '', 'joint_eff_1': '', 'joint_eff_2': '',
                'joint_eff_3': '', 'joint_eff_4': '', 'joint_eff_5': '',
                'joint_state_age_s': '',
            }

        pos = self.last_joint_state.position
        vel = self.last_joint_state.velocity
        eff = self.last_joint_state.effort
        age = (now - self.last_joint_state_time).nanoseconds / 1e9

        fields = {'joint_state_age_s': age}
        for i in range(6):
            fields[f'joint_pos_{i}'] = pos[i] if i < len(pos) else ''
            fields[f'joint_vel_{i}'] = vel[i] if i < len(vel) else ''
            fields[f'joint_eff_{i}'] = eff[i] if i < len(eff) else ''
        return fields
    
    def _servo_status_cb(self, msg):
        self.last_servo_status = msg
        self.last_servo_status_time = self.get_clock().now()

    def stop_robot(self):
        self.publish_twist(np.zeros(3), np.zeros(3))

    def _joint_state_cb(self, msg):
        self.last_joint_state = msg
        self.last_joint_state_time = self.get_clock().now()

    def get_servo_status_fields(self, now):
        if self.last_servo_status is None:
            return {'servo_status': '', 'servo_status_age_s': ''}

        age = (now - self.last_servo_status_time).nanoseconds / 1e9
        return {
            'servo_status': self.last_servo_status.data,
            'servo_status_age_s': age,
        }
    
    def health_check_callback(self):
        if self.last_features_time is None:
            return

        now = self.get_clock().now()
        feature_age_s = (now - self.last_features_time).nanoseconds / 1e9

        if feature_age_s <= 1.0:
            return

        frozen_row = dict(self._last_row)
        frozen_row['timestamp'] = now.nanoseconds
        frozen_row['t_relative'] = (now - self._start_time).nanoseconds / 1e9
        frozen_row['feature_age_s'] = feature_age_s
        frozen_row.update(self.get_joint_state_fields(now))
        frozen_row.update(self.get_servo_status_fields(now))
        self.csv_logger.log(frozen_row)

        self.get_logger().warn(
            f'⚠️ Sem mensagens em /ibvs/features há {feature_age_s:.2f}s (dado congelado)')

    def destroy_node(self):
        self.csv_logger.close()
        super().destroy_node()

    def features_callback(self, msg):
        self.last_features_time = self.get_clock().now()

        if self._converged:
            return

        data = np.array(msg.data)
        s = data[:8]
        z_list = data[8:12]
        marker_detected = bool(data[12])
        depth_valid = bool(data[13])
        image_stamp_sec = data[14]
        image_stamp_nanosec = data[15]

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
            t_relative = (self.last_features_time - self._start_time).nanoseconds / 1e9
            self._last_row = dict(self._last_row)
            self._last_row['timestamp'] = self.last_features_time.nanoseconds
            self._last_row['t_relative'] = t_relative
            self._last_row['e_0'], self._last_row['e_1'] = e[0], e[1]
            self._last_row['e_2'], self._last_row['e_3'] = e[2], e[3]
            self._last_row['e_4'], self._last_row['e_5'] = e[4], e[5]
            self._last_row['e_6'], self._last_row['e_7'] = e[6], e[7]
            self._last_row['z_0'], self._last_row['z_1'] = z_list[0], z_list[1]
            self._last_row['z_2'], self._last_row['z_3'] = z_list[2], z_list[3]
            self._last_row['err_norm'] = err_norm
            self._last_row['marker_detected'] = marker_detected
            self._last_row['depth_valid'] = depth_valid
            self._last_row['feature_age_s'] = 0.0
            self._last_row['image_stamp_sec'] = image_stamp_sec
            self._last_row['image_stamp_nanosec'] = image_stamp_nanosec
            self._last_row.update(self.get_joint_state_fields(self.last_features_time))
            self._last_row.update(self.get_servo_status_fields(self.last_features_time))
            self.csv_logger.log(self._last_row)
            return

        # ---- checagem de convergência ----
        if err_norm < self.threshold:
            self.stop_robot()
            self._converged = True
            self.get_logger().info(
                f'✅ CONVERGIU! ||e||={err_norm:.4f} < {self.threshold}')

            t_relative = (self.last_features_time - self._start_time).nanoseconds / 1e9
            self._last_row = dict(self._last_row)
            self._last_row['timestamp'] = self.last_features_time.nanoseconds
            self._last_row['t_relative'] = t_relative
            self._last_row['e_0'], self._last_row['e_1'] = e[0], e[1]
            self._last_row['e_2'], self._last_row['e_3'] = e[2], e[3]
            self._last_row['e_4'], self._last_row['e_5'] = e[4], e[5]
            self._last_row['e_6'], self._last_row['e_7'] = e[6], e[7]
            self._last_row['z_0'], self._last_row['z_1'] = z_list[0], z_list[1]
            self._last_row['z_2'], self._last_row['z_3'] = z_list[2], z_list[3]
            self._last_row['err_norm'] = err_norm
            self._last_row['marker_detected'] = marker_detected
            self._last_row['depth_valid'] = depth_valid
            self._last_row['feature_age_s'] = 0.0
            self._last_row['converged'] = True
            self._last_row['image_stamp_sec'] = image_stamp_sec
            self._last_row['image_stamp_nanosec'] = image_stamp_nanosec
            self._last_row.update(self.get_joint_state_fields(self.last_features_time))
            self._last_row.update(self.get_servo_status_fields(self.last_features_time))
            self.csv_logger.log(self._last_row)
            return

        L_s = self.build_interaction_matrix(s, z_list)
        cond_L_s = np.linalg.cond(L_s)
        L_s_pinv = np.linalg.pinv(L_s)

        v = -self.lam * (L_s_pinv @ e)   # frame óptico da câmera
        # temporário, só pra diagnosticar:
        self.get_logger().info(f'v_pre_clip norm = {np.linalg.norm(v):.5f}')
        v_lin_opt = v[0:3]
        v_ang_opt = v[3:6]

        v_lin_phys = R_OPT2PHYS @ v_lin_opt
        v_ang_phys = R_OPT2PHYS @ v_ang_opt

        v_lin_phys = np.clip(v_lin_phys, -MAX_L_VEL, MAX_L_VEL)
        v_ang_phys = np.clip(v_ang_phys, -MAX_A_VEL, MAX_A_VEL)

        self.publish_twist(v_lin_phys, v_ang_phys)

        self._log_counter += 1
        if self._log_counter % self._log_every_n == 0:
            self.get_logger().info(
                f'||e||={err_norm:.4f}  cond(L_s)={cond_L_s:.2f}  '
                f'v_lin=({v_lin_phys[0]:+.3f},{v_lin_phys[1]:+.3f},{v_lin_phys[2]:+.3f})  '
                f'v_ang=({v_ang_phys[0]:+.3f},{v_ang_phys[1]:+.3f},{v_ang_phys[2]:+.3f})'
            )

        t_relative = (self.last_features_time - self._start_time).nanoseconds / 1e9

        self._last_row = {
            'timestamp': self.last_features_time.nanoseconds,
            't_relative': t_relative,
            'e_0': e[0], 'e_1': e[1], 'e_2': e[2], 'e_3': e[3],
            'e_4': e[4], 'e_5': e[5], 'e_6': e[6], 'e_7': e[7],
            'z_0': z_list[0], 'z_1': z_list[1], 'z_2': z_list[2], 'z_3': z_list[3],
            'err_norm': err_norm,
            'cond_L_s': cond_L_s,
            'marker_detected': marker_detected,
            'depth_valid': depth_valid,
            'v_lin_pre_clip_x': v[0], 'v_lin_pre_clip_y': v[1], 'v_lin_pre_clip_z': v[2],
            'v_ang_pre_clip_x': v[3], 'v_ang_pre_clip_y': v[4], 'v_ang_pre_clip_z': v[5],
            'v_lin_phys_x': v_lin_phys[0], 'v_lin_phys_y': v_lin_phys[1], 'v_lin_phys_z': v_lin_phys[2],
            'v_ang_phys_x': v_ang_phys[0], 'v_ang_phys_y': v_ang_phys[1], 'v_ang_phys_z': v_ang_phys[2],
            'feature_age_s': 0.0,
            'converged': self._converged,
            'image_stamp_sec': image_stamp_sec,
            'image_stamp_nanosec': image_stamp_nanosec,
        }
        self._last_row.update(self.get_joint_state_fields(self.last_features_time))
        self._last_row.update(self.get_servo_status_fields(self.last_features_time))
        self.csv_logger.log(self._last_row)

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