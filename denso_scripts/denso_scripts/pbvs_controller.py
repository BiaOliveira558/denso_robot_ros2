#!/usr/bin/env python3

import math
import time
import csv
import os
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped

# ── Configurações ─────────────────────────────────────────────────────────
PRINT_HZ        = 5
LOOP_HZ         = 50
CAMERA_FRAME    = 'basic_camera_link'
ARUCO_TOPIC     = '/aruco/pose'
CMD_VEL_TOPIC   = '/servo_node/delta_twist_cmds'
POSE_TIMEOUT_S  = 0.5

# ── Ganhos e limites do controlador ────────────────────────────────────────
KP_LINEAR  = 0.5
KP_ANGULAR = 0.4
MAX_L_VEL  = 0.15
MAX_A_VEL  = 0.4
# ─────────────────────────────────────────────────────────────────────────


class PBVSController(Node):

    def __init__(self):
        super().__init__('pbvs_controller')

        # ── Parâmetros de log CSV ────────────────────────────────────────
        self.declare_parameter('save_csv', False)
        self.declare_parameter('csv_path', '/root/denso_ws/logs/pbvs_log.csv')

        self._save_csv = self.get_parameter('save_csv').value
        self._csv_path = self.get_parameter('csv_path').value
        self._csv_file = None
        self._csv_writer = None
        self._t0 = None

        if self._save_csv:
            self._init_csv()

        # ── Subscriber da pose do marcador (ArUco) ──────────────────────
        self._marker_pos     = None
        self._marker_rot     = None
        self._last_marker_ts = None
        self.create_subscription(PoseStamped, ARUCO_TOPIC, self._marker_pose_cb, 10)

        # ── Publisher de Velocidade ──────────────────────────────────────
        self._cmd_pub = self.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)

        # ── Estado interno ────────────────────────────────────────────────
        self._goal_pos     = None
        self._goal_rot     = None
        self._active       = False
        self._iter         = 0
        self._pose_warned  = False

        self.create_timer(1.0 / LOOP_HZ, self._loop)

        print('[CONTROLE] Nó iniciado', flush=True)
        if self._save_csv:
            print(f'[CONTROLE] Log CSV ATIVADO → {self._csv_path}', flush=True)
        else:
            print('[CONTROLE] Log CSV desativado (save_csv:=false)', flush=True)

    # ─────────────────────────────────────────── CSV ──────────────────────

    def _init_csv(self):
        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)

        if os.path.exists(self._csv_path):
            base, ext = os.path.splitext(self._csv_path)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._csv_path = f'{base}_{stamp}{ext}'

        self._csv_file = open(self._csv_path, mode='w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'time_s', 'iter',
            't_err_x', 't_err_y', 't_err_z', 't_norm',
            'r_err_x', 'r_err_y', 'r_err_z', 'r_norm_deg',
            'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z',
        ])
        self._csv_file.flush()

    def _log_csv(self, t_err, t_norm, r_err, r_norm, v_cmd, w_cmd):
        if not self._save_csv or self._csv_writer is None:
            return

        if self._t0 is None:
            self._t0 = self.get_clock().now()
        elapsed_s = (self.get_clock().now() - self._t0).nanoseconds / 1e9

        self._csv_writer.writerow([
            f'{elapsed_s:.4f}', self._iter,
            f'{t_err[0]:.6f}', f'{t_err[1]:.6f}', f'{t_err[2]:.6f}', f'{t_norm:.6f}',
            f'{r_err[0]:.6f}', f'{r_err[1]:.6f}', f'{r_err[2]:.6f}', f'{math.degrees(r_norm):.4f}',
            f'{v_cmd[0]:.6f}', f'{v_cmd[1]:.6f}', f'{v_cmd[2]:.6f}',
            f'{w_cmd[0]:.6f}', f'{w_cmd[1]:.6f}', f'{w_cmd[2]:.6f}',
        ])
        self._csv_file.flush()

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.close()
            print(f'[CONTROLE] CSV salvo em {self._csv_path}', flush=True)

    # ─────────────────────────────────────────── Callbacks ────────────────

    def _marker_pose_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        self._marker_pos = np.array([p.x, p.y, p.z])
        self._marker_rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        self._last_marker_ts = self.get_clock().now()

    # ─────────────────────────────────────────── API pública ──────────────

    def set_goal_from_quat(
        self,
        x: float, y: float, z: float,
        qx: float, qy: float, qz: float, qw: float,
    ) -> None:

        self._goal_pos = np.array([float(x), float(y), float(z)])
        self._goal_rot = Rotation.from_quat(
            [float(qx), float(qy), float(qz), float(qw)]
        ).as_matrix()
        self._active      = True
        self._iter         = 0
        self._pose_warned  = False
        self._t0            = None

        print(
            f'\n[CONTROLE] Goal (marcador relativo à câmera, via quaternion bruto) → '
            f'pos=({x:.4f}, {y:.4f}, {z:.4f}) m  '
            f'quat=({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})',
            flush=True
        )

        header = f'{"ciclo":>6} | {"et(m)":>7} | {"eang(°)":>7} | {"marker":>7}'
        print(header, flush=True)
        print('─' * 40, flush=True)

    def stop(self) -> None:
        self._active = False
        self.stop_robot()
        self._close_csv()
        print('[CONTROLE] Laço de controle pausado e robô parado.', flush=True)

    def stop_robot(self) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = CAMERA_FRAME
        self._cmd_pub.publish(msg)

    # ─────────────────────────────────────────── Cálculo de erro ──────────

    def compute_error(self):
        if self._marker_pos is None or self._goal_pos is None:
            return None

        R_goal_cam = self._goal_rot @ self._marker_rot.T
        R_cam_goal = R_goal_cam.T

        r_err  = Rotation.from_matrix(R_goal_cam).as_rotvec()
        r_norm = float(np.linalg.norm(r_err))

        goal_pos_in_current_cam = R_cam_goal @ self._goal_pos
        t_err  = goal_pos_in_current_cam - self._marker_pos
        t_norm = float(np.linalg.norm(t_err))

        return t_err, t_norm, r_err, r_norm

    # ─────────────────────────────────────────── Lei de controle ──────────

    def compute_control(self, t_err: np.ndarray, r_err: np.ndarray):
        v_cmd = -KP_LINEAR * t_err
        w_cmd = -KP_ANGULAR * r_err

        v_cmd = np.clip(v_cmd, -MAX_L_VEL, MAX_L_VEL)
        w_cmd = np.clip(w_cmd, -MAX_A_VEL, MAX_A_VEL)

        return v_cmd, w_cmd

    # ─────────────────────────────────────────── Laço principal ───────────

    def _loop(self) -> None:
        if not self._active:
            return

        if self._marker_pos is None:
            if not self._pose_warned:
                print('[CONTROLE] Aguardando primeira leitura de /aruco/pose...', flush=True)
                self._pose_warned = True
            return

        age_s = (self.get_clock().now() - self._last_marker_ts).nanoseconds / 1e9
        marker_ok = age_s < POSE_TIMEOUT_S
        marker_str = 'OK' if marker_ok else 'PERDIDO'

        if not marker_ok:
            if not self._pose_warned:
                print(f'[CONTROLE] ⚠ Marcador PERDIDO há {age_s:.2f}s! Parando robô.', flush=True)
                self._pose_warned = True
                self.stop_robot()
            return
        self._pose_warned = False

        result = self.compute_error()
        if result is None:
            return
        t_err, t_norm, r_err, r_norm = result

        if t_norm < 0.005 and r_norm < math.radians(1.5):
            self.stop_robot()
            self._log_csv(t_err, t_norm, r_err, r_norm, np.zeros(3), np.zeros(3))
            print("[CONTROLE] Braço posicionado no GOAL com precisão!", flush=True)
            self._active = False
            self._close_csv()
            return

        v_cmd, w_cmd = self.compute_control(t_err, r_err)

        print(
            f'[TWIST→ROBÔ] '
            f'v=({v_cmd[0]:+.4f}, {v_cmd[1]:+.4f}, {v_cmd[2]:+.4f}) m/s  '
            f'ω=({w_cmd[0]:+.4f}, {w_cmd[1]:+.4f}, {w_cmd[2]:+.4f}) rad/s  '
            f'|  et={t_norm:.3f} m  eang={math.degrees(r_norm):.2f}°',
            flush=True
        )

        self._log_csv(t_err, t_norm, r_err, r_norm, v_cmd, w_cmd)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = CAMERA_FRAME
        twist_msg.twist.linear.x = float(v_cmd[0])
        twist_msg.twist.linear.y = float(v_cmd[1])
        twist_msg.twist.linear.z = float(v_cmd[2])
        twist_msg.twist.angular.x = float(w_cmd[0])
        twist_msg.twist.angular.y = float(w_cmd[1])
        twist_msg.twist.angular.z = float(w_cmd[2])

        self._cmd_pub.publish(twist_msg)

        self._iter += 1
        if self._iter % max(1, (LOOP_HZ // PRINT_HZ)) == 0:
            print(
                f'{self._iter:>6} | '
                f'{t_norm:>7.3f} | '
                f'{math.degrees(r_norm):>7.2f} | '
                f'{marker_str:>7}',
                flush=True
            )


def main(args=None):
    rclpy.init(args=args)
    node = PBVSController()

    time.sleep(1.5)

    node.set_goal_from_quat(
        x=0.08916009571379095, y=-0.004867996375355384, z=-0.014628429185391114,
        qx=0.30693724802770034, qy=-0.47890568632771596, qz=-0.20143601541492778, qw=0.7974098074829016
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n[CONTROLE] Interrompido pelo usuário.', flush=True)
    finally:
        try:
            node.stop()
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()