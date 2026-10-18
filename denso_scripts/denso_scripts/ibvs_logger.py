#!/usr/bin/env python3
"""
Logger de experimentos IBVS.
Grava data.csv (1 linha por frame) e final_image.png (POV ao convergir).
Após convergir, continua gravando por settle_time (status SETTLING):
a última linha do CSV é a pose final com o robô parado.
"""
import os
import csv
import math
from datetime import datetime

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener

CORNERS = ['TL', 'TR', 'BR', 'BL']
STATUS = {0: 'OK', 1: 'MARKER_LOST', 2: 'STALE', 3: 'OUTLIER', 4: 'CONVERGED', 5: 'BAD_MSG'}


def fmt(v):
    return '' if (v is None or math.isnan(v)) else f'{v:.6f}'


class IBVSLogger(Node):
    def __init__(self):
        super().__init__('ibvs_logger')

        self.declare_parameter('log_dir', os.path.expanduser('~/ibvs_logs'))
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'basic_camera_link')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('joint_names',
                               ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('settle_time', 1.0)    # s gravando após convergir
        self.declare_parameter('settle_rate', 10.0)   # Hz durante o settling
        self.declare_parameter('timeout', 120.0)      # s sem convergir = falha
        self.declare_parameter('tag', '')

        gp = lambda n: self.get_parameter(n).value
        self.base_frame = gp('base_frame')
        self.camera_frame = gp('camera_frame')
        self.joint_names = list(gp('joint_names'))
        self.settle_time = gp('settle_time')
        self.settle_rate = gp('settle_rate')
        self.timeout = gp('timeout')

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = gp('tag')
        self.run_dir = os.path.join(gp('log_dir'), f'run_{stamp}' + (f'_{tag}' if tag else ''))
        os.makedirs(self.run_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.csv_file = open(os.path.join(self.run_dir, 'data.csv'), 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(
            ['t_ros', 't_rel', 'status', 'err_norm']
            + [f'e_{a}_{c}' for c in CORNERS for a in ('x', 'y')]
            + ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
            + [f'q_{n}' for n in self.joint_names]
            + ['px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw']
            + ['converged'])

        self._last_img = None
        self._joints = np.full(len(self.joint_names), np.nan)
        self._t0 = None
        self._converged = False
        self._settle_start = None
        self._settle_timer = None
        self._finished = False
        self.done = False

        self.create_subscription(Image, gp('image_topic'), self.image_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(Float64MultiArray, '/ibvs/debug', self.debug_cb, 50)
        self.create_timer(0.5, self.check_timeout)

        self.get_logger().info(f'Logger pronto — gravando em {self.run_dir}')

    # ------------------------------------------------------------------
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def lookup_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, Time())
        except Exception as ex:
            self.get_logger().warn(f'TF indisponível: {ex}', throttle_duration_sec=2.0)
            return np.full(7, np.nan)
        t, q = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w])

    def image_cb(self, msg):
        self._last_img = msg

    def joint_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}   # lê pelo NOME (ordem do tópico é embaralhada)
        for k, n in enumerate(self.joint_names):
            if n in idx:
                self._joints[k] = msg.position[idx[n]]

    def write_row(self, status, err_norm, e, cmd, converged):
        t = self.now_s()
        if self._t0 is None:
            self._t0 = t
        pose = self.lookup_pose()
        self.writer.writerow(
            [f'{t:.6f}', f'{t - self._t0:.6f}', status, fmt(err_norm)]
            + [fmt(v) for v in e]
            + [fmt(v) for v in cmd]
            + [fmt(v) for v in self._joints]
            + [fmt(v) for v in pose]
            + [int(converged)])
        self.csv_file.flush()

    # ------------------------------------------------------------------
    def debug_cb(self, msg):
        if self._converged or self._finished:
            return
        d = np.array(msg.data, dtype=float)
        if d.size < 21:
            return

        status = STATUS.get(int(d[0]), 'UNKNOWN')
        converged = int(d[0]) == 4
        self.write_row(status, d[1], d[2:10], d[14:20], converged)

        if converged:
            self._converged = True
            self.get_logger().info(
                f'✅ Convergiu em {self.now_s() - self._t0:.2f}s — '
                f'gravando settling por {self.settle_time}s')
            self._settle_start = self.now_s()
            self._settle_timer = self.create_timer(1.0 / self.settle_rate, self.settle_tick)

    def settle_tick(self):
        if self._finished:
            return
        self.write_row('SETTLING', np.nan, np.full(8, np.nan), np.zeros(6), True)
        if self.now_s() - self._settle_start >= self.settle_time:
            self.finalize()

    def check_timeout(self):
        if self._finished or self._converged or self._t0 is None:
            return
        if self.now_s() - self._t0 > self.timeout:
            self.get_logger().warn(f'⏱ Timeout de {self.timeout}s sem convergir — FALHA')
            self.finalize()

    def finalize(self):
        if self._finished:
            return
        self._finished = True
        if self._settle_timer is not None:
            self._settle_timer.cancel()

        if self._converged and self._last_img is not None:
            img = self.bridge.imgmsg_to_cv2(self._last_img, desired_encoding='bgr8')
            cv2.imwrite(os.path.join(self.run_dir, 'final_image.png'), img)

        self.csv_file.close()
        self.get_logger().info(
            f'Run finalizado ({"CONVERGIU" if self._converged else "NÃO convergiu"}) — {self.run_dir}')
        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = IBVSLogger()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._finished:
            node.finalize()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()