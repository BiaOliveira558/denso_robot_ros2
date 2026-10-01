#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
from cv_bridge import CvBridge


class ArucoFeatureExtractor(Node):
    def __init__(self):
        super().__init__('aruco_feature_extractor')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('marker_id', -1)          # -1 = qualquer (pega o primeiro)
        self.declare_parameter('depth_window', 5)        # meia-largura da janela de mediana (px)
        self.declare_parameter('depth_inset', 0.15)      # fração do canto em direção ao centro

        gp = lambda n: self.get_parameter(n).value
        image_topic = gp('image_topic')
        depth_topic = gp('depth_topic')
        camera_info_topic = gp('camera_info_topic')
        self.marker_id = gp('marker_id')
        self.depth_win = gp('depth_window')
        self.depth_inset = gp('depth_inset')

        self.bridge = CvBridge()
        self.K = None
        self.last_depth = None
        self.last_valid_z_list = None

        # API antiga do OpenCV (4.5.4) + refinamento subpixel dos cantos
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.n_frames = 0
        self.n_depth_fail = 0

        self.sub_image = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(
            Image, depth_topic, self.depth_callback, qos_profile_sensor_data)
        self.sub_camera_info = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)

        self.pub_features = self.create_publisher(Float64MultiArray, '/ibvs/features', 10)

        self.get_logger().info(
            f'ArucoFeatureExtractor iniciado | image={image_topic} | depth={depth_topic} | '
            f'info={camera_info_topic} | marker_id={self.marker_id}')

    # ------------------------------------------------------------------
    def camera_info_callback(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        self.get_logger().info(
            f'camera_info recebido ({msg.width}x{msg.height}, fx={self.K[0,0]:.1f})', once=True)

    def depth_callback(self, msg):
        if msg.encoding == '16UC1':      # RealSense: milímetros
            depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.last_depth = depth_mm.astype(np.float32) / 1000.0
        elif msg.encoding == '32FC1':    # Gazebo: metros
            self.last_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        else:
            self.get_logger().error(
                f'Encoding de profundidade não suportado: {msg.encoding}',
                throttle_duration_sec=5.0)
            return
        self.get_logger().info(f'depth recebido (encoding={msg.encoding})', once=True)

    def robust_depth(self, u, v, min_z=0.05, max_z=2.0):
        """Mediana dos pixels válidos numa janela em torno de (u, v)."""
        h, w = self.last_depth.shape[:2]
        u, v = int(round(u)), int(round(v))
        k = self.depth_win
        patch = self.last_depth[max(0, v - k):min(h, v + k + 1),
                                max(0, u - k):min(w, u + k + 1)].ravel()
        patch = patch[np.isfinite(patch) & (patch > min_z) & (patch < max_z)]
        if patch.size < 5:
            return None
        return float(np.median(patch))

    def publish_features(self, s, z_list, marker_detected, depth_valid, stamp_sec, stamp_nanosec):
        msg_out = Float64MultiArray()
        msg_out.data = (list(s) + list(z_list)
                        + [1.0 if marker_detected else 0.0, 1.0 if depth_valid else 0.0]
                        + [stamp_sec, stamp_nanosec])
        self.pub_features.publish(msg_out)

    def show(self, img):
        cv2.imshow('ArUco Feature Extractor', img)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    def image_callback(self, msg):
        if self.K is None or self.last_depth is None:
            return

        stamp_sec = float(msg.header.stamp.sec)
        stamp_nanosec = float(msg.header.stamp.nanosec)

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)

        # escolhe o marcador certo
        idx = None
        if ids is not None:
            ids_flat = ids.flatten()
            if self.marker_id < 0:
                idx = 0
            elif self.marker_id in ids_flat:
                idx = int(np.where(ids_flat == self.marker_id)[0][0])

        if idx is None:
            self.get_logger().warn('Marcador não detectado', throttle_duration_sec=2.0)
            # sem marcador: flag 0 (o controlador para); s/Z são só placeholders
            self.publish_features([0.0] * 8, [0.0] * 4, False, False, stamp_sec, stamp_nanosec)
            self.show(frame)
            return

        pixel_pts = corners[idx][0]            # 4x2: TL, TR, BR, BL
        center = pixel_pts.mean(axis=0)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        s, z_list = [], []
        depth_ok = True
        for (u, v) in pixel_pts:
            s.extend([(u - cx) / fx, (v - cy) / fy])

            # lê Z um pouco para DENTRO do marcador (longe da borda com o fundo)
            uz, vz = (1 - self.depth_inset) * np.array([u, v]) + self.depth_inset * center
            z = self.robust_depth(uz, vz)
            if z is None:
                depth_ok = False
                z = 0.0
            z_list.append(z)

        self.n_frames += 1
        if depth_ok:
            self.last_valid_z_list = z_list
        else:
            self.n_depth_fail += 1
            self.get_logger().warn(
                f'Profundidade inválida — s atual publicado, Z do último válido '
                f'({self.n_depth_fail}/{self.n_frames} frames com falha)',
                throttle_duration_sec=2.0)
            if self.last_valid_z_list is not None:
                z_list = self.last_valid_z_list

        # s é SEMPRE o atual; depth_valid diz se o Z é deste frame
        self.publish_features(s, z_list, True, depth_ok, stamp_sec, stamp_nanosec)

        # janela de visualização (sempre aberta)
        dbg = frame.copy()
        cv2.aruco.drawDetectedMarkers(dbg, [corners[idx]])
        for lbl, (u, v), z in zip(['TL', 'TR', 'BR', 'BL'], pixel_pts, z_list):
            pt = (int(u), int(v))
            cv2.circle(dbg, pt, 5, (0, 255, 0), -1)
            cv2.putText(dbg, f'{lbl} Z={z:.3f}', (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        self.show(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoFeatureExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()