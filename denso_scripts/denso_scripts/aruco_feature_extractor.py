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

        # Tópicos como parâmetros (default = câmera real)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')

        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value

        self.bridge = CvBridge()
        self.K = None

        # API antiga do OpenCV (4.5.4)
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.last_valid_s = None
        self.last_valid_z_list = None
        self.last_valid_stamp = None
        self.last_depth = None

        self.sub_image = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(
            Image, depth_topic, self.depth_callback, qos_profile_sensor_data)
        self.sub_camera_info = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)

        self.pub_features = self.create_publisher(Float64MultiArray, '/ibvs/features', 10)

        self.get_logger().info(
            f'ArucoFeatureExtractor iniciado | image={image_topic} | '
            f'depth={depth_topic} | info={camera_info_topic}')

    def camera_info_callback(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        self.get_logger().info('camera_info recebido', once=True)

    def validate_depth(self, z_list, min_z=0.05, max_z=2.0):
        for z in z_list:
            if np.isnan(z) or z <= min_z or z > max_z:
                return False
        return True

    def depth_callback(self, msg):
        # Detecta o formato pela própria mensagem
        if msg.encoding == '16UC1':      # RealSense: milímetros (inteiro)
            depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.last_depth = depth_mm.astype(np.float32) / 1000.0
        elif msg.encoding == '32FC1':    # Gazebo: metros (float)
            self.last_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        else:
            self.get_logger().error(
                f'Encoding de profundidade não suportado: {msg.encoding}',
                throttle_duration_sec=5.0)
            return
        self.get_logger().info(f'depth recebido (encoding={msg.encoding})', once=True)

    def publish_features(self, s, z_list, marker_detected, depth_valid, stamp_sec, stamp_nanosec):
        msg_out = Float64MultiArray()
        msg_out.data = (s + z_list
                        + [1.0 if marker_detected else 0.0, 1.0 if depth_valid else 0.0]
                        + [stamp_sec, stamp_nanosec])
        self.pub_features.publish(msg_out)

    def image_callback(self, msg):
        self.get_logger().info('image_callback chamado', once=True)

        if self.K is None or self.last_depth is None:
            return

        stamp_sec = float(msg.header.stamp.sec)
        stamp_nanosec = float(msg.header.stamp.nanosec)

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)
        debug_frame = frame.copy()

        if ids is None:
            self.get_logger().warn('Marcador não detectado', throttle_duration_sec=2.0)
            if self.last_valid_s is not None:
                self.publish_features(self.last_valid_s, self.last_valid_z_list,
                                      False, False, stamp_sec, stamp_nanosec)
            cv2.imshow('ArUco Feature Extractor', debug_frame)
            cv2.waitKey(1)
            return

        cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)

        pixel_pts = corners[0][0]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        s = []
        z_list = []
        labels = ['TL', 'TR', 'BR', 'BL']

        for i, (u, v) in enumerate(pixel_pts):
            x = (u - cx) / fx
            y = (v - cy) / fy
            s.extend([x, y])

            z = float(self.last_depth[int(v), int(u)])
            z_list.append(z)

            pt = (int(u), int(v))
            cv2.circle(debug_frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(debug_frame, f'{labels[i]} Z={z:.2f}m', (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if self.validate_depth(z_list):
            self.last_valid_s = s
            self.last_valid_z_list = z_list
            self.last_valid_stamp = (stamp_sec, stamp_nanosec)
            self.publish_features(s, z_list, True, True, stamp_sec, stamp_nanosec)
        else:
            self.get_logger().warn('Profundidade inválida detectada — congelando features',
                                   throttle_duration_sec=2.0)
            if self.last_valid_s is not None:
                self.publish_features(self.last_valid_s, self.last_valid_z_list,
                                      True, False, stamp_sec, stamp_nanosec)

        cv2.imshow('ArUco Feature Extractor', debug_frame)
        cv2.waitKey(1)


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
        rclpy.shutdown()


if __name__ == '__main__':
    main()