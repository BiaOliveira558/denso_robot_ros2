#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
import cv2
import numpy as np
from cv_bridge import CvBridge


class ArucoFeatureExtractor(Node):
    def __init__(self):
        super().__init__('aruco_feature_extractor')

        self.bridge = CvBridge()
        self.K = None  # intrínsecos, vem do tópico camera_info

        # API antiga do OpenCV (4.5.4) — Dictionary_get / DetectorParameters_create
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.sub_image = self.create_subscription(
            Image, '/basic_camera/rgb', self.image_callback, 10)
        self.sub_depth = self.create_subscription(
            Image, '/basic_camera/depth', self.depth_callback, 10)
        self.sub_camera_info = self.create_subscription(
            CameraInfo, '/basic_camera/camera_info', self.camera_info_callback, 10)

        self.pub_features = self.create_publisher(
            Float64MultiArray, '/ibvs/features', 10)

        self.last_depth = None
        self.get_logger().info('ArucoFeatureExtractor iniciado')

    def camera_info_callback(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        self.get_logger().info('camera_info recebido', once=True)

    def depth_callback(self, msg):
        self.last_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        self.get_logger().info('depth recebido', once=True)

    def image_callback(self, msg):
        self.get_logger().info('image_callback chamado', once=True)

        if self.K is None or self.last_depth is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)

        debug_frame = frame.copy()

        if ids is None:
            self.get_logger().warn('Marcador não detectado', throttle_duration_sec=2.0)
            cv2.imshow('ArUco Feature Extractor', debug_frame)
            cv2.waitKey(1)
            return

        # desenha o contorno do marcador e o ID
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

            # marca o ponto e escreve label + Z
            pt = (int(u), int(v))
            cv2.circle(debug_frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(debug_frame, f'{labels[i]} Z={z:.2f}m', (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow('ArUco Feature Extractor', debug_frame)
        cv2.waitKey(1)

        msg_out = Float64MultiArray()
        msg_out.data = s + z_list
        self.pub_features.publish(msg_out)


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