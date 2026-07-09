#!/usr/bin/env python3
"""
Nó ROS 2 que detecta um marcador ArUco na imagem simulada e estima sua pose
(posição + orientação) relativa à câmera via solvePnP, publicando o resultado
como PoseStamped para uso no PBVSController.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R


class ArucoPoseEstimator(Node):
    def __init__(self):
        super().__init__('aruco_pose_estimator')

        # --- Parâmetros configuráveis ---
        self.declare_parameter('marker_size_m', 0.04)      # tamanho físico real do marcador, em metros
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        self.declare_parameter('camera_frame', 'basic_camera_link')
        self.declare_parameter('rgb_topic', '/basic_camera/rgb')
        self.declare_parameter('camera_info_topic', '/basic_camera/camera_info')
        self.declare_parameter('pose_topic', '/aruco/pose')

        self.marker_size = self.get_parameter('marker_size_m').value
        dict_name = self.get_parameter('aruco_dict').value
        self.camera_frame = self.get_parameter('camera_frame').value

        self.aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
        self.aruco_params = aruco.DetectorParameters_create()

        self.bridge = CvBridge()
        self.K = None          # matriz intrínseca, vinda do CameraInfo
        self.dist = None       # coeficientes de distorção

        # Pontos 3D do marcador no seu próprio frame de referência (centrado, no plano Z=0)
        s = self.marker_size / 2.0
        self.obj_points = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0],
        ], dtype=np.float32)

        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self.cb_camera_info, 10)
        self.create_subscription(
            Image, self.get_parameter('rgb_topic').value,
            self.cb_image, 10)

        self.pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter('pose_topic').value, 10)

        self.get_logger().info('Aruco Pose Estimator iniciado, aguardando camera_info...')

    def cb_camera_info(self, msg: CameraInfo):
        # K vem como array 3x3 achatado (row-major)
        first_time = self.K is None
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist = np.array(msg.d, dtype=np.float64) if len(msg.d) > 0 else np.zeros(5)
        if first_time:
            self.get_logger().debug(f'camera_info recebido pela primeira vez! K =\n{self.K}')

    def cb_image(self, msg: Image):
        if self.K is None:
            self.get_logger().debug('Frame recebido, mas K ainda é None (camera_info não chegou)')
            return

        self.get_logger().debug('Recebendo frame, tentando detectar marcador...')

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        self.get_logger().debug(f'ids detectados: {ids}')

        if ids is None or len(corners) == 0:
            return

        # Usa apenas o primeiro marcador detectado
        marker_corners = corners[0][0]  # shape (4, 2): TL, TR, BR, BL

        success, rvec, tvec = cv2.solvePnP(
            self.obj_points, marker_corners, self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE  # otimizado para marcadores planos quadrados
        )

        if not success:
            self.get_logger().warn('solvePnP falhou nesse frame')
            return

        self.publish_pose(rvec, tvec, msg.header.stamp)

        # Desenho para depuração visual
        cv2.drawFrameAxes(frame, self.K, self.dist, rvec, tvec, self.marker_size * 0.5)
        aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('Aruco Pose', frame)
        cv2.waitKey(1)

    def publish_pose(self, rvec, tvec, stamp):
        rot_matrix, _ = cv2.Rodrigues(rvec)
        quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.camera_frame

        pose_msg.pose.position.x = float(tvec[0])
        pose_msg.pose.position.y = float(tvec[1])
        pose_msg.pose.position.z = float(tvec[2])

        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])

        self.pose_pub.publish(pose_msg)
        self.get_logger().debug(f'Pose publicada: {pose_msg.pose.position}')


def main():
    rclpy.init()
    node = ArucoPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()