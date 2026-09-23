import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime

SAVE_DIR = os.path.expanduser('~/capturas')
os.makedirs(SAVE_DIR, exist_ok=True)

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_cb, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_cb, 10)

        cv2.namedWindow('Camera Viewer')
        cv2.setMouseCallback('Camera Viewer', self.mouse_click)

        self.button_rect = (10, 10, 150, 50)  # x, y, w, h
        self.timer = self.create_timer(0.03, self.render)

    def rgb_cb(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = self.button_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.capture()

    def capture(self):
        if self.rgb_image is None or self.depth_image is None:
            self.get_logger().warn('Ainda sem imagem para capturar')
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(os.path.join(SAVE_DIR, f'rgb_{ts}.png'), self.rgb_image)
        cv2.imwrite(os.path.join(SAVE_DIR, f'depth_{ts}.png'),
                     self.depth_image.astype(np.uint16))
        self.get_logger().info(f'Captura salva: rgb_{ts}.png / depth_{ts}.png')

    def render(self):
        if self.rgb_image is None:
            return
        display = self.rgb_image.copy()

        bx, by, bw, bh = self.button_rect
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 200, 0), -1)
        cv2.putText(display, 'CAPTURAR', (bx + 10, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow('Camera Viewer', display)

        if self.depth_image is not None:
            depth_vis = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow('Depth', depth_colored)

        cv2.waitKey(1)

def main():
    rclpy.init()
    node = CameraViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()