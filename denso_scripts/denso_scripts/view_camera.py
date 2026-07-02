import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import cv2
import numpy as np

class DualViewer(Node):
    def __init__(self):
        super().__init__('dual_viewer')
        self.create_subscription(
            CompressedImage, '/basic_camera/compressed', self.cb_rgb, 10)
        self.create_subscription(
            Image, '/basic_camera/depth', self.cb_depth, 10)

    def cb_rgb(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("RGB", img)
            cv2.waitKey(1)

    def cb_depth(self, msg):
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow("Depth", depth_color)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = DualViewer()
    print("Mostrando RGB e Depth... Ctrl+C para sair")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()

if __name__ == "__main__":
    main()