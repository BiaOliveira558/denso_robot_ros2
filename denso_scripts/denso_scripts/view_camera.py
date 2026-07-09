import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import argparse
import threading
import os
from datetime import datetime

class DualViewer(Node):
    def __init__(self, output_dir):
        super().__init__('dual_viewer')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.last_rgb = None
        self.last_depth_raw = None
        self.last_depth_vis = None

        self.create_subscription(
            Image, '/basic_camera/rgb', self.cb_rgb, 10)
        self.create_subscription(
            Image, '/basic_camera/depth', self.cb_depth, 10)

    def cb_rgb(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.last_rgb = img_bgr
        cv2.imshow("RGB", img_bgr)
        cv2.waitKey(1)

    def cb_depth(self, msg):
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_vis = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        self.last_depth_raw = depth_clean
        self.last_depth_vis = depth_vis
        cv2.imshow("Depth", depth_vis)
        cv2.waitKey(1)

    def save_snapshot(self):
        if self.last_rgb is None or self.last_depth_raw is None:
            print("Ainda não há frames suficientes pra salvar.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = "/root/denso_ws/src/denso_robot_ros2/denso_scripts/denso_scripts/snapshot"
        os.makedirs(output_dir, exist_ok=True)

        rgb_path = os.path.join(output_dir, f"rgb_{timestamp}.png")
        depth_vis_path = os.path.join(output_dir, f"depth_vis_{timestamp}.png")
        depth_raw_path = os.path.join(output_dir, f"depth_raw_{timestamp}.npy")

        cv2.imwrite(rgb_path, self.last_rgb)
        cv2.imwrite(depth_vis_path, self.last_depth_vis)
        np.save(depth_raw_path, self.last_depth_raw)

        print(f"Snapshot salvo: {rgb_path}, {depth_vis_path}, {depth_raw_path}")


def keyboard_listener(node):
    print("Pressione ENTER para capturar um frame (RGB + Depth). Ctrl+C no terminal pra sair.")
    while rclpy.ok():
        input()
        node.save_snapshot()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./snapshots',
                         help='Diretório onde salvar os frames capturados')
    args = parser.parse_args()

    rclpy.init()
    node = DualViewer(output_dir=args.output_dir)

    listener_thread = threading.Thread(target=keyboard_listener, args=(node,), daemon=True)
    listener_thread.start()

    print("Mostrando RGB e Depth... Ctrl+C para sair")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()

if __name__ == "__main__":
    main()