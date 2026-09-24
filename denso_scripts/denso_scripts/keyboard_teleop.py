#!/usr/bin/env python3
"""
keyboard_teleop.py
------------------
Controle do end-effector do Denso via MoveIt Servo (real e simulação).
Movimento contínuo enquanto a tecla é mantida pressionada.

Uso:
    ros2 run denso_scripts keyboard_teleop
"""

import sys
import tty
import termios
import select
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


LINEAR_SPEED      = 0.12
ANGULAR_SPEED     = 0.4
FRAME_ID          = 'base_link'
PUBLISH_RATE      = 50.0   # Hz
KEY_TIMEOUT_FIRST = 0.6    # tolera o atraso inicial da repetição do teclado
KEY_TIMEOUT       = 0.15   # depois que a repetição começou

KEY_BINDINGS = {
    'w': ( 0.0,           LINEAR_SPEED,  0.0,           0.0,           0.0,           0.0),
    's': ( 0.0,          -LINEAR_SPEED,  0.0,           0.0,           0.0,           0.0),
    'a': ( LINEAR_SPEED,  0.0,           0.0,           0.0,           0.0,           0.0),
    'd': (-LINEAR_SPEED,  0.0,           0.0,           0.0,           0.0,           0.0),
    'q': ( 0.0,           0.0,           LINEAR_SPEED,  0.0,           0.0,           0.0),
    'e': ( 0.0,           0.0,          -LINEAR_SPEED,  0.0,           0.0,           0.0),
    'u': ( 0.0,           0.0,           0.0,           ANGULAR_SPEED, 0.0,           0.0),
    'o': ( 0.0,           0.0,           0.0,          -ANGULAR_SPEED, 0.0,           0.0),
    'i': ( 0.0,           0.0,           0.0,           0.0,           ANGULAR_SPEED, 0.0),
    'k': ( 0.0,           0.0,           0.0,           0.0,          -ANGULAR_SPEED, 0.0),
    'j': ( 0.0,           0.0,           0.0,           0.0,           0.0,           ANGULAR_SPEED),
    'l': ( 0.0,           0.0,           0.0,           0.0,           0.0,          -ANGULAR_SPEED),
}

MENU = f"""
╔══════════════════════════════════════╗
║    DENSO Keyboard Teleop (Servo)     ║
║    Segure a tecla para mover         ║
╠══════════════════════════════════════╣
  lin={LINEAR_SPEED} | ang={ANGULAR_SPEED} | frame={FRAME_ID}

  Translação:
    W / S  →  +Y / -Y
    A / D  →  +X / -X
    Q / E  →  +Z / -Z

  Rotação:
    U / O  →  roll  +/-
    I / K  →  pitch +/-
    J / L  →  yaw   +/-

  ESPAÇO     →  parar
  ESC/Ctrl+C →  sair
╚══════════════════════════════════════╝
"""


def get_key_nonblocking(settings, timeout):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else None
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class KeyboardTeleop(Node):

    def __init__(self):
        super().__init__('denso_keyboard_teleop')
        self._pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.get_logger().info('Keyboard teleop iniciado (movimento contínuo).')

    def publish(self, lx, ly, lz, ax, ay, az):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.twist.linear.x = lx
        msg.twist.linear.y = ly
        msg.twist.linear.z = lz
        msg.twist.angular.x = ax
        msg.twist.angular.y = ay
        msg.twist.angular.z = az
        self._pub.publish(msg)

    def stop(self):
        self.publish(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    settings = termios.tcgetattr(sys.stdin)

    print(MENU)
    print('Aguardando teclas...\n')

    period = 1.0 / PUBLISH_RATE
    zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    last_cmd = zero
    last_key = None
    last_key_time = 0.0
    first_key_time = 0.0
    moving = False

    try:
        while rclpy.ok():
            key = get_key_nonblocking(settings, period)

            if key:
                if ord(key) == 27 or key == '\x03':
                    print('\nEncerrando...')
                    break

                if key == ' ':
                    last_cmd = zero
                    last_key = None
                    moving = False
                    node.stop()
                    print('PARADO                    ', end='\r')
                elif key in KEY_BINDINGS:
                    now = time.time()
                    if key != last_key or not moving:
                        first_key_time = now
                    last_cmd = KEY_BINDINGS[key]
                    last_key = key
                    last_key_time = now
                    moving = True
                    print(f'Movendo: {key}                    ', end='\r')

            if moving:
                waiting_first_repeat = (last_key_time == first_key_time)
                timeout = KEY_TIMEOUT_FIRST if waiting_first_repeat else KEY_TIMEOUT
                if (time.time() - last_key_time) > timeout:
                    moving = False
                    last_key = None
                    last_cmd = zero
                    node.stop()
                    print('PARADO                    ', end='\r')
                else:
                    node.publish(*last_cmd)

    except Exception as e:
        print(f'\nErro: {e}')
    finally:
        node.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()