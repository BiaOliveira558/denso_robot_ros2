#!/usr/bin/env python3
"""
denso_keyboard_teleop.py
------------------------
Controle do end-effector do braço Denso via teclado usando MoveIt Servo.

Uso:
    ros2 run denso_scripts keyboard_teleop

Teclas de controle:
    w / s  →  +X / -X  (frente/trás)
    a / d  →  +Y / -Y  (esquerda/direita)
    q / e  →  +Z / -Z  (cima/baixo)
    i / k  →  rotação +Y / -Y (pitch)
    j / l  →  rotação +Z / -Z (yaw)
    u / o  →  rotação +X / -X (roll)
    ESPAÇO →  para o movimento
    ESC    →  encerra o nó
"""

import sys
import tty
import termios
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header
import time


LINEAR_SPEED  = 0.05   # m/s
ANGULAR_SPEED = 0.2    # rad/s
FRAME_ID      = 'base_link'

KEY_BINDINGS = {
    'w': ( LINEAR_SPEED,  0.0,           0.0,           0.0,           0.0,           0.0),
    's': (-LINEAR_SPEED,  0.0,           0.0,           0.0,           0.0,           0.0),
    'a': ( 0.0,           LINEAR_SPEED,  0.0,           0.0,           0.0,           0.0),
    'd': ( 0.0,          -LINEAR_SPEED,  0.0,           0.0,           0.0,           0.0),
    'q': ( 0.0,           0.0,           LINEAR_SPEED,  0.0,           0.0,           0.0),
    'e': ( 0.0,           0.0,          -LINEAR_SPEED,  0.0,           0.0,           0.0),
    'u': ( 0.0,           0.0,           0.0,           ANGULAR_SPEED, 0.0,           0.0),
    'o': ( 0.0,           0.0,           0.0,          -ANGULAR_SPEED, 0.0,           0.0),
    'i': ( 0.0,           0.0,           0.0,           0.0,           ANGULAR_SPEED, 0.0),
    'k': ( 0.0,           0.0,           0.0,           0.0,          -ANGULAR_SPEED, 0.0),
    'j': ( 0.0,           0.0,           0.0,           0.0,           0.0,           ANGULAR_SPEED),
    'l': ( 0.0,           0.0,           0.0,           0.0,           0.0,          -ANGULAR_SPEED),
    ' ': ( 0.0,           0.0,           0.0,           0.0,           0.0,           0.0),
}

MENU = """
╔══════════════════════════════════════╗
║    DENSO Keyboard Teleop (Servo)     ║
╠══════════════════════════════════════╣
║  Translação:                         ║
║    W / S  →  +X / -X (frente/trás)  ║
║    A / D  →  +Y / -Y (esq/dir)      ║
║    Q / E  →  +Z / -Z (cima/baixo)   ║
║                                      ║
║  Rotação:                            ║
║    U / O  →  roll  +/-               ║
║    I / K  →  pitch +/-               ║
║    J / L  →  yaw   +/-               ║
║                                      ║
║  ESPAÇO   →  parar                   ║
║  ESC      →  sair                    ║
╚══════════════════════════════════════╝
"""


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class KeyboardTeleop(Node):

    def __init__(self):
        super().__init__('denso_keyboard_teleop')
        self._pub = self.create_publisher(
            TwistStamped,
            '/servo_node/delta_twist_cmds',
            10
        )
        self.get_logger().info('Keyboard teleop iniciado.')

    def publish(self, lx, ly, lz, ax, ay, az):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.twist.linear.x  = lx
        msg.twist.linear.y  = ly
        msg.twist.linear.z  = lz
        msg.twist.angular.x = ax
        msg.twist.angular.y = ay
        msg.twist.angular.z = az
        self._pub.publish(msg)

    def stop(self):
        self.publish(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def main():
    rclpy.init()
    node = KeyboardTeleop()
    settings = termios.tcgetattr(sys.stdin)

    print(MENU)
    print('Aguardando teclas...\n')

    try:
        while rclpy.ok():
            key = get_key(settings)

            # ESC
            if ord(key) == 27 or key == '\x03':
                print('\nEncerrando...')
                node.stop()
                break

            if key in KEY_BINDINGS:
                lx, ly, lz, ax, ay, az = KEY_BINDINGS[key]
                node.publish(lx, ly, lz, ax, ay, az)
                time.sleep(1)
                node.stop()
                if key == ' ':
                    print('PARADO', end='\r')
                else:
                    direction = {
                        'w': '+X (frente)', 's': '-X (trás)',
                        'a': '+Y (esq)',    'd': '-Y (dir)',
                        'q': '+Z (cima)',   'e': '-Z (baixo)',
                        'u': 'roll+',      'o': 'roll-',
                        'i': 'pitch+',     'k': 'pitch-',
                        'j': 'yaw+',       'l': 'yaw-',
                    }
                    print(f'Movendo: {direction.get(key, key)}    ', end='\r')
            else:
                node.stop()

    except Exception as e:
        print(f'\nErro: {e}')
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()