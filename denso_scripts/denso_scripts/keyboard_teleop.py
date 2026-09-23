#!/usr/bin/env python3
"""
denso_keyboard_teleop.py
------------------------
Controle do end-effector via MoveIt Servo — mapeamento direto em base_link.
Movimento contínuo enquanto a tecla é mantida pressionada (usando
repetição automática do teclado do SO + timeout de "tecla solta").
"""

import sys
import tty
import termios
import select
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import time


LINEAR_SPEED   = 0.12
ANGULAR_SPEED  = 0.4
FRAME_ID       = 'base_link'
PUBLISH_RATE   = 50.0   # Hz de publicação contínua
KEY_TIMEOUT    = 0.15   # segundos sem receber a tecla = considera "soltou"

# X e Y invertidos em relação à versão anterior (W/S agora é Y, A/D é X)
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

MENU = """
╔══════════════════════════════════════╗
║    DENSO Keyboard Teleop (Servo)     ║
║    Segure a tecla para mover          ║
╠══════════════════════════════════════╣
║  Translação:                         ║
║    W / S  →  +Y / -Y                 ║
║    A / D  →  +X / -X                 ║
║    Q / E  →  +Z / -Z (cima/baixo)   ║
║                                      ║
║  Rotação:                            ║
║    U / O  →  roll  +/-               ║
║    I / K  →  pitch +/-               ║
║    J / L  →  yaw   +/-               ║
║                                      ║
║  ESPAÇO   →  parar                   ║
║  ESC/Ctrl+C → sair                   ║
╚══════════════════════════════════════╝
"""


def get_key_nonblocking(settings, timeout):
    """Lê uma tecla se disponível dentro de 'timeout' segundos; senão retorna None."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else None
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class KeyboardTeleop(Node):

    def __init__(self):
        super().__init__('denso_keyboard_teleop')
        self._pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.get_logger().info('Keyboard teleop iniciado (movimento contínuo).')

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

    period = 1.0 / PUBLISH_RATE
    last_cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    last_key_time = 0.0
    moving = False

    try:
        while rclpy.ok():
            key = get_key_nonblocking(settings, period)

            if key:
                if ord(key) == 27 or key == '\x03':
                    print('\nEncerrando...')
                    break

                if key == ' ':
                    last_cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    moving = False
                    node.stop()
                    print('PARADO                    ', end='\r')
                elif key in KEY_BINDINGS:
                    last_cmd = KEY_BINDINGS[key]
                    last_key_time = time.time()
                    moving = True
                    print(f'Movendo: {key}                    ', end='\r')

            # se estava se movendo mas a tecla parou de chegar (soltou) por muito tempo, para
            if moving and (time.time() - last_key_time) > KEY_TIMEOUT:
                moving = False
                last_cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                node.stop()
                print('PARADO                    ', end='\r')
            elif moving:
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