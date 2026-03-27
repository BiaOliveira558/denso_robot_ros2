"""
move_joints.py
--------------
Módulo para mover o braço Denso por posição de joints.
Pode ser importado pelo main.py ou rodado diretamente.

Uso direto:
    ros2 run denso_scripts move_joints

Uso como módulo (main.py):
    from denso_scripts.move_joints import move_joints, move_joints_sequence
    move_joints([[-114, -36, 86, -1, 0, -269]], duration=3, degrees=True)
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3',
               'joint_4', 'joint_5', 'joint_6']


def to_radians(positions: list) -> list:
    return [math.radians(p) for p in positions]


class MoveJointsNode(Node):

    def __init__(self, targets: list, degrees: bool = False):
        """
        targets : lista de (posições, duração_seg)
                  ex: [([0,0,0,0,0,0], 3), ([-114,-36,86,-1,0,-269], 5)]
        degrees : True  → posições em graus (converte automaticamente)
                  False → posições já em radianos
        """
        super().__init__('move_joints')
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/denso_joint_trajectory_controller/follow_joint_trajectory'
        )
        self._degrees = degrees
        self._original_targets = targets
        self._targets = [
            (to_radians(pos) if degrees else pos, t)
            for pos, t in targets
        ]
        self._index = 0
        self.done = False

    def send_next(self):
        if self._index >= len(self._targets):
            self.get_logger().info('Todos os movimentos de joints concluídos.')
            self.done = True
            rclpy.shutdown()
            return

        positions, duration = self._targets[self._index]
        self.get_logger().info('Aguardando servidor de ação...')
        self._client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=duration)
        goal.trajectory.points = [point]

        label = f'Target {self._index + 1}/{len(self._targets)}'
        if self._degrees:
            original = self._original_targets[self._index][0]
            self.get_logger().info(
                f'[{label}] {original}° → {[round(r, 4) for r in positions]} rad'
            )
        else:
            self.get_logger().info(f'[{label}] {positions} rad')

        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Goal rejeitado!')
            rclpy.shutdown()
            return
        self.get_logger().info('Goal aceito, aguardando resultado...')
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        result = future.result().result
        self.get_logger().info(f'Movimento concluído. Código: {result.error_code}')
        self._index += 1
        self.send_next()


def move_joints(positions: list, duration: int = 3, degrees: bool = True):
    """
    Move o robô para uma única posição de joints.

    Parâmetros:
        positions : lista com 6 valores de joint
                    ex: [-114, -36, 86, -1, 0, -269]
        duration  : tempo em segundos para o movimento (default: 3)
        degrees   : True  → valores em graus (default)
                    False → valores em radianos
    """
    rclpy.init()
    node = MoveJointsNode(targets=[(positions, duration)], degrees=degrees)
    node.send_next()
    rclpy.spin(node)


def move_joints_sequence(targets: list, degrees: bool = True):
    """
    Executa uma sequência de movimentos de joints.

    Parâmetros:
        targets : lista de tuplas (posições, duração)
                  ex: [([-114,-36,86,-1,0,-269], 3), ([0,0,0,0,0,0], 5)]
        degrees : True  → valores em graus (default)
                  False → valores em radianos
    """
    rclpy.init()
    node = MoveJointsNode(targets=targets, degrees=degrees)
    node.send_next()
    rclpy.spin(node)


#  USO DIRETO (ros2 run)

def main():
    targets = [
        ([0, 0, 0, 0, 0, 0],5),  # home
    ]
    move_joints_sequence(targets, degrees=True)


if __name__ == '__main__':
    main()