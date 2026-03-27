"""
move_xyz.py
-----------
Módulo para mover o end-effector do braço Denso para um XYZ via MoveIt.
Vai para HOME antes de executar o movimento para garantir IK consistente.

Pode ser importado pelo main.py ou rodado diretamente.

Uso direto:
    ros2 run denso_script move_xyz

Uso como módulo (main.py):

    from denso_scripts.move_xyz import move_xyz

    move_xyz(x=0.022, y=0.031, z=0.914,
             qx=0.035, qy=0.106, qz=0.947, qw=0.301)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import time

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    RobotState,
)
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header


JOINT_NAMES     = ['joint_1', 'joint_2', 'joint_3',
                   'joint_4', 'joint_5', 'joint_6']
HOME_POSITIONS  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HOME_DURATION   = 5

PLANNING_GROUP  = 'arm'
END_EFFECTOR    = 'J6'
FRAME_ID        = 'world'
TOLERANCE_POS_M = 0.001
TOLERANCE_ORI_R = 0.01


class MoveXYZNode(Node):

    def __init__(self, x, y, z, qx, qy, qz, qw, go_home_first=True):
        """
        Parâmetros:
            x, y, z         : posição alvo em metros
            qx, qy, qz, qw  : orientação alvo em quaternion
            go_home_first   : True → vai à home antes (recomendado para IK consistente)
        """
        super().__init__('move_xyz')

        self._xyz = {'x': x, 'y': y, 'z': z}
        self._orientation = {'x': qx, 'y': qy, 'z': qz, 'w': qw}
        self._go_home_first = go_home_first

        self._joint_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/denso_joint_trajectory_controller/follow_joint_trajectory'
        )
        self._moveit_client = ActionClient(self, MoveGroup, '/move_action')

    def start(self):
        if self._go_home_first:
            self._go_home()
        else:
            self._go_xyz()

    def _go_home(self):
        self.get_logger().info('Indo para HOME...')
        self._joint_client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = HOME_POSITIONS
        point.time_from_start = Duration(sec=HOME_DURATION)
        goal.trajectory.points = [point]

        future = self._joint_client.send_goal_async(goal)
        future.add_done_callback(self._on_home_response)

    def _on_home_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Goal HOME rejeitado!')
            rclpy.shutdown()
            return
        handle.get_result_async().add_done_callback(self._on_home_result)

    def _on_home_result(self, future):
        result = future.result().result
        self.get_logger().info(f'HOME concluído. Código: {result.error_code}')
        time.sleep(3.0)
        self._go_xyz()

    def _go_xyz(self):
        self.get_logger().info('Planejando movimento para XYZ...')
        self._moveit_client.wait_for_server()

        start_state = RobotState(
            joint_state=JointState(
                name=JOINT_NAMES,
                position=HOME_POSITIONS,
            )
        )

        position_constraint = PositionConstraint(
            header=Header(frame_id=FRAME_ID),
            link_name=END_EFFECTOR,
            constraint_region=BoundingVolume(
                primitives=[SolidPrimitive(
                    type=SolidPrimitive.SPHERE,
                    dimensions=[TOLERANCE_POS_M],
                )],
                primitive_poses=[Pose(
                    position=Point(**self._xyz),
                    orientation=Quaternion(**self._orientation),
                )],
            ),
            weight=1.0,
        )

        orientation_constraint = OrientationConstraint(
            header=Header(frame_id=FRAME_ID),
            link_name=END_EFFECTOR,
            orientation=Quaternion(**self._orientation),
            absolute_x_axis_tolerance=TOLERANCE_ORI_R,
            absolute_y_axis_tolerance=TOLERANCE_ORI_R,
            absolute_z_axis_tolerance=TOLERANCE_ORI_R,
            weight=1.0,
        )

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest(
            group_name=PLANNING_GROUP,
            start_state=start_state,
            goal_constraints=[
                Constraints(
                    position_constraints=[position_constraint],
                    orientation_constraints=[orientation_constraint],
                )
            ],
        )

        self.get_logger().info(
            f"Enviando XYZ → x={self._xyz['x']}, "
            f"y={self._xyz['y']}, z={self._xyz['z']}"
        )
        future = self._moveit_client.send_goal_async(goal)
        future.add_done_callback(self._on_xyz_response)

    def _on_xyz_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Goal XYZ rejeitado pelo MoveIt!')
            rclpy.shutdown()
            return
        self.get_logger().info('Planejando trajetória...')
        handle.get_result_async().add_done_callback(self._on_xyz_result)

    def _on_xyz_result(self, future):
        result = future.result().result
        self.get_logger().info(
            f'Movimento XYZ concluído. Código de erro: {result.error_code.val}'
        )
        rclpy.shutdown()


#  FUNÇÃO PRINCIPAL

def move_xyz(x: float, y: float, z: float,
             qx: float, qy: float, qz: float, qw: float,
             go_home_first: bool = True):
    """
    Move o end-effector para a posição XYZ com orientação quaternion.

    Parâmetros:
        x, y, z          : posição alvo em metros
        qx, qy, qz, qw   : orientação em quaternion
        go_home_first    : True → vai à home antes do movimento (default: True)
                           False → vai direto para o XYZ (menos seguro)
    """
    rclpy.init()
    node = MoveXYZNode(x=x, y=y, z=z,
                       qx=qx, qy=qy, qz=qz, qw=qw,
                       go_home_first=go_home_first)
    node.start()
    rclpy.spin(node)


#  USO DIRETO (ros2 run)

def main():
    move_xyz(
        x=0.022, y=0.031, z=0.914,
        qx=0.035, qy=0.106, qz=0.947, qw=0.301,
        go_home_first=True,
    )


if __name__ == '__main__':
    main()