from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    lc = LaunchConfiguration
    fl = lambda n: ParameterValue(LaunchConfiguration(n), value_type=float)

    args = [
        DeclareLaunchArgument('tag', default_value='run'),
        DeclareLaunchArgument('start_delay', default_value='2.0'),
        DeclareLaunchArgument('timeout', default_value='90.0'),
        DeclareLaunchArgument('lambda_gain', default_value='1.2'),
        DeclareLaunchArgument('damping', default_value='0.01'),
        DeclareLaunchArgument('max_lin_vel', default_value='0.08'),
        DeclareLaunchArgument('max_ang_vel', default_value='0.30'),
        DeclareLaunchArgument('convergence_threshold', default_value='0.025'),
        DeclareLaunchArgument('convergence_frames', default_value='5'),
    ]

    logger = Node(
        package='denso_scripts',
        executable='ibvs_xfeat_logger',
        name='ibvs_xfeat_logger',
        output='screen',
        parameters=[{
            'mode': 'run',
            'tag': ParameterValue(lc('tag'), value_type=str),
            'timeout': fl('timeout'),
        }])

    controller = Node(
        package='denso_scripts',
        executable='ibvs_xfeat_controller_real',
        name='ibvs_xfeat_controller',
        output='screen',
        parameters=[{
            'start_delay': fl('start_delay'),
            'lambda_gain': fl('lambda_gain'),
            'damping': fl('damping'),
            'max_lin_vel': fl('max_lin_vel'),
            'max_ang_vel': fl('max_ang_vel'),
            'convergence_threshold': fl('convergence_threshold'),
            'convergence_frames': ParameterValue(lc('convergence_frames'), value_type=int),
        }])

    # quando o logger termina (convergiu + settling, ou timeout), encerra o launch
    shutdown_on_logger_exit = RegisterEventHandler(
        OnProcessExit(target_action=logger,
                      on_exit=[EmitEvent(event=Shutdown(reason='logger finalizou o run'))]))

    return LaunchDescription(args + [logger, controller, shutdown_on_logger_exit])