from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    tag = LaunchConfiguration('tag')
    start_delay = LaunchConfiguration('start_delay')
    timeout = LaunchConfiguration('timeout')

    logger = Node(
        package='denso_scripts',
        executable='ibvs_logger',
        name='ibvs_logger',
        output='screen',
        parameters=[{
            'tag': tag,
            'timeout': ParameterValue(timeout, value_type=float),
        }])

    controller = Node(
        package='denso_scripts',
        executable='ibvs_controller_real',
        name='ibvs_controller',
        output='screen',
        parameters=[{
            'start_delay': ParameterValue(start_delay, value_type=float),
        }])

    # quando o logger termina (convergiu+settling ou timeout), encerra tudo
    shutdown_on_logger_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=logger,
            on_exit=[EmitEvent(event=Shutdown(reason='logger finalizou o run'))]))

    return LaunchDescription([
        DeclareLaunchArgument('tag', default_value='run'),
        DeclareLaunchArgument('start_delay', default_value='2.0'),
        DeclareLaunchArgument('timeout', default_value='120.0'),
        logger,
        controller,
        shutdown_on_logger_exit,
    ])