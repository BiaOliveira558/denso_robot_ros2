from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    save_csv_arg = DeclareLaunchArgument(
        'save_csv',
        default_value='True',
        description='Se true, salva o log do PBVS em CSV'
    )

    csv_path_arg = DeclareLaunchArgument(
        'csv_path',
        default_value='/root/denso_ws/denso_scripts/logs/pbvs_log.csv',
        description='Caminho do arquivo CSV de log'
    )

    aruco_node = Node(
        package='denso_scripts',
        executable='aruco_pose_estimator',
        name='aruco_pose_estimator',
        output='screen',
        additional_env={'PYTHONUNBUFFERED': '1'},
    )

    pbvs_node = Node(
        package='denso_scripts',
        executable='pbvs_controller',
        name='pbvs_controller',
        output='screen',
        additional_env={'PYTHONUNBUFFERED': '1'},
        parameters=[{
            'save_csv': LaunchConfiguration('save_csv'),
            'csv_path': LaunchConfiguration('csv_path'),
        }],
    )

    return LaunchDescription([
        save_csv_arg,
        csv_path_arg,
        aruco_node,
        pbvs_node,
    ])