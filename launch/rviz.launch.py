import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('trash_robot_sim')
    rviz_config = os.path.join(pkg, 'rviz', 'robot.rviz')

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),
    ])
