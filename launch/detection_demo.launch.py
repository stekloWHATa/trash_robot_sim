import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('trash_robot_sim')
    gz_pkg = get_package_share_directory('ros_gz_sim')

    world_file = os.path.join(pkg, 'worlds', 'detection_demo_world.sdf')
    robot_sdf = os.path.join(pkg, 'models', 'robot', 'only_robot.sdf')
    robot_xacro = os.path.join(pkg, 'urdf', 'trash_robot.urdf.xacro')
    params_file = os.path.join(pkg, 'config', 'params.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    scripted_motion = LaunchConfiguration('scripted_motion', default='true')

    gz_resource = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.pathsep.join([
            os.path.join(pkg, 'models'),
            os.path.join(pkg, 'models', 'trash'),
        ]),
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'robot',
            '-file', robot_sdf,
            '-x', '-0.5',
            '-y', '-2.0',
            '-z', '0.25',
            '-Y', '0.0',
        ],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/rgbd/image/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/rgbd/image/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/rgbd/image/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    robot_description = ParameterValue(
        Command(['xacro ', robot_xacro]),
        value_type=str,
    )
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': use_sim_time},
        ],
        output='screen',
    )

    map_builder = Node(
        package='trash_robot_sim',
        executable='map_builder.py',
        name='map_builder',
        parameters=[params_file, {'use_sim_time': False}],
        output='screen',
    )

    detector = Node(
        package='trash_robot_sim',
        executable='detector.py',
        name='trash_detector',
        parameters=[params_file, {'use_sim_time': False}],
        output='screen',
    )

    motion = Node(
        package='trash_robot_sim',
        executable='scripted_motion.py',
        name='scripted_motion',
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(scripted_motion),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'scripted_motion',
            default_value='true',
            description='Publish a simple repeatable cmd_vel path for the video demo',
        ),
        gz_resource,
        gazebo,
        spawn,
        bridge,
        robot_state_pub,
        map_builder,
        detector,
        motion,
    ])
