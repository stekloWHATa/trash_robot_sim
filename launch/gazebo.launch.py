import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg      = get_package_share_directory('trash_robot_sim')
    gz_pkg   = get_package_share_directory('ros_gz_sim')

    world_file = os.path.join(pkg, 'worlds', 'trash_world.sdf')
    robot_sdf  = os.path.join(pkg, 'models', 'robot', 'only_robot.sdf')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    run_nav      = LaunchConfiguration('run_nav',      default='true')

    # Чтобы Gazebo нашёл наши модели
    gz_resource = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg, 'models'),
    )

    # Gazebo Harmonic
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # Спавн робота — смещение x=-0.5 чтобы центр тела совпал с (0,0) odom
    # Модель спавнится в (-0.5, -2), тело (offset +0.5x) = (0, -2) — далеко от всех барьеров
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

    # Bridge Gazebo <-> ROS2
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/rgbd/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/rgbd/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/rgbd/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    # A* навигатор — use_sim_time=False чтобы таймер шёл по wall-clock,
    # иначе при паузе Gazebo таймер замирает и cmd_vel не публикуется.
    navigator = Node(
        package='trash_robot_sim',
        executable='navigator.py',
        name='astar_navigator',
        parameters=[{'use_sim_time': False}],
        output='screen',
    )

    # Построитель карты + детектор мусора
    map_builder = Node(
        package='trash_robot_sim',
        executable='map_builder.py',
        name='map_builder',
        parameters=[{'use_sim_time': False}],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Использовать время симуляции'),
        DeclareLaunchArgument('run_nav', default_value='true',
                              description='Запустить A* навигатор'),
        gz_resource,
        gazebo,
        spawn,
        bridge,
        navigator,
        map_builder,
    ])
