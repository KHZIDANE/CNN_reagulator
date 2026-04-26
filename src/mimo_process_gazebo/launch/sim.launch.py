import os
# Fix for Ubuntu 24.04 Snap GTK theme bug crashing Gazebo Harmonic GUI
if 'GTK_PATH' in os.environ:
    os.environ.pop('GTK_PATH')

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_mimo_description = get_package_share_directory('mimo_process_description')
    pkg_mimo_gazebo = get_package_share_directory('mimo_process_gazebo')
    
    # Process the URDF file (Now using the UR5e)
    xacro_file = os.path.join(pkg_mimo_description, 'urdf', 'ur5e_sim.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_description = {'robot_description': robot_description_config.toxml()}
    
    world_file = os.path.join(pkg_mimo_gazebo, 'worlds', 'zero_gravity.sdf')

    # Start Gazebo Server and Client
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # Bridge Gazebo clock into ROS 2 so controller timing advances correctly.
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )
    
    # Spawn Robot
    spawn_entity = Node(
        package='ros_gz_sim', 
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'mimo_arm'],
        output='screen'
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}]
    )

    # Load Effort Controller
    effort_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['effort_controllers'],
    )

    # Load Joint State Broadcaster (publishes /joint_states)
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
    )

    return LaunchDescription([
        gazebo,
        clock_bridge,
        robot_state_publisher,
        spawn_entity,
        TimerAction(period=5.0, actions=[effort_controller_spawner, joint_state_broadcaster_spawner]),
    ])
