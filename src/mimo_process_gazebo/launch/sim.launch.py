import os
# Fix for Ubuntu 24.04 Snap GTK theme bug crashing Gazebo Harmonic GUI
if 'GTK_PATH' in os.environ:
    os.environ.pop('GTK_PATH')

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
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
    
    # Start Gazebo Server and Client
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
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
        parameters=[robot_description]
    )

    # Load Joint State Broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
    )
    
    # Load Effort Controller
    effort_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['effort_controllers'],
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        joint_state_broadcaster_spawner,
        effort_controller_spawner
    ])
