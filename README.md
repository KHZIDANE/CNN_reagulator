CNN Regulator for UR5e Industrial Robot
A Convolutional Neural Network (CNN) based controller for 6-DOF collaborative robot manipulation, implementing learned policies through imitation learning (Behavioral Cloning).

Key Features
Robot: Universal Robots UR5e (6-DOF collaborative arm)
Neural Network: 1D CNN with 3 convolutional layers for MIMO control
Training: Behavioral cloning from PD controller demonstrations
Real-time: 100 Hz control loop with <1ms inference time
Simulation: Gazebo Harmonic with ROS 2 Jazzy integration
System Description
The controller uses a sliding window of 12 sensor inputs (6 joint positions + 6 joint velocities over 10 timesteps) to predict 6 effort commands (torques). The system achieves 99.8% training loss reduction and enables real-time autonomous control of the UR5e arm with full error feedback integration.

Technology Stack
ROS 2 Jazzy
Gazebo Harmonic (v8.11.0)
PyTorch (CNN implementation)
gz_ros2_control for robot integration
