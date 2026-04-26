
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
# CNN Regulator for UR5e Industrial Robot

## Project Overview
This project implements a Convolutional Neural Network (CNN) based regulator for a 6-DOF Universal Robots UR5e industrial manipulator using **Imitation Learning** through Behavioral Cloning. The system uses ROS 2 Jazzy and Gazebo Harmonic for simulation.

## Architecture
- **Plant Model:** Universal Robots UR5e (6-DOF robotic arm)
- **Simulation:** Gazebo Harmonic with `gz_ros2_control` integration
- **Neural Network:** 1D CNN with 3 convolutional layers → fully connected layers
- **Input:** 12 sensors (6 joint positions + 6 joint velocities) over a 10-timestep window
- **Output:** 6 control torques (one per joint)
- **Training Data:** Generated via PD controller imitation (Behavioral Cloning)

## Project Structure
```
project_cnn_regulation/
├── src/
│   ├── mimo_process_description/        # URDF and robot meshes
│   │   └── urdf/
│   │       └── ur5e_sim.urdf.xacro      # UR5e with ros2_control integration
│   ├── mimo_process_gazebo/              # Gazebo simulation setup
│   │   ├── launch/
│   │   │   └── sim.launch.py             # Main launch file
│   │   └── config/
│   │       └── controllers.yaml          # ros2_control configuration
│   ├── cnn_regulator/                    # Core CNN implementation
│   │   ├── cnn_regulator/
│   │   │   ├── cnn_model.py              # MIMO_CNN_Regulator class
│   │   │   ├── ros_controller.py         # ROS 2 controller node (inference)
│   │   │   ├── data_collector.py         # PD controller for data collection
│   │   │   ├── gui_target_ui.py          # Tkinter GUI control panel
│   │   │   └── train.py                  # Training script
│   │   └── setup.py
│   └── ur_description/                   # Official UR robot URDF (submodule)
└── install/                              # Built packages

```

## Quick Start Guide

### 1. Build the Workspace
```bash
cd ~/Desktop/project_cnn_regulation
colcon build --symlink-install
source install/setup.bash
```

### 2. Launch Gazebo with UR5e Robot
```bash
env -u GTK_PATH ros2 launch mimo_process_gazebo sim.launch.py
```
*(Note: `env -u GTK_PATH` is required when running from VS Code terminal to avoid Snap GTK conflicts)*

### 3. Collect Training Data (in a new terminal)
```bash
source install/setup.bash
ros2 run cnn_regulator data_collector
```
Run for 30-60 seconds to collect enough trajectories. This generates `training_data.csv`.

### 4. Train the CNN Model
```bash
python3 src/cnn_regulator/cnn_regulator/train.py
```
This trains the model for 50 epochs and saves weights to `cnn_regulator_weights.pth`.

### 5. Deploy the Trained CNN as Controller (in a new terminal)
```bash
source install/setup.bash
ros2 run cnn_regulator ros_controller
```
The trained CNN will now regulate the robot in real-time, moving it toward the target position.

### 6. Open the Graphical Control Panel
```bash
source install/setup.bash
ros2 run cnn_regulator target_gui
```
Use the sliders to set a 6-joint target, then click `Send Target`. You can also use `Home`, `Training`, `Extended`, `Compact`, `Demo`, or `Hold Current`.

## Key Files & Components

### Neural Network Architecture (`cnn_model.py`)
- **Input shape:** (Batch, 12 sensors, 10 timesteps)
- **Layer 1:** Conv1D(12→16, kernel=3) + ReLU
- **Layer 2:** Conv1D(16→32, kernel=3) + ReLU
- **FC Layer 1:** 320 → 64 + ReLU
- **Output Layer:** 64 → 6 (torque commands)
- **Loss:** MSE (Mean Squared Error)

### Training Process
- **Dataset:** RobotDataset class reads `training_data.csv`
- **Optimizer:** Adam (lr=0.001)
- **Batch Size:** 32
- **Epochs:** 50
- **Output:** Trained weights saved as `cnn_regulator_weights.pth`

### ROS 2 Integration
- **State Publisher:** `joint_state_broadcaster` publishes `/joint_states`
- **Command Interface:** `effort_controllers/JointGroupEffortController` receives torque commands
- **Topics:**
  - `/joint_states` → Input to CNN
  - `/effort_controllers/commands` → Output from CNN

## Performance Notes
- Loss converged smoothly from 331.43 → 0.5791 (no overfitting detected)
- CNN successfully learned to mimic the PD controller behavior
- Real-time inference runs at 100 Hz (ros2_control loop rate)

## Dependencies
- ROS 2 Jazzy
- Gazebo Harmonic (8.11.0+)
- PyTorch (`torch`, `torchvision`)
- NumPy, Pandas
- `ros2_control`, `controller_manager`, `effort_controllers`

## Troubleshooting

**Gazebo GUI crashes on Ubuntu 24.04 with VS Code terminal:**
- Solution: Prepend `env -u GTK_PATH` to launch commands to strip Snap GTK environment variables

**ImportError: librcl_action.so**
- Solution: Re-source ROS setup: `source /opt/ros/jazzy/setup.bash && source install/setup.bash`

**Data collector exits with rcl warnings:**
- These are benign ROS cleanup warnings and do not affect data collection

## Future Enhancements
1. **Reinforcement Learning:** Replace PD imitation with direct RL training
2. **Additional Sensors:** Add camera input for vision-based control
3. **Multi-task Learning:** Train on multiple target positions simultaneously
4. **Hardware Deployment:** Test on real UR5e robot
5. **Performance Metrics:** Add trajectory tracking error visualization

