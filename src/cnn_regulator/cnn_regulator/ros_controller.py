import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import torch
from .cnn_model import MIMO_CNN_Regulator
import collections
import numpy as np
import os
from pathlib import Path

class CNNControllerNode(Node):
    def __init__(self):
        super().__init__('cnn_regulator_node')
        
        # ROS parameters (can be tuned externally via ROS params)
        self.declare_parameter('sequence_length', 10)
        self.declare_parameter('num_sensors', 12)     # 6 joints * 2 (pos, vel)
        self.declare_parameter('num_actuators', 6)   # 6 joint torques
        self.declare_parameter('target_position', [1.0, -1.0, 1.0, -1.0, 1.0, 0.5])
        self.declare_parameter('error_feedback_gain', 0.1)  # Proportional control on error
        
        # Retrieve parameters
        self.seq_len = self.get_parameter('sequence_length').get_parameter_value().integer_value
        self.num_sensors = self.get_parameter('num_sensors').get_parameter_value().integer_value
        self.num_actuators = self.get_parameter('num_actuators').get_parameter_value().integer_value
        target_list = self.get_parameter('target_position').get_parameter_value().double_array_value
        self.error_gain = self.get_parameter('error_feedback_gain').get_parameter_value().double_value
        
        # Target positions for the 6 joints (from parameter or defaults)
        self.target_q = np.array(target_list if target_list else [1.0, -1.0, 1.0, -1.0, 1.0, 0.5])
        
        # Joint names in order
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        # Current position (for error computation)
        self.current_q = np.zeros(6)
        self.current_dq = np.zeros(6)
        
        # Instantiate Network Structure
        self.model = MIMO_CNN_Regulator(
            sequence_length=self.seq_len,
            num_sensors=self.num_sensors,
            num_actuators=self.num_actuators
        )
        
        # Find the weights file - check multiple locations
        weights_file = self._find_weights_file()
        
        # Load pre-trained weights from the training phase
        if weights_file:
            try:
                self.model.load_state_dict(torch.load(weights_file, weights_only=True))
                self.get_logger().info(f'Loaded trained CNN weights from {weights_file}')
            except Exception as e:
                self.get_logger().warn(f'Failed to load weights from {weights_file}: {e}. Running with random initialized weights.')
        else:
            self.get_logger().warn('cnn_regulator_weights.pth not found. Running with random initialized weights.')
        
        self.model.eval() # We are just doing inference when running normally in ROS
        
        # A sliding window buffer to keep the recent sensor states (sequence_length)
        # using a deque of length self.seq_len for each sensor
        self.state_history = collections.deque(maxlen=self.seq_len)
        
        # ROS Interfaces
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',  # Topic that publishes joint state from Gazebo
            self.state_callback,
            10
        )
        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/effort_controllers/commands', # Topic to send effort commands to the robot
            10
        )
        
        # Additional publishers for monitoring
        self.error_publisher = self.create_publisher(
            Float64MultiArray,
            '/cnn_regulator/position_error',  # For debugging: shows error to target
            10
        )
        
        self.get_logger().info(f'Target position set to: {self.target_q}')
        self.get_logger().info('CNN Regulator Node has been started.')

    def state_callback(self, msg):
        # Extract joint positions and velocities from JointState message
        try:
            indices = [msg.name.index(name) for name in self.joint_names]
        except ValueError:
            self.get_logger().warn('Could not find all joint names in JointState message.')
            return
            
        q = np.array([msg.position[idx] for idx in indices])
        dq = np.array([msg.velocity[idx] for idx in indices])
        
        # Store current state for error computation
        self.current_q = q.copy()
        self.current_dq = dq.copy()
        
        # Create state vector: [q1, q2, q3, q4, q5, q6, dq1, dq2, dq3, dq4, dq5, dq6]
        current_state = np.concatenate((q, dq)).tolist()
        self.state_history.append(current_state)
        
        # Only compute the model when we have enough data (at least sequence_length)
        if len(self.state_history) == self.seq_len:
            self.compute_and_publish_control()

    def compute_and_publish_control(self):
        # Read target position dynamically (allows runtime updates via ros2 param set)
        target_param = self.get_parameter('target_position').get_parameter_value().double_array_value
        self.target_q = np.array(target_param) if target_param else self.target_q
        
        # Format the sliding window data to fit the CNN Input (Batch, Channels/Sensors, SequenceLength)
        # Currently, data is shaped as (sequence_length, num_sensors) -> [[s1, s2], [s1, s2]]
        
        # Convert List of Lists to a PyTorch tensor
        tensor_data = torch.tensor(list(self.state_history), dtype=torch.float32)
        
        # Transpose to get -> (num_sensors, sequence_length)
        tensor_data = tensor_data.T
        
        # Add the batch dimension -> (1, num_sensors, sequence_length)
        x = tensor_data.unsqueeze(0)
        
        # Pass through our CNN Regulator
        with torch.no_grad():
            cnn_action = self.model(x)  # action tensor of shape (1, num_actuators)
            
        cnn_output = cnn_action.squeeze(0).numpy()
        
        # Compute position error and add error feedback
        error = self.target_q - self.current_q
        
        # Combine CNN output with proportional position feedback
        # This helps the controller track the target position
        error_feedback = self.error_gain * error
        final_action = cnn_output + error_feedback
        
        # Publish the control signal
        command_msg = Float64MultiArray()
        command_msg.data = final_action.tolist()
        self.publisher.publish(command_msg)
        
        # Publish error for monitoring/debugging
        error_msg = Float64MultiArray()
        error_msg.data = error.tolist()
        self.error_publisher.publish(error_msg)
        
        self.get_logger().debug(f'CNN: {cnn_output}, Error: {error}, Total: {final_action}')

    def _find_weights_file(self):
        """Find the weights file in common locations"""
        potential_paths = [
            'cnn_regulator_weights.pth',  # Current directory
            Path.home() / 'Desktop' / 'project_cnn_regulation' / 'cnn_regulator_weights.pth',  # Home desktop
            Path(__file__).parent / 'cnn_regulator_weights.pth',  # Same directory as this file
            Path.cwd() / 'cnn_regulator_weights.pth',  # Current working directory
        ]
        
        for path in potential_paths:
            if isinstance(path, str):
                path = Path(path)
            if path.exists():
                self.get_logger().info(f'Found weights file at: {path}')
                return str(path)
        
        return None

def main(args=None):
    rclpy.init(args=args)
    
    node = CNNControllerNode()
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
