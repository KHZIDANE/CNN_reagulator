import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import csv
import collections

class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')
        
        # Target positions for the 6 joints
        self.target_q = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 0.5])
        
        # PD gains for each joint
        self.Kp = np.array([50.0, 100.0, 50.0, 20.0, 10.0, 10.0])
        self.Kd = np.array([5.0, 10.0, 5.0, 2.0, 1.0, 1.0])
        
        self.seq_len = 10
        self.state_history = collections.deque(maxlen=self.seq_len)
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        self.csv_file = open('training_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Header: seq_len * 12 states (q1..6, dq1..6) + 6 actions (tau1..6)
        header = []
        for i in range(self.seq_len):
            for j in range(1, 7):
                header.extend([f'q{j}_{i}'])
            for j in range(1, 7):
                header.extend([f'dq{j}_{i}'])
        for j in range(1, 7):
            header.append(f'tau{j}')
        self.csv_writer.writerow(header)
        
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/effort_controllers/commands',
            10
        )
        
        self.get_logger().info('Data Collector (PD Controller) started.')

    def state_callback(self, msg):
        try:
            indices = [msg.name.index(name) for name in self.joint_names]
        except ValueError:
            return
            
        q = np.array([msg.position[idx] for idx in indices])
        dq = np.array([msg.velocity[idx] for idx in indices])
        
        current_state = np.concatenate((q, dq)).tolist()
        self.state_history.append(current_state)
        
        # Calculate PD control
        error = self.target_q - q
        error_dot = -dq
        
        tau = self.Kp * error + self.Kd * error_dot
        
        # Publish effort command
        command_msg = Float64MultiArray()
        command_msg.data = tau.tolist()
        self.publisher.publish(command_msg)
        
        # If we have a full sequence, save to CSV
        if len(self.state_history) == self.seq_len:
            row = []
            for state in self.state_history:
                row.extend(state)
            row.extend(tau.tolist())
            self.csv_writer.writerow(row)

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
