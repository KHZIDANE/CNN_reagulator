import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data
import numpy as np

class ClassicControllerNode(Node):
    def __init__(self):
        super().__init__('classic_controller_node')
        
        # PID control gains
        self.declare_parameter('kp', [50.0, 50.0, 50.0, 20.0, 10.0, 2.0])
        self.declare_parameter('kd', [10.0, 10.0, 10.0, 5.0, 1.0, 1.0])
        
        self.kp = np.array(self.get_parameter('kp').get_parameter_value().double_array_value)
        self.kd = np.array(self.get_parameter('kd').get_parameter_value().double_array_value)
        
        # Target position
        self.target_q = np.zeros(6)
        
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        # Subscribers
        self.state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10
        )
        
        self.target_sub = self.create_subscription(
            Float64MultiArray,
            '/cnn_regulator/target_position', # Subscribes to the GUI's topic
            self.target_callback,
            10
        )
        
        # Publisher
        self.effort_pub = self.create_publisher(
            Float64MultiArray,
            '/effort_controllers/commands',
            qos_profile_sensor_data
        )
        
        self.get_logger().info('Classic PID Controller started. Connect using target_gui.')

    def target_callback(self, msg):
        if len(msg.data) == 6:
            self.target_q = np.array(msg.data)
            self.get_logger().info(f'Target updated to: {np.round(self.target_q, 3)}')

    def state_callback(self, msg):
        try:
            indices = [msg.name.index(name) for name in self.joint_names]
            current_q = np.array([msg.position[idx] for idx in indices])
            current_dq = np.array([msg.velocity[idx] for idx in indices])
        except ValueError:
            return

        # PD control law: tau = Kp * (q_des - q) - Kd * dq
        error = self.target_q - current_q
        tau = self.kp * error - self.kd * current_dq
        
        # Saturation to avoid explosions
        max_effort = 150.0
        tau = np.clip(tau, -max_effort, max_effort)
        
        command_msg = Float64MultiArray()
        command_msg.data = tau.tolist()
        self.effort_pub.publish(command_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ClassicControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
