#!/usr/bin/env python3

"""
CNN Regulator Target Position UI
User-friendly interface to control robot via CNN regulator
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import time
import subprocess
from datetime import datetime
import sys

class TargetControlUI(Node):
    def __init__(self):
        super().__init__('target_control_ui')
        
        # Current state
        self.current_position = np.zeros(6)
        self.current_velocity = np.zeros(6)
        self.target_position = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 0.5])
        
        # Joint names
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publisher for target position
        self.target_pub = self.create_publisher(
            Float64MultiArray,
            '/effort_controllers/commands',
            10
        )
        
        # Pre-defined targets
        self.targets = {
            'home': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'training': np.array([1.0, -1.0, 1.0, -1.0, 1.0, 0.5]),
            'extended': np.array([2.0, -0.5, 0.5, -1.0, 1.0, 0.5]),
            'compact': np.array([0.5, -1.5, 1.5, -1.0, 0.5, 0.2]),
            'demo': np.array([1.5, -0.8, 0.8, -1.2, 1.2, 0.3]),
        }
    
    def joint_state_callback(self, msg):
        """Update current position"""
        try:
            indices = [msg.name.index(name) for name in self.joint_names]
            self.current_position = np.array([msg.position[idx] for idx in indices])
            self.current_velocity = np.array([msg.velocity[idx] for idx in indices])
        except ValueError:
            pass
    
    def set_target(self, target):
        """Set target position via ROS parameter"""
        self.target_position = target.copy()
        
        # Format for ROS param set command
        target_str = "[" + ", ".join(f"{x:.2f}" for x in target) + "]"
        
        # Use ros2 param set to update the controller
        try:
            result = subprocess.run(
                ['ros2', 'param', 'set', '/cnn_regulator_node', 'target_position', target_str],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                self.get_logger().info(f"✓ Target updated: {target}")
                return True
            else:
                print(f"\n⚠ Failed to set parameter: {result.stderr}")
                print(f"Make sure CNN controller is running: ros2 run cnn_regulator ros_controller")
                return False
        except Exception as e:
            print(f"\n⚠ Error setting parameter: {e}")
            print(f"Make sure CNN controller is running: ros2 run cnn_regulator ros_controller")
            return False

    
    def get_error(self):
        """Calculate error to target"""
        return self.target_position - self.current_position
    
    def print_header(self):
        """Print UI header"""
        print("\n" + "="*60)
        print("CNN REGULATOR TARGET POSITION CONTROL")
        print("="*60)
    
    def print_state(self):
        """Print current state"""
        error = self.get_error()
        print("\n[CURRENT STATE]")
        print(f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        print("\nJoint Positions:")
        for i, (name, pos) in enumerate(zip(self.joint_names, self.current_position)):
            target = self.target_position[i]
            print(f"  {i+1}. {name:20s} | pos: {pos:7.3f} rad | target: {target:7.3f} | error: {error[i]:7.3f}")
        
        print("\nJoint Velocities (rad/s):")
        for i, (name, vel) in enumerate(zip(self.joint_names, self.current_velocity)):
            print(f"  {i+1}. {name:20s} | {vel:+.6f}")
        
        error_magnitude = np.linalg.norm(error)
        print(f"\nTotal Position Error: {error_magnitude:.4f} rad")
        
        # Move indicator
        if error_magnitude < 0.1:
            status = "✓ AT TARGET"
        elif np.max(np.abs(self.current_velocity)) > 0.01:
            status = "→ MOVING"
        else:
            status = "○ IDLE"
        print(f"Status: {status}")
    
    def show_menu(self):
        """Show interactive menu"""
        while True:
            self.print_header()
            self.print_state()
            
            print("\n[PRESET TARGETS]")
            presets = list(self.targets.keys())
            for i, preset in enumerate(presets, 1):
                print(f"  {i}. {preset.upper()}")
                print(f"     {self.targets[preset]}")
            
            print("\n[OPTIONS]")
            print(f"  {len(presets)+1}. CUSTOM (enter values)")
            print(f"  {len(presets)+2}. REFRESH state")
            print(f"  {len(presets)+3}. EXIT")
            
            print("\n" + "-"*60)
            choice = input("Select target (1-{}) or option: ".format(len(presets)+3)).strip()
            
            if choice.isdigit():
                choice = int(choice)
                
                if 1 <= choice <= len(presets):
                    preset = presets[choice-1]
                    if self.set_target(self.targets[preset]):
                        print(f"\n✓ Set target to: {preset.upper()}")
                    time.sleep(1)
                
                elif choice == len(presets) + 1:
                    self.custom_target()
                
                elif choice == len(presets) + 2:
                    print("Refreshing state...")
                    time.sleep(1)
                
                elif choice == len(presets) + 3:
                    print("\nExiting...")
                    break
                else:
                    print("Invalid choice!")
            else:
                print("Invalid input!")
    
    def custom_target(self):
        """Allow user to enter custom target"""
        print("\n" + "="*60)
        print("CUSTOM TARGET")
        print("="*60)
        print("\nEnter target positions for each joint (in radians)")
        print("Current: " + str(np.round(self.current_position, 3)))
        print("Target:  " + str(np.round(self.target_position, 3)))
        print("\nType values separated by commas (e.g., 1.0, -1.0, 1.0, -1.0, 1.0, 0.5)")
        print("Or press Enter to cancel")
        
        user_input = input("\nTarget values: ").strip()
        
        if not user_input:
            print("Cancelled")
            return
        
        try:
            values = [float(x.strip()) for x in user_input.split(',')]
            if len(values) != 6:
                print(f"Error: Expected 6 values, got {len(values)}")
                return
            
            target = np.array(values)
            if self.set_target(target):
                print(f"\n✓ Target updated to: {np.round(target, 3)}")
            time.sleep(1)
        
        except ValueError as e:
            print(f"Error parsing values: {e}")
            time.sleep(1)


def main(args=None):
    rclpy.init(args=args)
    
    print("\nInitializing CNN Regulator Target Control UI...")
    print("Waiting for ROS connection...\n")
    
    ui = TargetControlUI()
    
    # Spin once to get initial state
    for i in range(10):
        rclpy.spin_once(ui, timeout_sec=0.1)
        if not np.allclose(ui.current_position, 0):
            break
    
    if np.allclose(ui.current_position, 0):
        print("\n⚠ WARNING: No joint state received yet.")
        print("Make sure these are running in other terminals:")
        print("  1. ros2 launch mimo_process_gazebo sim.launch.py")
        print("  2. ros2 run cnn_regulator ros_controller")
        print("\nWaiting for connection...")
        
        for i in range(20):
            rclpy.spin_once(ui, timeout_sec=0.1)
            if not np.allclose(ui.current_position, 0):
                print("✓ Connection established!")
                break
    
    print("\nReady! Use the menu to select target positions.\n")
    
    try:
        ui.show_menu()
    except KeyboardInterrupt:
        print("\n\nShutdown requested")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        ui.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass  # Already shutdown


if __name__ == '__main__':
    main()
