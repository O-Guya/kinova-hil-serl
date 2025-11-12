"""
机械臂指挥模块：发送控制指令到 Kinova Gen3
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
import numpy as np
import threading
import time



class RobotCommander(Node):
    """ROS2 机械臂控制器"""
    
    def __init__(self, robot_ip: str, node_name: str = 'robot_commander'):
        """
        Args:
            robot_ip: 机械臂 IP 地址（用于日志，实际连接由 launch 文件处理）
            node_name: ROS2 节点名称
        """
        super().__init__(node_name)
        
        self.robot_ip = robot_ip
        self.get_logger().info(f'初始化机械臂控制器: {robot_ip}')
        
        # 安全限制参数（保守设置，用于打通系统）
        self.safety_max_linear_vel = 0.1   # 最大线速度 0.1 m/s（很慢）
        self.safety_max_angular_vel = 0.3  # 最大角速度 0.3 rad/s
        self.enable_safety_check = True    # 启用安全检查
        
        # 创建 Twist 发布者
        self.twist_publisher = self.create_pulisher(
            Twist,
            '/twist_controller/commands',
            10
        )
        
        # 创建夹爪 Action 客户端
        self.gripper_action_client = ActionClient(
            self,
            GripperCommand,
            '/robotiq_gripper_controller/gripper_cmd'
        )
        
        # 订阅关节状态（用于监控，可选）
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # 状态变量
        self.current_joint_states = None
        self.last_twist_time = time.time()
        self.is_emergency_stopped = False
        
        self.get_logger().info('机械臂控制器初始化完成')
        
    def joint_state_callback(self, msg):
        """关节状态回调（用于监控）"""
        self.current_joint_states = msg
        
    def safety_check_twist(self, twist_dict: dict) -> dict:
        """
        安全检查：限制速度到安全范围
        Args:
            twist_dict: 原始 twist 字典
        Returns:
            安全限制后的 twist 字典
        """
        if not self.enable_safety_check:
            return twist_dict
            
        linear = np.array([
            twist_dict['linear']['x'],twist_dict['linear']['y'],twist_dict['linear']['z']
        ])
        
        angular = np.array([
            twist_dict['angular']['x'],twist_dict['angular']['y'],twist_dict['angular']['z']
        ])
        np.linalg.norm(angular)
        linear_speed = np.linalg.norm(linear)
        angular_speed = np.linalg.norm(angular)
        
        # 限制线速度
        if linear_speed > self.safety_max_linear_vel:
            scale = self.safety_max_linear_vel / linear_speed
            linear = linear * scale
            self.get_logger().warn(f'线速度超限，已缩放: {linear_speed:.3f} -> {self.safety_max_linear_vel:.3f} m/s')
            
        # 限制角速度
        if angular_speed > self.safety_max_angular_vel:
            scale = self.safety_max_angular_vel / angular_speed
            angular = angular * scale
            self.get_logger().warn(f'角速度超限，已缩放: {angular_speed:.3f} -> {self.safety_max_angular_vel:.3f} rad/s')
            
        # 构造安全的 twist
        safe_twist = {
            'linear': {'x': float(linear[0]), 'y': float(linear[1]), 'z': float(linear[2])},
            'angular': {'x': float(angular[0]), 'y': float(angular[1]), 'z': float(angular[2])}
        }
        
        return safe_twist
        
    def send_twist(self, twist_dict: dict):
        """
        发送 Twist 指令到机械臂
        Args:
            twist_dict: Twist 字典，格式：
                {
                    'linear': {'x': float, 'y': float, 'z': float},
                    'angular': {'x': float, 'y': float, 'z': float}
                }
        """
        if self.is_emergency_stopped:
            self.get_logger().warn('急停状态，忽略 Twist 指令')
            return
            
        # 安全检查
        safe_twist = self.safety_check_twist(twist_dict)
        
        msg = Twist()
        msg.linear.x = safe_twist['linear']['x']
        msg.linear.y = safe_twist['linear']['y']
        msg.linear.z = safe_twist['linear']['z']
        msg.angular.x = safe_twist['angular']['x']
        msg.angular.y = safe_twist['angular']['y']
        msg.angular.z = safe_twist['angular']['z']

        self.twist_publisher.publish(msg)
        
        self.last_twist_time = time.time()
        
    def send_zero_twist(self):
        """发送零速度（停止）"""
        zero_twist = {
            'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
        
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        
        if self.twist_publisher:
            self.twist_publisher.publish(msg)
        
    def emergency_stop(self):
        """急停：立即停止所有运动"""
        self.get_logger().error('!!! 急停触发 !!!')
        self.is_emergency_stopped = True
        
        # 发送零速度
        for _ in range(10):  # 连续发送确保收到
            self.send_zero_twist()
            time.sleep(0.01)
            
    def resume(self):
        """解除急停"""
        self.get_logger().info('解除急停')
        self.is_emergency_stopped = False
        
    def control_gripper(self, position: float, max_effort: float = 100.0):
        """
        控制夹爪
        
        Args:
            position: 夹爪位置，0.0=完全张开，0.8=完全闭合
            max_effort: 最大力度 (0-100)
        """
        if not self.gripper_action_client:
            self.get_logger().warn('夹爪 Action 客户端未初始化')
            return
            
        # 等待 Action 服务器
        if not self.gripper_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('夹爪 Action 服务器不可用')
            return
            
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command,max_effort = max_effort
        
        future = self.gripper_action_client.send_goal_async(goal)
        
        self.get_logger().info(f'夹爪指令已发送: position={position:.2f}')
        
    def set_safety_limits(self, max_linear: float = None, max_angular: float = None):
        """
        设置安全限制
        
        Args:
            max_linear: 最大线速度 (m/s)
            max_angular: 最大角速度 (rad/s)
        """
        if max_linear is not None:
            self.safety_max_linear_vel = max_linear
            self.get_logger().info(f'安全限制 - 最大线速度: {self.safety_max_linear_vel} m/s')
            
        if max_angular is not None:
            self.safety_max_angular_vel = max_angular
            self.get_logger().info(f'安全限制 - 最大角速度: {self.safety_max_angular_vel} rad/s')
            
    def enable_safety(self, enable: bool = True):
        """启用/禁用安全检查"""
        self.enable_safety_check = enable
        status = "启用" if enable else "禁用"
        self.get_logger().info(f'安全检查已{status}')
        
    def get_info(self) -> dict:
        """获取控制器信息"""
        return {
            'robot_ip': self.robot_ip,
            'emergency_stopped': self.is_emergency_stopped,
            'safety_enabled': self.enable_safety_check,
            'max_linear_vel': self.safety_max_linear_vel,
            'max_angular_vel': self.safety_max_angular_vel,
            'last_command_age': time.time() - self.last_twist_time
        }
    
    @classmethod
    def from_config(cls, config_dict: dict):
        """从配置字典创建 Commander"""
        commander = cls(robot_ip=config_dict['robot']['ip'])
        
        safety_config = config_dict['safety']
        commander.safety_max_linear_vel = safety_config['max_linear_velocity']
        commander.safety_max_angular_vel = safety_config['max_angular_velocity']
        commander.enable_safety_check = safety_config['enable_safety_check']
        
        return commander