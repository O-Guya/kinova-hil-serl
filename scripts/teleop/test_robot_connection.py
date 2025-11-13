#!/usr/bin/env python3
"""
测试 ROS2 和 Kinova 机械臂连接
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import time


class RobotConnectionTest(Node):
    def __init__(self):
        super().__init__('robot_connection_test')
        
        self.get_logger().info('初始化机械臂连接测试...')
        
        # 订阅关节状态
        self.joint_state_received = False
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # 创建 Twist 发布者
        self.twist_pub = self.create_publisher(
            Twist,
            '/twist_controller/commands',
            10
        )
        
        self.get_logger().info('节点已初始化')
        
    def joint_state_callback(self, msg):
        """关节状态回调"""
        if not self.joint_state_received:
            self.get_logger().info('✓ 接收到关节状态数据')
            self.get_logger().info(f'  关节数量: {len(msg.name)}')
            self.get_logger().info(f'  关节名称: {msg.name}')
            self.joint_state_received = True
            
    def test_twist_publish(self):
        """测试发布 Twist 消息"""
        self.get_logger().info('\n测试 Twist 发布...')
        
        # 发布零速度
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        
        for i in range(10):
            self.twist_pub.publish(twist)
            self.get_logger().info(f'  发布 Twist 消息 {i+1}/10')
            time.sleep(0.1)
            
        self.get_logger().info('✓ Twist 消息发布成功')


def main():
    print("="*60)
    print("测试 4: 机械臂连接")
    print("="*60)
    print("\n请确保已启动 kortex_driver:")
    print("  ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10\n")
    
    input("按 Enter 继续测试...")
    
    rclpy.init()
    
    try:
        node = RobotConnectionTest()
        
        # 等待接收关节状态
        print("\n等待接收关节状态数据 (5秒)...")
        start_time = time.time()
        
        while time.time() - start_time < 5.0:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.joint_state_received:
                break
        
        if not node.joint_state_received:
            print("\n✗ 测试失败: 未接收到关节状态")
            print("  检查项:")
            print("  1. kortex_driver 是否已启动?")
            print("  2. 机械臂 IP 是否正确?")
            print("  3. 网线是否连接?")
            return False
        
        # 测试发布 Twist
        node.test_twist_publish()
        
        print("\n✓ 测试通过：机械臂连接正常")
        
        # 检查可用的 topics
        print("\n可用的控制 topics:")
        import subprocess
        result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'twist' in line.lower() or 'gripper' in line.lower():
                print(f"  {line}")
        
        node.destroy_node()
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)