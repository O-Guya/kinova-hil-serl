"""
坐标映射器：VisionPro 手部空间 → 机械臂控制空间
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy.spatial.transform import Rotation as R


class CoordinateMapper:
    """将 VisionPro 手部相对位姿映射到机械臂控制指令"""
    
    def __init__(self, calibration_file: Path):
        """
        Args:
            calibration_file: 标定文件路径
        """
        self.calibration_file = Path(calibration_file)
        
        # 加载标定数据
        self._load_calibration()
        
        # 映射参数
        self.position_gain = 1.0  # 位置速度增益
        self.rotation_gain = 1.0  # 角速度增益
        self.max_linear_velocity = 0.3  # m/s
        self.max_angular_velocity = 1.0  # rad/s
        self.deadzone_radius = 0.02  # 死区半径 (m)
        
        # 滤波器状态
        self.filter_alpha = 0.3  # 低通滤波系数 (0-1, 越小越平滑)
        self.filtered_linear_vel = np.zeros(3)
        self.filtered_angular_vel = np.zeros(3)
        
    def _load_calibration(self):
        """加载标定数据"""
        import yaml
        
        if not self.calibration_file.exists():
            raise FileNotFoundError(f"标定文件不存在: {self.calibration_file}")
        
        with open(self.calibration_file, 'r') as f:
            data = yaml.safe_load(f)
        
        bounds = data['workspace_bounds']
        points = data['calibration_points']
        
        # 工作空间参数
        self.center = np.array(bounds['center'])
        self.x_range = np.array(bounds['x_range'])
        self.y_range = np.array(bounds['y_range'])
        self.z_range = np.array(bounds['z_range'])
        self.x_span = bounds['x_span']
        self.y_span = bounds['y_span']
        self.z_span = bounds['z_span']
        
        # 标定姿态（用于相对旋转）
        if 'center_rotation' in points and points['center_rotation'] is not None:
            self.center_rotation = np.array(points['center_rotation'])
        else:
            self.center_rotation = np.eye(3)
            print("警告: 未找到 center_rotation，使用单位矩阵")
        
        print(f"✓ 已加载标定数据: {self.calibration_file}")
        print(f"  工作空间中心: {self.center}")
        print(f"  工作空间范围: X={self.x_span:.3f}, Y={self.y_span:.3f}, Z={self.z_span:.3f}")
        
    def normalize_position(self, position: np.ndarray) -> np.ndarray:
        """
        将手部位置归一化到 [-1, 1] 范围
        超出边界会饱和到 ±1
        Args:
            position: 手部相对头部的位置 (3,)
        Returns:
            normalized: 归一化位置 (3,)
        """
        # 计算相对中心的偏移
        offset = position - self.center

        # 归一化到 [-1, 1]
        normalized = np.zeros(3)
        normalized[0] = offset[0] / (self.x_span / 2.0)  # X 方向
        normalized[1] = offset[1] / (self.y_span / 2.0)  # Y 方向
        normalized[2] = offset[2] / (self.z_span / 2.0)  # Z 方向

        # 饱和限制（方案 1）
        normalized = np.clip(normalized, -1.0, 1.0)
        return normalized
        
    def apply_deadzone(self, normalized_pos: np.ndarray) -> np.ndarray:
        """
        应用死区：中心附近不动
        Args:
            normalized_pos: 归一化位置 (3,)
        Returns:
            位置（应用死区后）
        """
        # 计算到中心的距离
        distance = np.linalg.norm(normalized_pos)
        
        # 归一化的死区半径
        deadzone_norm = self.deadzone_radius / (np.mean([self.x_span, self.y_span, self.z_span]) / 2.0)
        
        if distance < deadzone_norm:
            return np.zeros(3)
        else:
            # 线性缩放：死区边缘为 0，边界为 1
            scale = (distance - deadzone_norm) / (1.0 - deadzone_norm)
            scale = min(scale, 1.0)
            return normalized_pos * scale
            
    def compute_relative_rotation(self, current_rotation: np.ndarray) -> np.ndarray:
        """
        计算相对于标定姿态的旋转差
        Args:
            current_rotation: 当前手腕旋转矩阵 (3, 3)
        Returns:
            相对旋转的轴角表示 (3,) [rx, ry, rz]
        """
        # 计算相对旋转: R_relative = R_center^-1 @ R_current
        R_center_inv = self.center_rotation.T  # 旋转矩阵的逆 = 转置
        R_relative = R_center_inv @ current_rotation

        # 转换为 scipy Rotation 对象
        rot = R.from_matrix(R_relative)
        
        # 转换为轴角表示（旋转向量）
        rotvec = rot.as_rotvec()
        
        return rotvec
        
    def position_to_linear_velocity(self, position: np.ndarray) -> np.ndarray:
        """
        将手部位置映射到线速度
        
        Args:
            position: 手部相对头部的位置 (3,)
        Returns:
            linear_velocity: 线速度 (3,) [vx, vy, vz]
        """
        # 1. 归一化
        normalized = self.normalize_position(position)
        
        # 2. 应用死区
        normalized = self.apply_deadzone(normalized)
        
        # 3. 计算速度（简单比例控制）
        velocity = normalized * self.position_gain
        
        # 4. 限制最大速度
        speed = np.linalg.norm(velocity)
        if speed > self.max_linear_velocity:
            velocity = velocity / speed * self.max_linear_velocity
            
        # 5. 低通滤波
        self.filtered_linear_vel = (self.filter_alpha * velocity + 
                                    (1 - self.filter_alpha) * self.filtered_linear_vel)
        
        return self.filtered_linear_vel
        
    def rotation_to_angular_velocity(self, rotation: np.ndarray) -> np.ndarray:
        """
        将手腕姿态映射到角速度
        
        Args:
            rotation: 当前手腕旋转矩阵 (3, 3)
        Returns:
            angular_velocity: 角速度 (3,) [wx, wy, wz]
        """
        # 1. 计算相对旋转
        rotvec = self.compute_relative_rotation(rotation)
        
        # 2. 转换为角速度（简单比例控制）
        angular_velocity = rotvec * self.rotation_gain
        
        # 3. 限制最大角速度
        angular_speed = np.linalg.norm(angular_velocity)
        if angular_speed > self.max_angular_velocity:
            angular_velocity = angular_velocity / angular_speed * self.max_angular_velocity
            
        # 4. 低通滤波
        self.filtered_angular_vel = (self.filter_alpha * angular_velocity + 
                                     (1 - self.filter_alpha) * self.filtered_angular_vel)
        
        return self.filtered_angular_vel
        
    def map_to_twist(self, position: np.ndarray, rotation: np.ndarray) -> Dict:
        """
        将手部位姿映射到 Twist 消息
        
        Args:
            position: 手部相对头部的位置 (3,)
            rotation: 手腕旋转矩阵 (3, 3)
        Returns:
            twist: 字典格式的 Twist 消息
                {
                    'linear': {'x': float, 'y': float, 'z': float},
                    'angular': {'x': float, 'y': float, 'z': float}
                }
        """
        # 计算速度
        linear_vel = self.position_to_linear_velocity(position)
        angular_vel = self.rotation_to_angular_velocity(rotation)
        
        # 构造 Twist 消息格式
        twist = {
            'linear': {
                'x': float(linear_vel[0]),
                'y': float(linear_vel[1]),
                'z': float(linear_vel[2])
            },
            'angular': {
                'x': float(angular_vel[0]),
                'y': float(angular_vel[1]),
                'z': float(angular_vel[2])
            }
        }
        
        return twist
        
    def reset_filter(self):
        """重置滤波器状态"""
        self.filtered_linear_vel = np.zeros(3)
        self.filtered_angular_vel = np.zeros(3)
        
    def set_gains(self, position_gain: float = None, rotation_gain: float = None):
        """
        设置控制增益
        
        Args:
            position_gain: 位置增益
            rotation_gain: 旋转增益
        """
        if position_gain is not None:
            self.position_gain = position_gain
            print(f"位置增益: {self.position_gain}")
            
        if rotation_gain is not None:
            self.rotation_gain = rotation_gain
            print(f"旋转增益: {self.rotation_gain}")
            
    def set_velocity_limits(self, max_linear: float = None, max_angular: float = None):
        """
        设置速度限制
        
        Args:
            max_linear: 最大线速度 (m/s)
            max_angular: 最大角速度 (rad/s)
        """
        if max_linear is not None:
            self.max_linear_velocity = max_linear
            print(f"最大线速度: {self.max_linear_velocity} m/s")
            
        if max_angular is not None:
            self.max_angular_velocity = max_angular
            print(f"最大角速度: {self.max_angular_velocity} rad/s")
            
    def set_filter_alpha(self, alpha: float):
        """
        设置滤波系数
        
        Args:
            alpha: 滤波系数 (0-1)，越小越平滑，但延迟越大
        """
        self.filter_alpha = np.clip(alpha, 0.0, 1.0)
        print(f"滤波系数: {self.filter_alpha}")
        
    def print_info(self):
        """打印映射器配置信息"""
        print("\n" + "="*50)
        print("坐标映射器配置:")
        print(f"  标定文件: {self.calibration_file}")
        print(f"  工作空间中心: {self.center}")
        print(f"  工作空间范围: X={self.x_span:.3f}, Y={self.y_span:.3f}, Z={self.z_span:.3f}")
        print(f"  位置增益: {self.position_gain}")
        print(f"  旋转增益: {self.rotation_gain}")
        print(f"  最大线速度: {self.max_linear_velocity} m/s")
        print(f"  最大角速度: {self.max_angular_velocity} rad/s")
        print(f"  死区半径: {self.deadzone_radius} m")
        print(f"  滤波系数: {self.filter_alpha}")
        print("="*50 + "\n")