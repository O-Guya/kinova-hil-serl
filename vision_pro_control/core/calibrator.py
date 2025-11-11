import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional
import time

class WorkspaceCalibrator:
    def __init__(self):
        self.calibration_points = {
            'center': None,
            'front': None,
            'back': None,
            'left': None,
            'right': None,
            'top': None,
            'bottom': None,
        }

        self.samples_per_point = []

    def add_sample(self, position: np.nparray):
        self.samples_per_point.append(position.copy())

    def save_point(self, point_name: str):
        if point_name not in self.calibration_points:
            print(f"ERROR:[UNKOWN POINT] '{point_name}'")
            return False
            
        if len(self.samples_per_point) == 0:
            print("ERROR:[NO SAMPLE POINT]")
            return False
        
        # 计算平均
        avg_position = np.mean(self.samples_per_point, axis=0)
        self.calibration_points[point_name] = avg_position

        print(f"SAVE '{point_name}': {avg_position}")
        print(f"BASED {len(self.samples_per_point)} SAMPLES")
        
        # 清空样本
        self.samples_per_point = []
        return True

    def clear_samples(self):
        self.samples_per_point = []
        print("SAMPLES CLEARED")

    def is_complete(self) -> bool:
        return all(v is not None for v in self.calibration_points.values())
    
    def get_workspace_bounds(self):
        if not self.is_complete():
            return None
        
        points = self.calibration_points

        # 计算范围
        x_range = [points['back'][0], points['front'][0]]
        y_range = [points['right'][1], points['left'][1]]
        z_range = [points['bottom'][2], points['top'][2]]

        center = points['center']

        return {
            'center': center.tolist(),
            'x_range': x_range,  # forward/backward
            'y_range': y_range,  # left/right
            'z_range': z_range,  # up/down
            'x_span': abs(x_range[1] - x_range[0]),
            'y_span': abs(y_range[1] - y_range[0]),
            'z_span': abs(z_range[1] - z_range[0]),
        }
    
    def save_to_file(self, filepath: Path, overwrite: bool = False):

        filepath = Path(filepath)

        if filepath.exists() and not overwrite:
            print(f"File existed: {filepath}")
            print("Please set overwrite = True")
            return False
        
        if not self.is_complete():
            print("Calibration hasn't completed.")
            return False
        
        # 准备数据
        data = {
            'calibration_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'workspace_bounds': self.get_workspace_bounds(),
            'calibration_points': {
                k: v.tolist() for k, v in self.calibration_points.items()
            }
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 保存
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"Data save to {filepath}")

        return True
    
    def load_from_file(cls, filepath: Path) -> Optional['WorkspaceCalibrator']:
        """从 YAML 文件加载标定数据"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"错误: 文件不存在 '{filepath}'")
            return None
            
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            
        calibrator = cls()
        
        # 恢复标定点
        for k, v in data['calibration_points'].items():
            calibrator.calibration_points[k] = np.array(v)
            
        print(f"✓ 从文件加载标定数据: {filepath}")
        print(f"  标定时间: {data['calibration_time']}")
        
        return calibrator
    
    def print_status(self):
        """打印标定状态"""
        print("\n" + "="*50)
        print("标定状态:")
        for name, pos in self.calibration_points.items():
            status = "✓" if pos is not None else "✗"
            print(f"  {status} {name:10s}: {pos if pos is not None else '未标定'}")
        
        if self.is_complete():
            bounds = self.get_workspace_bounds()
            print("\n工作空间:")
            print(f"  中心: {bounds['center']}")
            print(f"  X 范围: {bounds['x_span']:.3f} m")
            print(f"  Y 范围: {bounds['y_span']:.3f} m")
            print(f"  Z 范围: {bounds['z_span']:.3f} m")
        print("="*50 + "\n")