#!/usr/bin/env python3
"""
测试标定数据加载和映射功能
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visionpro_control.core.calibrator import WorkspaceCalibrator
from visionpro_control.core.coordinate_mapper import CoordinateMapper


def test_calibration_load():
    """测试标定文件加载"""
    print("="*60)
    print("测试 2: 标定数据加载")
    print("="*60)
    
    calib_file = Path(__file__).parent.parent / "config" / "calibration.yaml"
    
    if not calib_file.exists():
        print(f"\n✗ 标定文件不存在: {calib_file}")
        print("请先运行标定程序: python scripts/run_calibration.py")
        return False
    
    try:
        # 加载标定数据
        print(f"\n加载标定文件: {calib_file}")
        calibrator = WorkspaceCalibrator.load_from_file(calib_file)
        
        # 打印状态
        calibrator.print_status()
        
        # 检查完整性
        if not calibrator.is_complete():
            print("\n✗ 测试失败: 标定数据不完整")
            return False
        
        print("✓ 测试通过：标定数据加载成功")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coordinate_mapping():
    """测试坐标映射"""
    print("\n" + "="*60)
    print("测试 3: 坐标映射")
    print("="*60)
    
    calib_file = Path(__file__).parent.parent / "config" / "calibration.yaml"
    
    try:
        # 创建 mapper
        print(f"\n初始化坐标映射器...")
        mapper = CoordinateMapper(calibration_file=calib_file)
        mapper.print_info()
        
        # 测试几个点
        test_cases = [
            ("中心点", np.array([0.35, 0.02, -0.05])),
            ("前方", np.array([0.50, 0.00, 0.00])),
            ("左侧", np.array([0.30, 0.30, 0.00])),
            ("上方", np.array([0.30, 0.00, 0.20])),
        ]
        
        print("\n测试位置映射:")
        print("-" * 60)
        
        for name, position in test_cases:
            # 测试旋转（使用单位矩阵）
            rotation = np.eye(3)
            
            # 映射
            twist = mapper.map_to_twist(position, rotation)
            
            linear = twist['linear']
            angular = twist['angular']
            
            print(f"\n{name}:")
            print(f"  输入位置: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
            print(f"  输出线速度: [{linear['x']:.3f}, {linear['y']:.3f}, {linear['z']:.3f}] m/s")
            print(f"  输出角速度: [{angular['x']:.3f}, {angular['y']:.3f}, {angular['z']:.3f}] rad/s")
        
        print("\n" + "-" * 60)
        print("✓ 测试通过：坐标映射正常工作")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_calibration_load()
    if success1:
        success2 = test_coordinate_mapping()
    else:
        success2 = False
    
    print("\n" + "="*60)
    if success1 and success2:
        print("所有测试通过！")
        sys.exit(0)
    else:
        print("部分测试失败")
        sys.exit(1)