#!/usr/bin/env python3
"""
测试 VisionPro 数据接收
"""
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from visionpro_control.core.visionpro_bridge import VisionProBridge


def test_bridge():
    """测试 VisionPro 连接和数据接收"""
    print("="*60)
    print("测试 1: VisionPro Bridge")
    print("="*60)
    
    # 配置
    AVP_IP = input("请输入 VisionPro IP 地址 [默认: 10.31.181.201]: ").strip()
    if not AVP_IP:
        AVP_IP = "10.31.181.201"
    
    use_right = input("使用右手? (y/n) [默认: y]: ").strip().lower()
    USE_RIGHT_HAND = use_right != 'n'
    
    print(f"\n连接到: {AVP_IP}")
    print(f"使用{'右' if USE_RIGHT_HAND else '左'}手\n")
    
    try:
        # 创建 bridge
        bridge = VisionProBridge(avp_ip=AVP_IP, use_right_hand=USE_RIGHT_HAND)
        bridge.start()
        
        print("等待连接稳定...")
        time.sleep(2.0)
        
        print("\n开始接收数据 (10秒)...\n")
        
        for i in range(100):  # 10秒，每0.1秒一次
            position, rotation = bridge.get_hand_relative_to_head()
            pinch_distance = bridge.latest_data['pinch_distance']
            
            # 每10次打印一次
            if i % 10 == 0:
                print(f"[{i//10 + 1}s]")
                print(f"  位置 (相对头部): [{position[0]:6.3f}, {position[1]:6.3f}, {position[2]:6.3f}] m")
                print(f"  旋转矩阵:")
                for row in rotation:
                    print(f"    [{row[0]:6.3f}, {row[1]:6.3f}, {row[2]:6.3f}]")
                print(f"  捏合距离: {pinch_distance:.4f} m")
                print(f"  捏合状态: {'✓ 捏合' if bridge.get_pinch_state() else '✗ 未捏合'}")
                print()
            
            time.sleep(0.1)
        
        print("✓ 测试通过：数据接收正常")
        bridge.stop()
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_bridge()
    sys.exit(0 if success else 1)