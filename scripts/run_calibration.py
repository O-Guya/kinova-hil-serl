#!/usr/bin/env python3
"""
VisionPro 工作空间标定脚本
"""
import sys
import time
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))


from vision_pro_control.core.visionpro_bridge import VisionProBridge
from vision_pro_control.core.calibrator import WorkspaceCalibrator
from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor


def print_instructions():
    """打印使用说明"""
    print("\n" + "="*60)
    print("VisionPro 工作空间标定")
    print("="*60)
    print("\n标定步骤:")
    print("  1. 将手移动到指定位置")
    print("  2. 按住对应按键采样（至少 1 秒）")
    print("  3. 完成所有 7 个点的标定")
    print("  4. 保存标定文件")
    print("\n按键说明:")
    print("  '0' - 标定中心点（舒适的操作位置）")
    print("  '1' - 标定最前点")
    print("  '2' - 标定最后点")
    print("  '3' - 标定最左点")
    print("  '4' - 标定最右点")
    print("  '5' - 标定最高点")
    print("  '6' - 标定最低点")
    print("  'c' - 清空当前采样")
    print("  's' - 保存标定文件")
    print("  'p' - 打印当前状态")
    print("  'q' - 退出")
    print("="*60 + "\n")


def main():
    # 配置
    AVP_IP = "10.31.181.201"  # 修改为你的 VisionPro IP
    USE_RIGHT_HAND = True

    SAVE_PATH = Path(__file__).parent.parent / "config" / "calibration.yaml"
    
    # 检查是否覆盖
    overwrite = False
    if SAVE_PATH.exists():
        print(f"\n警告: 标定文件已存在: {SAVE_PATH}")
        response = input("是否覆盖? (y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return
        overwrite = True
        print()
    
    # 初始化
    print("初始化 VisionPro 连接...")
    bridge = VisionProBridge(AVP_IP, use_right_hand=USE_RIGHT_HAND)
    calibrator = WorkspaceCalibrator()
    
    try:
        bridge.start()
        time.sleep(1.0)  # 等待连接稳定
        
        print_instructions()
        
        # 键盘监听
        with KeyboardMonitor() as kb:
            key_to_point = {
                '0': 'center',
                '1': 'front',
                '2': 'back',
                '3': 'left',
                '4': 'right',
                '5': 'top',
                '6': 'bottom',
            }
            
            current_key = None
            key_press_start = None
            
            print("准备就绪! 开始标定...")
            
            while True:
                # 获取按键
                key = kb.get_key(timeout=0.01)
                
                # 获取当前手部位置
                position, _ = bridge.get_hand_relative_to_head()
                
                # 处理按键
                if key:
                    if key == 'q':
                        print("\n退出标定")
                        break
                        
                    elif key == 'c':
                        calibrator.clear_samples()
                        
                    elif key == 'p':
                        calibrator.print_status()
                        
                    elif key == 's':
                        if calibrator.is_complete():
                            success = calibrator.save_to_file(SAVE_PATH, overwrite=True)
                            if success:
                                print("\n标定完成! 文件已保存")
                                break
                        else:
                            print("\n错误: 标定未完成，请完成所有 7 个点")
                            calibrator.print_status()
                            
                    elif key in key_to_point:
                        if current_key != key:
                            current_key = key
                            key_press_start = time.time()
                            print(f"\n>>> 开始采样 '{key_to_point[key]}' 点...")
                            
                # 按键持续按下时采样
                if current_key and current_key in key_to_point:
                    # 检查按键是否还在按下（简化：假设一直按）
                    elapsed = time.time() - key_press_start
                    
                    if elapsed < 2.0:  # 持续采样 2 秒
                        calibrator.add_sample(position)
                        
                        # 显示进度
                        if int(elapsed * 10) % 5 == 0:  # 每 0.5 秒打印
                            print(f"  采样中... {len(calibrator.samples_per_point)} 个样本", end='\r')
                    else:
                        # 采样完成，保存
                        point_name = key_to_point[current_key]
                        calibrator.save_point(point_name)
                        current_key = None
                        key_press_start = None
                        
                        # 检查是否全部完成
                        if calibrator.is_complete():
                            print("\n所有点已标定完成! 按 's' 保存")
                            calibrator.print_status()
                
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n\n用户中断")
        
    finally:
        bridge.stop()
        print("连接已关闭")


if __name__ == "__main__":
    print(f"Path:{Path(__file__)}")