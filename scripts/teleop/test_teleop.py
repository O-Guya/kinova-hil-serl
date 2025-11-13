# test_visionpro_connection.py
import time
from VisionProTeleop.avp_stream import VisionProStreamer

def test_connection(ip="192.168.1.125"):
    print(f"测试连接到 VisionPro: {ip}")
    
    try:
        # 创建streamer
        streamer = VisionProStreamer(ip=ip)
        print("✓ Streamer 创建成功")
        
        # 等待数据
        print("\n等待接收数据 (10秒)...")
        for i in range(10):
            time.sleep(1)
            
            # 获取hand pose
            hand_data = streamer.get_hand_pose()
            head_data = streamer.get_head_pose()
            
            print(f"[{i+1}s] Hand: {hand_data is not None and hand_data.size > 0}, "
                  f"Head: {head_data is not None and head_data.size > 0}")
            
            if hand_data is not None and hand_data.size > 0:
                print(f"  Hand shape: {hand_data.shape}")
            if head_data is not None and head_data.size > 0:
                print(f"  Head shape: {head_data.shape}")
        
        print("\n✓ 测试完成")
        streamer.stop()
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.125"
    test_connection(ip)