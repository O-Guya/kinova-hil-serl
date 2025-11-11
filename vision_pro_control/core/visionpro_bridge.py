import numpy as np
from VisionProTeleop.avp_stream import VisionProStreamer
import threading
import time

class VisionProBridge:

    def __init__(self, avp_ip: str, use_right_hand: bool = True):
        self.avp_ip = avp_ip
        self.use_right_hand = use_right_hand
        self.streamer = None
        self.running = False
        self.thread = None

        self.lastest_data = {
            'head_pose' : np.eye(4),
            'wrist_post' : np.eye(4),
            'pinch_distance' : 1.0,
            'wrist_roll' : 0.0,
            'timestamp' : time.time()
        }
        self.data_lock = threading.Lock()

    def start(self):
        print(f"Connecting VisionPro(ip:{self.avp_ip})")
        self.streamer = VisionProStreamer(ip=self.avp_ip, record=False)
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print("VisionPro data stream started!")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("VisionPro data stream stopped!")

    def _update_loop(self):
        while self.running:
            try:
                r = self.streamer.latest

                with self.data_lock:
                    self.lastest_data['head_pose'] = r['head'][0]

                    if self.use_right_hand:
                        self.lastest_data['wrist_pose'] = r['right_wrist'][0]
                        self.lastest_data['pinch_distance'] = r['right_pinch_distance']
                        self.lastest_data['wrist_roll'] = r['right_wrist_roll']
                    else:
                        self.lastest_data['wrist_pose'] = r['left_wrist'][0]
                        self.lastest_data['pinch_distance'] = r['left_pinch_distance']
                        self.lastest_data['wrist_roll'] = r['left_wrist_roll']
                self.lastest_data['timestamp'] = time.time()

            except Exception as e:
                print("Update Error!")
            
            time.sleep(0.05)

    def get_hand_relative_to_head(self):
        with self.data_lock:
            head_pose = self.lastest_data['head_pose'].copy
            wrist_pose = self.lastest_data['wrist_pose'].copy

        head_inv = np.linalg.inv(head_pose)
        relative_pose = head_inv @ wrist_pose

        position = relative_pose[:3,3]
        rotation = relative_pose[:3,:3]

        return position, rotation
    
    def get_pinch_state(self, threshold: float = 0.02):

        with self.data_lock:
            pinch_distance = self.lastest_data['pinch_distance']

        return pinch_distance < threshold
