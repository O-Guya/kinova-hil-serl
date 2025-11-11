import sys
import termios
import tty
import select

class KeyboardMonitor:

    def __init__(self):
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self
        
    def __exit__(self, type, value, traceback):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
    def get_key(self, timeout: float = 0.0) -> str:
        """
        非阻塞获取按键
        Args:
            timeout: 超时时间（秒），0 表示立即返回
        Returns:
            按键字符，无按键返回空字符串
        """
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return ''

