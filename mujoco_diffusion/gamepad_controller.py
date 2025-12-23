import pygame
import numpy as np

class GamepadController:
    def __init__(self, max_pos_speed=0.5, deadzone=0.1):
        """
        初始化 Pygame 手柄控制器
        Args:
            max_pos_speed: 最大平移速度 (m/s)
            deadzone: 摇杆死区 (0.0-1.0)
        """
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[手柄] 已连接: {self.joystick.get_name()}")
        else:
            print("[手柄] 未找到手柄！将返回零动作。")
            
        self.max_pos_speed = max_pos_speed
        self.deadzone = deadzone
        
        
        # 录制按钮状态 (用于边缘检测)
        self.last_record_state = False
        # 重置按钮状态 (用于边缘检测)
        self.last_reset_state = False

    def get_action(self):
        """
        读取手柄输入并返回控制指令
        Returns:
            dpos (np.ndarray): 平移速度 [dx, dy, dz]
            drot (np.ndarray): 旋转速度 [drx, dry, drz] (始终为零)
            gripper_cmd (float): 夹爪控制指令 (-1.0 闭合, 1.0 张开, 0.0 保持)
            reset_cmd (bool): 是否触发重置
        """
        dpos = np.zeros(3)
        drot = np.zeros(3)
        gripper_cmd = 0.0
        reset_cmd = False
        
        if not self.joystick:
            return dpos, drot, gripper_cmd, reset_cmd

        # 处理事件队列，必须调用以更新手柄状态
        pygame.event.pump()
        
        # --- 轴映射 (基于标准 Xbox 手柄布局) ---
        # 0: 左摇杆 X (左右)
        # 1: 左摇杆 Y (上下)
        # 2: 左扳机 (LT)
        # 3: 右摇杆 X (左右)
        # 4: 右摇杆 Y (上下)
        # 5: 右扳机 (RT)
        
        # 1. 平移控制 (左摇杆 + 右摇杆Y轴)
        # 注意：摇杆 Y 轴通常向上为 -1，向下为 1，这里取反符合直觉
        dpos[0] = -self.apply_deadzone(self.joystick.get_axis(1)) * self.max_pos_speed # 前/后 (X)
        dpos[1] = -self.apply_deadzone(self.joystick.get_axis(0)) * self.max_pos_speed # 左/右 (Y)
        
        # Z轴：使用右摇杆Y轴，与 keyboard.py 行为一致
        # This matches the user's reference from `keyboard.py`
        if self.joystick.get_numaxes() > 4:
            dpos[2] = -self.apply_deadzone(self.joystick.get_axis(4)) * self.max_pos_speed

        # 3. 夹爪控制 (A键闭合, B键张开)
        # 按钮 0 (A), 1 (B)
        if self.joystick.get_button(0): 
            gripper_cmd = -1.0 # Close (A键)
        elif self.joystick.get_button(1): 
            gripper_cmd = 1.0 # Open (B键)
            
        # 4. 重置控制 (Start键/按钮7)
        if self.joystick.get_button(7):
            if not self.last_reset_state:
                reset_cmd = True
            self.last_reset_state = True
        else:
            self.last_reset_state = False
            
        return dpos, drot, gripper_cmd, reset_cmd

    def is_record_toggled(self):
        """
        检测录制按钮（Back键/按钮6）是否被按下（上升沿触发）。用于切换录制状态。
        """
        if not self.joystick:
            return False
            
        # 按钮 6 通常是 Xbox 手柄的 "Back" 键或 "View" 键
        # 你可以根据你的手柄型号调整这个 ID
        current_record_state = self.joystick.get_button(6)
        
        is_toggled = False
        if current_record_state and not self.last_record_state:
            is_toggled = True
            
        self.last_record_state = current_record_state
        return is_toggled

    def apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value