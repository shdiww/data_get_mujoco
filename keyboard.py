import glfw
import mujoco
import numpy as np
import mink

class InputListener:
    """处理键盘和鼠标输入的类"""

    def __init__(self, model, data, scene, cam, box_id, abs_pos_limits):
        self.model = model
        self.data = data
        self.scene = scene
        self.cam = cam
        self.box_id = box_id
        self.abs_pos_limits = abs_pos_limits

        self.gripper_target = 0.04
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        self.key_states = {}
        self.latched_keys = set()  # 用于记录当前帧内触发过的按键，防止短按丢失
        self.MOVE_SPEED = 0.2  # m/s
        self.GRIPPER_SPEED = 0.15  # m/s
        self.prev_gamepad_buttons = []

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS:
            self.key_states[key] = True
            self.latched_keys.add(key)
        elif act == glfw.RELEASE:
            self.key_states[key] = False

        # 重置逻辑保留在回调中，因为它是单次触发动作
        if act == glfw.PRESS and key == glfw.KEY_R:
            self.reset_simulation()

    def reset_simulation(self):
        """重置仿真环境"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        if self.box_id != -1:
            jnt_id = self.model.body_jntadr[self.box_id]
            if jnt_id != -1:
                q_addr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[q_addr:q_addr+7] = self.model.qpos0[q_addr:q_addr+7]
                self.data.qpos[q_addr] += np.random.uniform(-0.15, 0.15)
                self.data.qpos[q_addr+1] += np.random.uniform(-0.15, 0.15)
        mujoco.mj_forward(self.model, self.data)
        mink.move_mocap_to_frame(self.model, self.data, "target", "attachment_site", "site")
        self.gripper_target = 0.04

    def update(self, dt):
        """在主循环中调用，处理连续移动"""
        # 检查 Shift 键状态
        is_shift = self.key_states.get(glfw.KEY_LEFT_SHIFT, False) or \
                   self.key_states.get(glfw.KEY_RIGHT_SHIFT, False)
        multiplier = 5.0 if is_shift else 1.0

        move_step = self.MOVE_SPEED * multiplier * dt
        gripper_step = self.GRIPPER_SPEED * multiplier * dt

        # 移动控制 (支持同时按键，并归一化速度向量，优化点按体验)
        move_vec = np.zeros(3)
        
        def get_weight(key):
            if self.key_states.get(key, False):
                return 1.0
            elif key in self.latched_keys:
                return 0.4  # 点按(未长按)时给予较小的权重，避免突变
            return 0.0

        move_vec[0] += get_weight(glfw.KEY_W)
        move_vec[0] -= get_weight(glfw.KEY_S)
        move_vec[1] += get_weight(glfw.KEY_A)
        move_vec[1] -= get_weight(glfw.KEY_D)
        move_vec[2] += get_weight(glfw.KEY_Q)
        move_vec[2] -= get_weight(glfw.KEY_E)

        norm = np.linalg.norm(move_vec)
        if norm > 0:
            # 如果合力大于1 (例如同时按W和A)，归一化到1
            # 如果合力小于1 (例如只点按W)，保持原大小，从而实现微动
            scale = 1.0 / norm if norm > 1.0 else 1.0
            self.data.mocap_pos[0] += move_vec * scale * move_step

        # 夹爪控制
        self.gripper_target -= get_weight(glfw.KEY_Z) * gripper_step
        self.gripper_target += get_weight(glfw.KEY_X) * gripper_step

        # --- Xbox 手柄控制 ---
        if glfw.joystick_present(glfw.JOYSTICK_1):
            axes_state, axes_count = glfw.get_joystick_axes(glfw.JOYSTICK_1)
            axes = [axes_state[i] for i in range(axes_count)]
            buttons_state, buttons_count = glfw.get_joystick_buttons(glfw.JOYSTICK_1)
            buttons = [buttons_state[i] for i in range(buttons_count)]
            
            # 死区设置 (防止漂移)
            deadzone = 0.15
            
            # 左摇杆 (Axes 0/1): 控制 XY 平面移动
            # Axis 1 (Y) 通常是反向的 (上是 -1)
            if abs(axes[1]) > deadzone:
                self.data.mocap_pos[0, 0] -= axes[1] * self.MOVE_SPEED * dt
            if abs(axes[0]) > deadzone:
                self.data.mocap_pos[0, 1] -= axes[0] * self.MOVE_SPEED * dt

            # 右摇杆 Y轴 (Axis 4): 控制 Z 轴高度
            if abs(axes[4]) > deadzone:
                self.data.mocap_pos[0, 2] -= axes[4] * self.MOVE_SPEED * dt

            # 按钮 A (0): 闭合夹爪
            if buttons[0]:
                self.gripper_target -= self.GRIPPER_SPEED * dt
            # 按钮 B (1): 张开夹爪
            if buttons[1]:
                self.gripper_target += self.GRIPPER_SPEED * dt

            # 按钮 Start (7): 重置 (防止连续触发)
            if len(buttons) > 7 and buttons[7] and (len(self.prev_gamepad_buttons) <= 7 or not self.prev_gamepad_buttons[7]):
                self.reset_simulation()
            
            self.prev_gamepad_buttons = list(buttons)

        # 限制范围
        self.gripper_target = np.clip(self.gripper_target, 0.0, 0.04)
        np.clip(self.data.mocap_pos[0], self.abs_pos_limits[:, 0], self.abs_pos_limits[:, 1], out=self.data.mocap_pos[0])

        # 清除本帧的锁存按键
        self.latched_keys.clear()

    def mouse_button(self, window, button, act, mods):
        if button == glfw.MOUSE_BUTTON_LEFT: self.button_left = act == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_MIDDLE: self.button_middle = act == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT: self.button_right = act == glfw.PRESS
        self.lastx, self.lasty = glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        if not (self.button_left or self.button_right): return
        dx, dy = xpos - self.lastx, ypos - self.lasty
        self.lastx, self.lasty = xpos, ypos
        width, height = glfw.get_window_size(window)
        if self.button_left:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx / width, 0, self.scene, self.cam)
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0, dy / height, self.scene, self.cam)
        elif self.button_right:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_H, dx / width, 0, self.scene, self.cam)
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V, 0, dy / height, self.scene, self.cam)

    def scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, self.scene, self.cam)