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
        self.MOVE_SPEED = 0.08  # m/s
        self.GRIPPER_SPEED = 0.10  # m/s
        self.GAMEPAD_DEADZONE = 0.28
        self.GAMEPAD_SMOOTH_ALPHA = 0.35
        self.GAMEPAD_STOP_EPS = 0.02
        self.prev_gamepad_buttons = []
        self.active_gamepad_id = None
        self.filtered_move = np.zeros(3)

    def _find_active_gamepad(self):
        """选择一个最可能的手柄设备ID。"""
        candidates = []
        joystick_ids = [
            glfw.JOYSTICK_1, glfw.JOYSTICK_2, glfw.JOYSTICK_3, glfw.JOYSTICK_4,
            glfw.JOYSTICK_5, glfw.JOYSTICK_6, glfw.JOYSTICK_7, glfw.JOYSTICK_8,
            glfw.JOYSTICK_9, glfw.JOYSTICK_10, glfw.JOYSTICK_11, glfw.JOYSTICK_12,
            glfw.JOYSTICK_13, glfw.JOYSTICK_14, glfw.JOYSTICK_15, glfw.JOYSTICK_16,
        ]

        for jid in joystick_ids:
            if not glfw.joystick_present(jid):
                continue

            name = glfw.get_joystick_name(jid) or ""
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            name_lower = name.lower()

            axes_state, axes_count = glfw.get_joystick_axes(jid)
            buttons_state, buttons_count = glfw.get_joystick_buttons(jid)

            if axes_count < 4 or buttons_count < 2:
                continue

            score = 0
            if glfw.joystick_is_gamepad(jid):
                score += 3
            if any(k in name_lower for k in ("xbox", "xinput", "controller", "gamepad")):
                score += 2
            score += min(axes_count, 8) * 0.1 + min(buttons_count, 16) * 0.05

            candidates.append((score, jid))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]

    def _reset_simulation(self):
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
        """在主循环中调用，仅处理手柄连续移动"""

        def apply_deadzone_and_curve(value):
            magnitude = abs(value)
            if magnitude <= self.GAMEPAD_DEADZONE:
                return 0.0
            norm = (magnitude - self.GAMEPAD_DEADZONE) / (1.0 - self.GAMEPAD_DEADZONE)
            curved = norm * norm
            return np.sign(value) * curved

        # --- Xbox 手柄控制 ---
        if self.active_gamepad_id is None or not glfw.joystick_present(self.active_gamepad_id):
            self.active_gamepad_id = self._find_active_gamepad()
        if self.active_gamepad_id is not None:
            jid = self.active_gamepad_id
            axes_state, axes_count = glfw.get_joystick_axes(jid)
            axes = [axes_state[i] for i in range(axes_count)]
            buttons_state, buttons_count = glfw.get_joystick_buttons(jid)
            buttons = [buttons_state[i] for i in range(buttons_count)]

            # 过滤非标准手柄映射，避免漂移
            if len(axes) < 5 or len(buttons) < 2:
                self.prev_gamepad_buttons = []
                axes = []
                buttons = []
                self.filtered_move[:] = 0.0

            def axis_value(index):
                return axes[index] if index < len(axes) else 0.0

            def button_pressed(index):
                return index < len(buttons) and bool(buttons[index])
            
            # 左摇杆 (Axes 0/1): 控制 XY 平面移动
            # Axis 1 (Y) 通常是反向的 (上是 -1)
            axis_1 = axis_value(1)
            axis_0 = axis_value(0)

            # 右摇杆 Y轴: 控制 Z 轴高度
            # 一些驱动映射在 Axis 4，另一些映射在 Axis 3
            axis_4 = axis_value(4)
            axis_3 = axis_value(3)
            z_axis = axis_4 if abs(axis_4) > abs(axis_3) else axis_3

            raw_move = np.array(
                [
                    apply_deadzone_and_curve(-axis_1),
                    apply_deadzone_and_curve(-axis_0),
                    apply_deadzone_and_curve(-z_axis),
                ],
                dtype=float,
            )

            if np.max(np.abs(raw_move)) < self.GAMEPAD_STOP_EPS:
                self.filtered_move[:] = 0.0
            else:
                self.filtered_move = (1.0 - self.GAMEPAD_SMOOTH_ALPHA) * self.filtered_move + self.GAMEPAD_SMOOTH_ALPHA * raw_move

            move_norm = np.linalg.norm(self.filtered_move)
            if move_norm > 1.0:
                move_cmd = self.filtered_move / move_norm
            else:
                move_cmd = self.filtered_move

            self.data.mocap_pos[0] += move_cmd * self.MOVE_SPEED * dt

            # 按钮 A (0): 闭合夹爪
            if button_pressed(0):
                self.gripper_target -= self.GRIPPER_SPEED * dt
            # 按钮 B (1): 张开夹爪
            if button_pressed(1):
                self.gripper_target += self.GRIPPER_SPEED * dt

            # 按钮 Start (7): 重置 (防止连续触发)
            prev_start = len(self.prev_gamepad_buttons) > 7 and bool(self.prev_gamepad_buttons[7])
            if button_pressed(7) and not prev_start:
                self._reset_simulation()
            
            self.prev_gamepad_buttons = list(buttons)
        else:
            self.prev_gamepad_buttons = []
            self.filtered_move[:] = 0.0

        # 限制范围
        self.gripper_target = np.clip(self.gripper_target, 0.0, 0.04)
        np.clip(self.data.mocap_pos[0], self.abs_pos_limits[:, 0], self.abs_pos_limits[:, 1], out=self.data.mocap_pos[0])

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
