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
        self.MOVE_SPEED = 0.2  # m/s
        self.GRIPPER_SPEED = 0.15  # m/s

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS:
            self.key_states[key] = True
        elif act == glfw.RELEASE:
            self.key_states[key] = False

        # 重置逻辑保留在回调中，因为它是单次触发动作
        if act == glfw.PRESS and key == glfw.KEY_R:
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

        # 切换相机视角 (V键)
        if act == glfw.PRESS and key == glfw.KEY_V:
            if self.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
                if cam_id != -1:
                    self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.cam.fixedcamid = cam_id
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def update(self, dt):
        """在主循环中调用，处理连续移动"""
        # 检查 Shift 键状态
        is_shift = self.key_states.get(glfw.KEY_LEFT_SHIFT, False) or \
                   self.key_states.get(glfw.KEY_RIGHT_SHIFT, False)
        multiplier = 5.0 if is_shift else 1.0

        move_step = self.MOVE_SPEED * multiplier * dt
        gripper_step = self.GRIPPER_SPEED * multiplier * dt

        # 移动控制 (支持同时按键)
        if self.key_states.get(glfw.KEY_W, False):
            self.data.mocap_pos[0, 0] += move_step
        if self.key_states.get(glfw.KEY_S, False):
            self.data.mocap_pos[0, 0] -= move_step
        if self.key_states.get(glfw.KEY_A, False):
            self.data.mocap_pos[0, 1] += move_step
        if self.key_states.get(glfw.KEY_D, False):
            self.data.mocap_pos[0, 1] -= move_step
        if self.key_states.get(glfw.KEY_Q, False):
            self.data.mocap_pos[0, 2] += move_step
        if self.key_states.get(glfw.KEY_E, False):
            self.data.mocap_pos[0, 2] -= move_step

        # 夹爪控制
        if self.key_states.get(glfw.KEY_Z, False):
            self.gripper_target -= gripper_step
        if self.key_states.get(glfw.KEY_X, False):
            self.gripper_target += gripper_step

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