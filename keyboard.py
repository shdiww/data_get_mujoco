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
        self.KEY_STEP = 0.001
        self.GRIPPER_STEP = 0.002

    def keyboard(self, window, key, scancode, act, mods):
        step_mult = 5.0 if (mods & glfw.MOD_SHIFT) else 1.0

        if act == glfw.PRESS or act == glfw.REPEAT:
            if key == glfw.KEY_W: self.data.mocap_pos[0, 0] += self.KEY_STEP * step_mult
            elif key == glfw.KEY_S: self.data.mocap_pos[0, 0] -= self.KEY_STEP * step_mult
            elif key == glfw.KEY_A: self.data.mocap_pos[0, 1] += self.KEY_STEP * step_mult
            elif key == glfw.KEY_D: self.data.mocap_pos[0, 1] -= self.KEY_STEP * step_mult
            elif key == glfw.KEY_Q: self.data.mocap_pos[0, 2] += self.KEY_STEP * step_mult
            elif key == glfw.KEY_E: self.data.mocap_pos[0, 2] -= self.KEY_STEP * step_mult
            
            if key == glfw.KEY_Z: self.gripper_target -= self.GRIPPER_STEP * step_mult
            elif key == glfw.KEY_X: self.gripper_target += self.GRIPPER_STEP * step_mult
            
            elif key == glfw.KEY_R:
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