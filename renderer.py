import glfw
import mujoco


class MujocoRenderer:
    """封装 GLFW 初始化、MuJoCo 渲染与窗口刷新。"""

    def __init__(self, model, width=1800, height=900, title="Franka Emika Panda"):
        self.model = model
        self.width = width
        self.height = height
        self.title = title

        self.window = None
        self.cam = None
        self.cam2 = None
        self.cam3 = None
        self.opt = None
        self.pert = None
        self.scene = None
        self.context = None

    def init(self):
        if not glfw.init():
            raise Exception("初始化GLFW失败")

        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("创建GLFW窗口失败")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)

        self.cam2 = mujoco.MjvCamera()
        self.cam2.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam2.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")

        self.cam3 = mujoco.MjvCamera()
        self.cam3.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam3.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "hand_camera")

    def register_mouse_callbacks(self, input_listener):
        glfw.set_cursor_pos_callback(self.window, input_listener.mouse_move)
        glfw.set_mouse_button_callback(self.window, input_listener.mouse_button)
        glfw.set_scroll_callback(self.window, input_listener.scroll)

    def should_close(self):
        return glfw.window_should_close(self.window)

    def render(self, data):
        width, height = glfw.get_framebuffer_size(self.window)

        viewport1 = mujoco.MjrRect(0, 0, width // 3, height)
        mujoco.mjv_updateScene(self.model, data, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport1, self.scene, self.context)

        viewport2 = mujoco.MjrRect(width // 3, 0, width // 3, height)
        mujoco.mjv_updateScene(self.model, data, self.opt, self.pert, self.cam2, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport2, self.scene, self.context)

        viewport3 = mujoco.MjrRect(2 * (width // 3), 0, width - 2 * (width // 3), height)
        mujoco.mjv_updateScene(self.model, data, self.opt, self.pert, self.cam3, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport3, self.scene, self.context)

    def refresh(self):
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        glfw.terminate()
