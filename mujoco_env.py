import glfw
import mujoco
import numpy as np
import mink
from keyboard import InputListener
from task import PickAndPlaceTask
from main import converge_ik, _XML, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS

class MujocoEnv:
    def __init__(self, obs_resolution=(640, 480)):
        # 加载MuJoCo模型和数据
        self.model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        self.data = mujoco.MjData(self.model)
        
        self.task = PickAndPlaceTask(self.model)
        self.configuration = mink.Configuration(self.model)

        # 定义 IK 任务
        self.end_effector_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=self.model, cost=1e-2)
        self.tasks = [self.end_effector_task, self.posture_task]

        # 初始化 GLFW 窗口
        if not glfw.init():
            raise Exception("初始化GLFW失败")
        self.window = glfw.create_window(1800, 900, "Record Episode", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("创建GLFW窗口失败")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # 初始化仿真状态 (为了获取初始位置)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.task.randomize_box(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        mink.move_mocap_to_frame(self.model, self.data, "target", "attachment_site", "site")
        initial_pos = self.data.mocap_pos[0].copy()

        # 更新 IK 配置
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

        # 初始化渲染相关
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)

        # 初始化第二个相机 (用于右侧固定视角)
        self.cam2 = mujoco.MjvCamera()
        self.cam2.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam2.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")

        # 初始化第三个相机 (用于手眼视角)
        self.cam3 = mujoco.MjvCamera()
        self.cam3.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam3.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "hand_camera")

        # 设置输入监听器
        POS_LIMITS = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.6, 0.5]])
        abs_pos_limits = POS_LIMITS + initial_pos[:, np.newaxis]

        self.input_listener = InputListener(self.model, self.data, self.scene, self.cam, self.task.box_id, abs_pos_limits)

        glfw.set_cursor_pos_callback(self.window, self.input_listener.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.input_listener.mouse_button)
        glfw.set_scroll_callback(self.window, self.input_listener.scroll)
        
        # 初始化观测图像缓冲区
        self.obs_width, self.obs_height = obs_resolution
        self.offscreen_viewport = mujoco.MjrRect(0, 0, self.obs_width, self.obs_height)
        self.rgb_buffer = np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8)

        # 初始重置
        self.reset()

    def reset(self):
        """重置仿真环境到初始状态"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.task.randomize_box(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        mink.move_mocap_to_frame(self.model, self.data, "target", "attachment_site", "site")
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        if hasattr(self.input_listener, 'gripper_target'):
            self.input_listener.gripper_target = 0.04

    def check_completion(self):
        return self.task.check_completion(self.model, self.data)

    def get_images(self):
        """捕获所有相机的图像"""
        images = {}
        
        # 定义要捕获的相机列表及其名称
        cameras = [
            ('camera_0', self.cam),   # 主视角
            ('camera_1', self.cam2),  # 全局概览
            ('camera_2', self.cam3)   # 手眼相机
        ]

        for name, cam in cameras:
            # 更新场景并渲染到视口
            mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(self.offscreen_viewport, self.scene, self.context)
            # 读取像素
            mujoco.mjr_readPixels(self.rgb_buffer, None, self.offscreen_viewport, self.context)
            # MuJoCo 渲染是倒置的，需要垂直翻转；copy() 确保数据连续且独立
            images[name] = np.flipud(self.rgb_buffer).copy()
            
        return images

    def get_obs(self):
        # 返回包含状态和图像的字典，与 RealEnv 结构对齐
        obs = self.get_images()
        obs['state'] = self.data.qpos.copy()
        return obs

    def get_action_from_input(self, dt):
        # 获取并应用输入 (生成 Action)
        self.input_listener.update(dt)
        
        # action: [x, y, z, qw, qx, qy, qz, gripper]
        action = np.concatenate([
            self.data.mocap_pos[0],
            self.data.mocap_quat[0],
            [self.input_listener.gripper_target]
        ])
        return action

    def step(self, dt):
        # 3. 执行控制 (IK + Step)
        T_wt = mink.SE3.from_mocap_name(self.model, self.data, "target")
        self.end_effector_task.set_target(T_wt)
        converge_ik(self.configuration, self.tasks, dt, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS)

        gripper_target = self.input_listener.gripper_target
        self.configuration.q[7] = gripper_target
        if self.configuration.q.shape[0] > 8:
            self.configuration.q[8] = gripper_target

        if self.model.nu > 0:
            if self.task.gripper_act1 != -1: self.data.ctrl[self.task.gripper_act1] = gripper_target
            if self.task.gripper_act2 != -1: self.data.ctrl[self.task.gripper_act2] = gripper_target
            for i in range(self.model.nu):
                if i != self.task.gripper_act1 and i != self.task.gripper_act2:
                    self.data.ctrl[i] = self.configuration.q[i]
        
        mujoco.mj_step(self.model, self.data)

    def render(self, is_recording=False, episode_count=0, buffer_len=0):
        width, height = glfw.get_framebuffer_size(self.window)

        # 1. 左侧视图 (主相机)
        viewport1 = mujoco.MjrRect(0, 0, width // 3, height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport1, self.scene, self.context)
        
        # Overlay status
        status_text = "RECORDING" if is_recording else "IDLE"
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, 
            mujoco.mjtGridPos.mjGRID_TOPLEFT, 
            viewport1, 
            status_text, 
            f"Ep: {episode_count} | Buf: {buffer_len}",
            self.context
        )

        # 2. 中间视图 (全局概览)
        viewport2 = mujoco.MjrRect(width // 3, 0, width // 3, height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam2, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport2, self.scene, self.context)

        # 3. 右侧视图 (手眼相机)
        viewport3 = mujoco.MjrRect(2 * (width // 3), 0, width - 2 * (width // 3), height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam3, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport3, self.scene, self.context)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        glfw.terminate()