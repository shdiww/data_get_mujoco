import time
import numpy as np
import mujoco
import glfw
import mink
from scipy.spatial.transform import Rotation as R
from .task import PickAndPlaceTask

class MujocoEnv:
    """
    仿照 RealEnv 结构的 MuJoCo 仿真环境类。
    负责：
    1. 加载 XML 模型
    2. 渲染图像 (GLFW 窗口 + 离屏渲染)
    3. 执行动作 (替代机器人控制器)
    4. 获取状态 (替代机器人反馈)
    """
    def __init__(self, xml_path, camera_names=None, frequency=30, obs_resolution=(640, 480)):
        """
        Args:
            xml_path (str): MuJoCo xml 文件路径
            camera_names (list): 需要渲染的相机名称列表，例如 ['top_cam', 'wrist_cam']
            frequency (int): 控制频率 (Hz)
        """
        if camera_names is None:
            camera_names = []
        
        self.xml_path = xml_path
        self.camera_names = camera_names
        self.frequency = frequency
        self.dt = 1.0 / frequency

        # 1. 加载 MuJoCo 模型和数据
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except ValueError:
            print(f"错误: 无法从 {xml_path} 加载 XML 文件")
            raise

        # 2. 初始化 GLFW 窗口 (用于显示，复刻 record_episode 的界面)
        if not glfw.init():
            raise Exception("初始化 GLFW 失败")
        self.window = glfw.create_window(1280, 720, "Mujoco Teleop", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("创建 GLFW 窗口失败")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1) # 开启垂直同步

        # 3. 初始化 Mink 配置和任务
        self.configuration = mink.Configuration(self.model)
        
        # 定义末端执行器任务 (追踪 "target" mocap body)
        self.end_effector_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        # 定义姿态任务 (保持自然姿态，避免奇异点)
        self.posture_task = mink.PostureTask(model=self.model, cost=1e-2)
        self.tasks = [self.end_effector_task, self.posture_task]
        
        # 初始化业务任务 (PickAndPlace)
        self.task = PickAndPlaceTask(self.model)

        # 4. 初始化渲染上下文
        self.cam = mujoco.MjvCamera() # 主视角 (Free camera)
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)

        # 初始化固定相机 (用于界面显示)
        self.cam_overview = mujoco.MjvCamera()
        self.cam_overview.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam_overview.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        
        self.cam_hand = mujoco.MjvCamera()
        self.cam_hand.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam_hand.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "hand_camera")

        # 初始化离屏渲染 (用于获取观测数据)
        self.obs_width, self.obs_height = obs_resolution
        self.offscreen_viewport = mujoco.MjrRect(0, 0, self.obs_width, self.obs_height)
        self.rgb_buffer = np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8)

        # 鼠标交互回调 (允许旋转主视角)
        self.button_left = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        glfw.set_cursor_pos_callback(self.window, self._mouse_move)
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_scroll_callback(self.window, self._scroll)

        # 5. 动作空间维度
        self.action_dim = self.model.nu
        
        print(f"[Mujoco环境] 已通过XML初始化: {xml_path}")
        print(f"[Mujoco环境] 动作维度: {self.action_dim}, 相机: {camera_names}")

    def reset(self):
        """重置环境到初始状态"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.task.randomize_box(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # 强制将夹爪设为张开状态 (0.04)
        for act_id in self.task.gripper_act_ids:
            if act_id != -1 and act_id < self.model.nu:
                self.data.ctrl[act_id] = 0.04
                joint_id = self.model.actuator_trnid[act_id, 0]
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_addr] = 0.04
        
        # 使用 mink 将 mocap 移动到当前末端位置和姿态
        # 这会保留 XML 中定义的初始姿态 (Home Keyframe)，且初始 IK 误差为 0，不会产生旋转
        mink.move_mocap_to_frame(self.model, self.data, "target", "attachment_site", "site")
        
        # 同步 Mink 配置
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

        return self.get_observation()

    def step(self, action):
        """
        执行一步动作
        Args:
            action (float): 夹爪目标位置 (0.0=闭合, 0.04=张开)。
                            注意：机械臂的移动由 self.data.mocap_pos 控制，不通过 action 传递。
        """
        gripper_target = action

        # --- Mink IK 求解循环 ---
        # 获取 Mocap "target" 的位姿作为 IK 目标
        T_wt = mink.SE3.from_mocap_name(self.model, self.data, "target")
        self.end_effector_task.set_target(T_wt)
        
        # 在一个控制周期内多次迭代 IK 以逼近目标
        for _ in range(10): # 迭代次数可调
            vel = mink.solve_ik(
                self.configuration, 
                self.tasks, 
                self.dt, 
                solver="daqp", 
                damping=1e-3
            )
            self.configuration.integrate_inplace(vel, self.dt)
            
        # --- 应用控制量 ---
        # 将计算出的关节位置应用到 ctrl (假设前7个是机械臂关节)
        for i in range(7):
            self.data.ctrl[i] = self.configuration.q[i]
            
        # 应用夹爪控制
        for act_id in self.task.gripper_act_ids:
            if act_id != -1 and act_id < self.model.nu:
                self.data.ctrl[act_id] = gripper_target

        # --- 物理步进 ---
        # 为了模拟真实时间的控制频率，我们可能需要在一个控制周期内多次 step 物理引擎
        # 假设 MuJoCo 的 timestep 是 0.002s，控制频率是 30Hz (0.033s)，则需要 step 约 16 次
        n_substeps = int(self.dt / self.model.opt.timestep)
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        # 获取观测
        obs = self.get_observation()
        
        return obs

    def get_observation(self):
        """
        获取当前观测数据，结构与 RealEnv 保持一致
        Returns:
            dict: 包含 'qpos', 'qvel', 'images'
        """
        # 获取本体感知信息 (Proprioception)
        qpos = np.array(self.data.qpos)
        qvel = np.array(self.data.qvel)

        # 获取图像信息 (Visual)
        images = {}
        # 使用离屏渲染获取数据
        cameras = [
            ('overview', self.cam_overview),
            ('hand_camera', self.cam_hand)
        ]
        for name, cam in cameras:
            if name in self.camera_names:
                mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                mujoco.mjr_render(self.offscreen_viewport, self.scene, self.context)
                mujoco.mjr_readPixels(self.rgb_buffer, None, self.offscreen_viewport, self.context)
                images[name] = np.flipud(self.rgb_buffer).copy()

        # 获取末端执行器位姿 (7维: 3 pos + 4 quat)
        eef_pos = self.data.site_xpos[self.task.attachment_site_id].copy()
        eef_mat = self.data.site_xmat[self.task.attachment_site_id].reshape(3, 3)
        eef_quat = R.from_matrix(eef_mat).as_quat()[[3, 0, 1, 2]] # xyzw -> wxyz (MuJoCo convention)
        state = np.concatenate([eef_pos, eef_quat])

        return {
            'state': state,
            'qpos': qpos,
            'qvel': qvel,
            'images': images,
            'timestep': time.time(),
            'robot_eef_pose': state
        }

    def render(self, fps=0.0, gripper_val=0.0, is_recording=False):
        """
        渲染到 GLFW 窗口 (复刻 record_episode 的三视图)
        """
        if glfw.window_should_close(self.window):
            return
            
        width, height = glfw.get_framebuffer_size(self.window)
        
        # 1. 左侧视图 (主相机 - 可交互)
        viewport1 = mujoco.MjrRect(0, 0, width // 3, height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport1, self.scene, self.context)
        
        # 显示状态信息 (Overlay)
        status_text = "RECORDING" if is_recording else "IDLE"
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport1,
            status_text,
            f"FPS: {fps:.1f} | Gripper: {gripper_val:.3f}",
            self.context
        )
        
        # 2. 中间视图 (全局概览)
        viewport2 = mujoco.MjrRect(width // 3, 0, width // 3, height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam_overview, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport2, self.scene, self.context)

        # 3. 右侧视图 (手眼相机)
        viewport3 = mujoco.MjrRect(2 * (width // 3), 0, width - 2 * (width // 3), height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam_hand, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport3, self.scene, self.context)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        glfw.terminate()

    # --- 鼠标回调函数 (用于控制主视角) ---
    def _mouse_button(self, window, button, act, mods):
        if button == glfw.MOUSE_BUTTON_LEFT: self.button_left = act == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT: self.button_right = act == glfw.PRESS
        self.lastx, self.lasty = glfw.get_cursor_pos(window)

    def _mouse_move(self, window, xpos, ypos):
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

    def _scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, self.scene, self.cam)
