from pathlib import Path

import glfw
import mujoco
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

from keyboard import InputListener
from task import PickAndPlaceTask

_HERE = Path(__file__).parent
_XML = _HERE / "model/franka_emika_panda/mjx_scene.xml"

# IK 求解器参数
SOLVER = "daqp"  # 使用的二次规划求解器
POS_THRESHOLD = 1e-4  # 位置误差收敛阈值 (米)
ORI_THRESHOLD = 1e-4  # 方向误差收敛阈值 (弧度)
MAX_ITERS = 20  # 单步IK的最大迭代次数


def converge_ik(
    configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters
):
    """
    运行最多 'max_iters' 次IK计算。

    Args:
        configuration: 机器人构型。
        tasks: IK任务列表。
        dt: 时间步长。
        solver: QP求解器名称。
        pos_threshold: 位置误差收敛阈值。
        ori_threshold: 方向误差收敛阈值。
        max_iters: 最大迭代次数。

    Returns:
        如果位置和姿态误差都低于阈值，则返回 True，否则返回 False。
    """
    for _ in range(max_iters):
        # 调用mink求解IK，得到关节速度
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        # 将计算出的速度积分到机器人构型中
        configuration.integrate_inplace(vel, dt)

        # 这里只检查第一个任务（末端执行器任务）的误差。
        # 如果有多个任务，你可能需要组合它们的误差。
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        # 如果位置和姿态都达到要求，提前返回
        if pos_achieved and ori_achieved:
            return True
    return False


def main():
    # 加载MuJoCo模型和数据
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    
    # 初始化任务环境 (自动获取相关ID)
    task = PickAndPlaceTask(model)

    # 创建一个Mink机器人构型对象
    configuration = mink.Configuration(model)

    # 定义IK任务
    # 1. 末端执行器任务：控制 "attachment_site" 的位置和姿态
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",  # 目标帧的名称
        frame_type="site",  # 目标帧的类型
        position_cost=1.0,  # 位置误差的权重
        orientation_cost=1.0,  # 方向误差的权重
        lm_damping=1.0,  # Levenberg-Marquardt 阻尼系数
    )
    # 2. 姿态任务：让机器人保持一个较为自然的姿态，避免奇异构型
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    # 任务列表
    tasks = [end_effector_task, posture_task]

    # --- GLFW 和渲染设置 ---
    if not glfw.init():
        raise Exception("初始化GLFW失败")
    window = glfw.create_window(1800, 900, "Franka Emika Panda", None, None)
    if not window:
        glfw.terminate()
        raise Exception("创建GLFW窗口失败")
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # 开启垂直同步

    # --- 仿真和mocap(运动捕捉)目标初始化 ---
    # 直接重置到 "home" 关键帧，该关键帧现在是我们的低位起始姿态
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    start_qpos = data.qpos.copy()  # 保存初始姿态

    # 随机化方块位置
    task.randomize_box(model, data)

    # 更新正向运动学，以反映关节和方块位置的变化
    mujoco.mj_forward(model, data)

    # 将 mocap 目标 ("target") 移动到当前末端执行器的位置
    # 这样可以确保控制开始时，目标和机械臂末端是对齐的，避免跳动
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
    initial_pos = data.mocap_pos[0].copy()

    # 更新IK求解器的状态
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)

    # --- 为交互定义回调函数和状态变量 ---
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjv_defaultFreeCamera(model, cam)

    # 初始化第二个相机 (用于右侧固定视角)
    cam2 = mujoco.MjvCamera()
    cam2.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam2.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")

    # 初始化第三个相机 (用于手眼视角)
    cam3 = mujoco.MjvCamera()
    cam3.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam3.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "hand_camera")

    # 定义一个相对于初始位置的可达工作空间
    POS_LIMITS = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.6, 0.5]])  # x, y, z的相对范围
    abs_pos_limits = POS_LIMITS + initial_pos[:, np.newaxis]

    # 初始化输入监听器
    input_listener = InputListener(model, data, scene, cam, task.box_id, abs_pos_limits)

    glfw.set_key_callback(window, input_listener.keyboard)
    glfw.set_cursor_pos_callback(window, input_listener.mouse_move)
    glfw.set_mouse_button_callback(window, input_listener.mouse_button)
    glfw.set_scroll_callback(window, input_listener.scroll)

    # --- 主循环 ---
    rate = RateLimiter(frequency=200.0, warn=False)
    print("\n---")
    print("控制说明:")
    print("  W/S/A/D/Q/E: 移动机械臂末端目标")
    print("  鼠标左键拖动: 旋转视角")
    print("  鼠标右键拖动: 平移视角")
    print("  鼠标滚轮: 缩放视角")
    print("  Z / X: 闭合/张开 夹爪")
    print("  R: 重置仿真并随机化方块位置")
    print("  按住 Shift: 加快移动/开合速度")
    print("---\n")
    print("控制说明 (手柄):")
    print("  左摇杆: 前后左右移动 (XY轴)")
    print("  右摇杆 (上下): 上下移动 (Z轴)")
    print("  A 键: 闭合夹爪")
    print("  B 键: 张开夹爪")
    print("  Start 键: 重置仿真")
    print("任务: 将方块移动到地板上的红色圆形区域")
    print("目标点已被限制在安全工作区域内。")

    while not glfw.window_should_close(window):
        dt = rate.dt
        input_listener.update(dt)
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        end_effector_task.set_target(T_wt)
        converge_ik(configuration, tasks, dt, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS)

        # 夹爪控制: 覆盖IK计算出的夹爪关节角度
        gripper_target = input_listener.gripper_target
        configuration.q[7] = gripper_target
        if configuration.q.shape[0] > 8:
            configuration.q[8] = gripper_target

        # 将控制信号应用到致动器
        if model.nu > 0:
            # 1. 优先设置夹爪致动器 (如果存在)
            if task.gripper_act1 != -1: data.ctrl[task.gripper_act1] = gripper_target
            if task.gripper_act2 != -1: data.ctrl[task.gripper_act2] = gripper_target
            
            # 2. 设置手臂致动器 (假设非夹爪的致动器对应手臂关节)
            for i in range(model.nu):
                if i != task.gripper_act1 and i != task.gripper_act2:
                    data.ctrl[i] = configuration.q[i]
        else:
            # 如果没有任何致动器，回退到直接修改关节位置 (瞬移)
            if configuration.q.shape[0] > 8:
                data.qpos[7] = gripper_target
                data.qpos[8] = gripper_target

        mujoco.mj_step(model, data)
        
        # --- 分屏渲染 ---
        width, height = glfw.get_framebuffer_size(window)
        
        # 1. 左侧视图 (主相机)
        viewport1 = mujoco.MjrRect(0, 0, width // 3, height)
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport1, scene, context)

        # 2. 中间视图 (全局概览)
        viewport2 = mujoco.MjrRect(width // 3, 0, width // 3, height)
        mujoco.mjv_updateScene(model, data, opt, pert, cam2, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport2, scene, context)

        # 3. 右侧视图 (手眼相机)
        viewport3 = mujoco.MjrRect(2 * (width // 3), 0, width - 2 * (width // 3), height)
        mujoco.mjv_updateScene(model, data, opt, pert, cam3, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport3, scene, context)

        # 计算方块与目标区域的重合面积，如果完全重合则重置
        if task.check_completion(model, data):
            task.reset(model, data, start_qpos, configuration, initial_pos, input_listener)

        glfw.swap_buffers(window)
        glfw.poll_events()
        rate.sleep()

    glfw.terminate()


if __name__ == "__main__":
    main()