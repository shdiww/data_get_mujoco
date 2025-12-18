from pathlib import Path

import glfw
import mujoco
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

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
    window = glfw.create_window(1200, 900, "Franka Emika Panda", None, None)
    if not window:
        glfw.terminate()
        raise Exception("创建GLFW窗口失败")
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # 开启垂直同步

    # --- 仿真和mocap(运动捕捉)目标初始化 ---
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)
    mujoco.mj_forward(model, data)  # 正向运动学，确保数据一致
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
    initial_pos = data.mocap_pos[0].copy()

    # --- 为交互定义回调函数和状态变量 ---
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjv_defaultFreeCamera(model, cam)

    button_left, button_middle, button_right = False, False, False
    lastx, lasty = 0, 0
    KEY_STEP = 0.001  # 键盘控制的步长 (米)

    # 定义一个相对于初始位置的可达工作空间
    POS_LIMITS = np.array([[-0.4, 0.4], [-0.4, 0.4], [-0.3, 0.5]])  # x, y, z的相对范围
    abs_pos_limits = POS_LIMITS + initial_pos[:, np.newaxis]

    def keyboard(window, key, scancode, act, mods):
        """键盘回调函数: 控制mocap目标的位置"""
        if act == glfw.PRESS or act == glfw.REPEAT:
            if key == glfw.KEY_W: data.mocap_pos[0, 0] += KEY_STEP
            elif key == glfw.KEY_S: data.mocap_pos[0, 0] -= KEY_STEP
            elif key == glfw.KEY_A: data.mocap_pos[0, 1] += KEY_STEP
            elif key == glfw.KEY_D: data.mocap_pos[0, 1] -= KEY_STEP
            elif key == glfw.KEY_Q: data.mocap_pos[0, 2] += KEY_STEP
            elif key == glfw.KEY_E: data.mocap_pos[0, 2] -= KEY_STEP
            np.clip(
                data.mocap_pos[0],
                abs_pos_limits[:, 0],
                abs_pos_limits[:, 1],
                out=data.mocap_pos[0],
            )

    def mouse_button(window, button, act, mods):
        """鼠标按键回调函数: 记录按键状态"""
        nonlocal button_left, button_middle, button_right, lastx, lasty
        if button == glfw.MOUSE_BUTTON_LEFT: button_left = act == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_MIDDLE: button_middle = act == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT: button_right = act == glfw.PRESS
        lastx, lasty = glfw.get_cursor_pos(window)

    def mouse_move(window, xpos, ypos):
        """鼠标移动回调函数: 控制相机"""
        nonlocal lastx, lasty
        if not (button_left or button_right): return
        dx, dy = xpos - lastx, ypos - lasty
        lastx, lasty = xpos, ypos
        width, height = glfw.get_window_size(window)
        if button_left:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx / width, 0, scene, cam)
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0, dy / height, scene, cam)
        elif button_right:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_H, dx / width, 0, scene, cam)
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_V, 0, dy / height, scene, cam)

    def scroll(window, xoffset, yoffset):
        """鼠标滚轮回调函数: 缩放相机"""
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scene, cam)

    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, scroll)

    # --- 主循环 ---
    rate = RateLimiter(frequency=200.0, warn=False)
    print("\n---")
    print("控制说明:")
    print("  W/S/A/D/Q/E: 移动机械臂末端目标")
    print("  鼠标左键拖动: 旋转视角")
    print("  鼠标右键拖动: 平移视角")
    print("  鼠标滚轮: 缩放视角")
    print("目标点已被限制在安全工作区域内。")
    print("---\n")

    while not glfw.window_should_close(window):
        dt = rate.dt
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        end_effector_task.set_target(T_wt)
        converge_ik(configuration, tasks, dt, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS)
        data.ctrl = configuration.q[:8]
        mujoco.mj_step(model, data)
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        rate.sleep()

    glfw.terminate()


if __name__ == "__main__":
    main()