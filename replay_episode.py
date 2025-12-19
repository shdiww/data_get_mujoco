from pathlib import Path
import argparse

import glfw
import mujoco
import numpy as np
import zarr
from loop_rate_limiters import RateLimiter

import mink

from keyboard import InputListener
from task import PickAndPlaceTask
from main import converge_ik, _XML, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS

def main():
    parser = argparse.ArgumentParser(description="Replay Mujoco episode from Zarr")
    parser.add_argument("--dataset_path", type=str, default="episode.zarr", help="Input path for the zarr dataset")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not Path(dataset_path).exists():
        print(f"错误: 找不到数据集 {dataset_path}")
        return

    print(f"正在加载数据集 {dataset_path} ...")
    root = zarr.open(dataset_path, mode='r')
    states = root['state'][:]
    actions = root['action'][:]
    print(f"加载了 {len(states)} 帧数据。")

    # 加载MuJoCo模型
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    
    task = PickAndPlaceTask(model)
    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    if not glfw.init():
        raise Exception("初始化GLFW失败")
    window = glfw.create_window(1800, 900, "Replay Episode", None, None)
    if not window:
        glfw.terminate()
        raise Exception("创建GLFW窗口失败")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # --- 初始化状态 ---
    # 使用录制的第一帧状态初始化 (包含机器人和方块位置)
    data.qpos[:] = states[0]
    mujoco.mj_forward(model, data)
    
    # 同步IK配置
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)

    # 渲染设置
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjv_defaultFreeCamera(model, cam)

    # 仅用于相机控制的 Input Listener (不处理机器人控制)
    # 传入 dummy 参数即可，因为我们不调用 update
    input_listener = InputListener(model, data, scene, cam, -1, np.zeros((3, 2)))
    glfw.set_cursor_pos_callback(window, input_listener.mouse_move)
    glfw.set_mouse_button_callback(window, input_listener.mouse_button)
    glfw.set_scroll_callback(window, input_listener.scroll)

    rate = RateLimiter(frequency=200.0, warn=False)
    
    print("\n--- 开始回放 ---")
    
    for i in range(len(actions)):
        if glfw.window_should_close(window):
            break
        
        dt = rate.dt

        # 1. 读取 Action
        action = actions[i]
        target_pos = action[:3]
        target_quat = action[3:7]
        gripper_target = action[7]

        # 2. 设置目标 (Action Application)
        data.mocap_pos[0] = target_pos
        data.mocap_quat[0] = target_quat

        # 3. 执行控制 (IK + Step)
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        end_effector_task.set_target(T_wt)
        converge_ik(configuration, tasks, dt, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS)

        # 应用夹爪
        configuration.q[7] = gripper_target
        if configuration.q.shape[0] > 8:
            configuration.q[8] = gripper_target

        if model.nu > 0:
            if task.gripper_act1 != -1: data.ctrl[task.gripper_act1] = gripper_target
            if task.gripper_act2 != -1: data.ctrl[task.gripper_act2] = gripper_target
            for j in range(model.nu):
                if j != task.gripper_act1 and j != task.gripper_act2:
                    data.ctrl[j] = configuration.q[j]

        mujoco.mj_step(model, data)

        # 渲染
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()
        rate.sleep()

    print("回放结束。")
    glfw.terminate()

if __name__ == "__main__":
    main()