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

def save_dataset(states, actions, filename="episode.zarr", episode_idx=0):
    """保存数据到 Zarr 文件 (Group format)"""
    print(f"Saving episode {episode_idx} with {len(states)} frames to {filename}...")
    mode = 'a' if Path(filename).exists() else 'w'
    root = zarr.open(filename, mode=mode)
    
    for key in ['action', 'state']:
        if key not in root:
            root.create_group(key)
            
    key_name = f"chunk_{episode_idx}"
    root['action'].create_dataset(key_name, data=np.array(actions), chunks=False, overwrite=True)
    root['state'].create_dataset(key_name, data=np.array(states), chunks=False, overwrite=True)
    print(f"Saved episode {episode_idx} to action/{key_name} and state/{key_name}.")

def main():
    parser = argparse.ArgumentParser(description="Record Mujoco episode to Zarr")
    parser.add_argument("--dataset_path", type=str, default="episode.zarr", help="Output path for the zarr dataset")
    args = parser.parse_args()

    # 加载MuJoCo模型和数据
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
    window = glfw.create_window(1800, 900, "Record Episode", None, None)
    if not window:
        glfw.terminate()
        raise Exception("创建GLFW窗口失败")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    task.randomize_box(model, data)
    mujoco.mj_forward(model, data)

    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
    initial_pos = data.mocap_pos[0].copy()

    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)

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

    POS_LIMITS = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.6, 0.5]])
    abs_pos_limits = POS_LIMITS + initial_pos[:, np.newaxis]

    input_listener = InputListener(model, data, scene, cam, task.box_id, abs_pos_limits)

    glfw.set_key_callback(window, input_listener.keyboard)
    glfw.set_cursor_pos_callback(window, input_listener.mouse_move)
    glfw.set_mouse_button_callback(window, input_listener.mouse_button)
    glfw.set_scroll_callback(window, input_listener.scroll)

    rate = RateLimiter(frequency=200.0, warn=False)
    
    # 确定起始 episode_idx
    episode_idx = 0
    if Path(args.dataset_path).exists():
        try:
            root = zarr.open(args.dataset_path, mode='r')
            if 'action' in root:
                indices = []
                for k in root['action'].keys():
                    if k.startswith('chunk_'):
                        try:
                            indices.append(int(k.split('_')[1]))
                        except ValueError:
                            continue
                if indices:
                    episode_idx = max(indices) + 1
        except Exception:
            pass
    print(f"Next episode index: {episode_idx}")

    # 数据缓冲区
    episode_states = []
    episode_actions = []
    prev_time = data.time

    print("\n--- 开始录制 ---")
    print("操作完成后会自动保存并退出。")
    print("按 R 重置 (会清空当前录制缓冲区)")
    print("----------------\n")

    while not glfw.window_should_close(window):
        dt = rate.dt

        # 检测是否发生了重置 (通过时间回跳判断)
        if data.time < prev_time:
            episode_states = []
            episode_actions = []
            print("检测到重置，缓冲区已清空。")
        prev_time = data.time

        # 1. 记录当前状态 (State) - 在应用动作之前
        # state: 完整的关节位置 (包含机械臂和方块)
        episode_states.append(data.qpos.copy())

        # 2. 获取并应用输入 (生成 Action)
        input_listener.update(dt)
        
        # action: 手柄/键盘信号处理后的末端目标状态 (位置 + 姿态 + 夹爪)
        # [x, y, z, qw, qx, qy, qz, gripper]
        action = np.concatenate([
            data.mocap_pos[0],
            data.mocap_quat[0],
            [input_listener.gripper_target]
        ])
        episode_actions.append(action)

        # 3. 执行控制 (IK + Step)
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        end_effector_task.set_target(T_wt)
        converge_ik(configuration, tasks, dt, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS)

        gripper_target = input_listener.gripper_target
        configuration.q[7] = gripper_target
        if configuration.q.shape[0] > 8:
            configuration.q[8] = gripper_target

        if model.nu > 0:
            if task.gripper_act1 != -1: data.ctrl[task.gripper_act1] = gripper_target
            if task.gripper_act2 != -1: data.ctrl[task.gripper_act2] = gripper_target
            for i in range(model.nu):
                if i != task.gripper_act1 and i != task.gripper_act2:
                    data.ctrl[i] = configuration.q[i]
        
        mujoco.mj_step(model, data)

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

        if task.check_completion(model, data):
            print(f"任务完成！保存第 {episode_idx} 条轨迹...")
            save_dataset(episode_states, episode_actions, args.dataset_path, episode_idx)
            episode_idx += 1
            input_listener.reset_simulation()

        glfw.swap_buffers(window)
        glfw.poll_events()
        rate.sleep()

    glfw.terminate()

if __name__ == "__main__":
    main()