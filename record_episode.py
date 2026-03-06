import argparse
from typing import List, Dict

import glfw
import mujoco
import numpy as np
import zarr
from loop_rate_limiters import RateLimiter

import mink

from keyboard import InputListener
from task import PickAndPlaceTask
from main import converge_ik, _XML, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS

def save_dataset(episodes: List[Dict[str, np.ndarray]], filename="episode.zarr"):
    """保存多条轨迹为 ReplayBuffer 兼容的 Zarr 格式。"""
    if len(episodes) == 0:
        print("No completed episodes. Skip saving.")
        return

    states = np.concatenate([ep["state"] for ep in episodes], axis=0).astype(np.float32)
    actions = np.concatenate([ep["action"] for ep in episodes], axis=0).astype(np.float32)

    ends = []
    total = 0
    for ep in episodes:
        total += len(ep["state"])
        ends.append(total)
    episode_ends = np.array(ends, dtype=np.int64)

    print(f"Saving {len(episodes)} episodes / {len(states)} frames to {filename}...")
    root = zarr.open(filename, mode='w')
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)
    data_group.create_dataset('state', data=states, overwrite=True)
    data_group.create_dataset('action', data=actions, overwrite=True)
    meta_group.create_dataset('episode_ends', data=episode_ends, overwrite=True)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Record Mujoco episode to Zarr")
    parser.add_argument("--dataset_path", type=str, default="episode.zarr", help="Output path for the zarr dataset")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of successful episodes to collect")
    parser.add_argument("--min_episode_steps", type=int, default=80, help="Minimum steps for a valid episode")
    parser.add_argument("--max_episode_steps", type=int, default=3000, help="Drop current episode if too long")
    parser.add_argument("--save_partial", action="store_true", help="Save finished episodes when window closes early")
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
    start_qpos = data.qpos.copy()
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
    
    # 当前回合缓冲区 + 全部成功回合
    episode_states: List[np.ndarray] = []
    episode_actions: List[np.ndarray] = []
    completed_episodes: List[Dict[str, np.ndarray]] = []
    prev_time = data.time

    print("\n--- 开始录制 ---")
    print(f"目标成功回合数: {args.num_episodes}")
    print(f"最短回合长度: {args.min_episode_steps} steps")
    print("动作维度: 4 -> [x, y, z, gripper]")
    print("按 R 重置 (会丢弃当前回合，不影响已成功回合)")
    print("收集完成后自动保存并退出。")
    print("----------------\n")

    def reset_for_next_episode():
        task.reset(model, data, start_qpos, configuration, initial_pos, input_listener)
        posture_task.set_target_from_configuration(configuration)

    user_aborted = False

    while not glfw.window_should_close(window):
        dt = rate.dt

        # 检测是否发生了重置 (通过时间回跳判断)
        if data.time < prev_time:
            dropped_len = len(episode_states)
            episode_states = []
            episode_actions = []
            print(f"检测到重置，已丢弃当前回合 {dropped_len} steps。")
        prev_time = data.time

        # 1. 记录当前状态 (State) - 在应用动作之前
        # state: 完整的关节位置 (包含机械臂和方块)
        episode_states.append(data.qpos.copy())

        # 2. 获取并应用输入 (生成 Action)
        input_listener.update(dt)
        
        # action (4D): [x, y, z, gripper]
        action = np.concatenate([data.mocap_pos[0], [input_listener.gripper_target]])
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
            ep_len = len(episode_states)
            if ep_len >= args.min_episode_steps:
                completed_episodes.append({
                    "state": np.array(episode_states, dtype=np.float32),
                    "action": np.array(episode_actions, dtype=np.float32),
                })
                print(f"任务完成！已保存回合 {len(completed_episodes)}/{args.num_episodes} (steps={ep_len})")
            else:
                print(f"任务完成但回合过短 (steps={ep_len} < {args.min_episode_steps})，已丢弃。")

            episode_states = []
            episode_actions = []

            if len(completed_episodes) >= args.num_episodes:
                save_dataset(completed_episodes, filename=args.dataset_path)
                break
            reset_for_next_episode()

        if len(episode_states) > args.max_episode_steps:
            print(f"当前回合超过最大长度 {args.max_episode_steps}，已自动丢弃并重置。")
            episode_states = []
            episode_actions = []
            reset_for_next_episode()

        glfw.swap_buffers(window)
        glfw.poll_events()
        rate.sleep()

    if glfw.window_should_close(window):
        user_aborted = True

    if user_aborted and args.save_partial and len(completed_episodes) > 0:
        print("窗口关闭，按 --save_partial 保存已完成回合。")
        save_dataset(completed_episodes, filename=args.dataset_path)

    glfw.terminate()

if __name__ == "__main__":
    main()
