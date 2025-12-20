from pathlib import Path
import argparse

import glfw
import click
import numpy as np
import zarr
from loop_rate_limiters import RateLimiter
from mujoco_env import MujocoEnv

def save_episode_to_zarr(dataset_path, states, actions, chunk_size=2000):
    """Appends a new episode to the Zarr dataset in ReplayBuffer format."""
    # 将新的 episode 追加到 Zarr 数据集中，采用 ReplayBuffer 格式
    print(f"Saving episode with {len(states)} frames to {dataset_path}...")
    mode = 'a' if Path(dataset_path).exists() else 'w'
    root = zarr.open(dataset_path, mode=mode)
    
    if 'data' not in root:
        root.create_group('data')
    if 'meta' not in root:
        root.create_group('meta')
        
    data_group = root['data']
    meta_group = root['meta']
    
    states_arr = np.array(states)
    actions_arr = np.array(actions)
    
    # Append to action
    # 追加 action 数据
    if 'action' not in data_group:
        # 如果是第一次创建，设置 chunk 大小
        data_group.create_dataset('action', data=actions_arr, chunks=(chunk_size, actions_arr.shape[1]))
    else:
        # 否则直接追加
        data_group['action'].append(actions_arr)
        
    # Append to state
    # 追加 state 数据
    if 'state' not in data_group:
        data_group.create_dataset('state', data=states_arr, chunks=(chunk_size, states_arr.shape[1]))
    else:
        data_group['state'].append(states_arr)
        
    # Update episode_ends
    # 更新 episode_ends 元数据，记录当前 episode 结束的索引位置
    end_idx = data_group['action'].shape[0]
    if 'episode_ends' not in meta_group:
        meta_group.create_dataset('episode_ends', data=np.array([end_idx]), chunks=(100,))
    else:
        meta_group['episode_ends'].append(np.array([end_idx]))
        
    print(f"Saved. Total frames in dataset: {end_idx}")
    
    # 重新读取验证，确保用户能看到确切的 Episode 数量变化
    root_check = zarr.open(dataset_path, mode='r')
    print(f"✅ 验证成功: 数据集当前共包含 {len(root_check['meta']['episode_ends'])} 条 Episode。")

def drop_last_episode_from_zarr(dataset_path):
    """Removes the last recorded episode from the Zarr dataset."""
    # 从 Zarr 数据集中删除最后录制的 episode
    if not Path(dataset_path).exists():
        print("Dataset does not exist.")
        return
    
    root = zarr.open(dataset_path, mode='a')
    if 'meta' not in root or 'episode_ends' not in root['meta'] or len(root['meta']['episode_ends']) == 0:
        print("No episodes to drop.")
        return
        
    episode_ends = root['meta']['episode_ends']
    end_curr = episode_ends[-1]
    # 获取倒数第二个 episode 的结束位置，如果只有一个则为 0
    end_prev = episode_ends[-2] if len(episode_ends) > 1 else 0
    
    # Resize data arrays
    # 调整数据数组的大小，截断到上一个 episode 的结束位置
    root['data']['action'].resize(end_prev, root['data']['action'].shape[1])
    root['data']['state'].resize(end_prev, root['data']['state'].shape[1])
    
    # Resize meta
    # 调整元数据数组大小
    root['meta']['episode_ends'].resize(len(episode_ends) - 1)
    print(f"Dropped last episode. New total frames: {end_prev}")

def main():
    parser = argparse.ArgumentParser(description="Record Mujoco episode to Zarr")
    parser.add_argument("--dataset_path", type=str, default="episode.zarr", help="Output path for the zarr dataset")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Zarr chunk size (frames)")
    args = parser.parse_args()

    # 初始化环境
    env = MujocoEnv()

    rate = RateLimiter(frequency=200.0, warn=False)
    
    # State variables
    # 状态变量
    is_recording = False
    manual_save_trigger = False
    episode_states = []
    episode_actions = []
    episode_count = 0
    
    # Check existing episodes
    # 检查已存在的 episode 数量
    if Path(args.dataset_path).exists():
        try:
            root = zarr.open(args.dataset_path, mode='r')
            if 'meta' in root and 'episode_ends' in root['meta']:
                episode_count = len(root['meta']['episode_ends'])
        except Exception:
            pass
    print(f"Found {episode_count} existing episodes in {args.dataset_path}")
    print(f"New episodes will be appended. Delete the file to start fresh.")

    print("\n--- 开始录制 ---")
    print(" [C] 开始录制 (Start Recording)")
    print(" [Backspace] 删除上一条 (Drop Episode)")
    print(" [T] 手动触发保存并重置 (Trigger Save & Reset)")
    print(" [Q] 退出 (Quit)")
    print("----------------\n")

    # Custom key callback to handle recording logic
    # 自定义按键回调处理录制逻辑
    def on_key(window, key, scancode, action, mods):
        nonlocal is_recording, episode_states, episode_actions, episode_count, manual_save_trigger
        
        # Pass to original listener for robot control
        # 传递给原始监听器以控制机器人
        env.input_listener.keyboard(window, key, scancode, action, mods)
        
        if action == glfw.PRESS:
            if key == glfw.KEY_C:
                # 按 C 开始录制
                if not is_recording:
                    is_recording = True
                    episode_states = []
                    episode_actions = []
                    print("Recording started!")
            elif key == glfw.KEY_T:
                # 按 T 手动触发保存
                if is_recording:
                    manual_save_trigger = True
                    print("Manual save triggered!")
            elif key == glfw.KEY_BACKSPACE:
                # 按 Backspace 删除上一条或取消当前录制
                if is_recording:
                    is_recording = False
                    episode_states = []
                    episode_actions = []
                    print("Recording cancelled.")
                else:
                    if click.confirm('Are you sure to drop the last episode?'):
                        drop_last_episode_from_zarr(args.dataset_path)
                        episode_count = max(0, episode_count - 1)
            elif key == glfw.KEY_Q:
                # 按 Q 退出
                glfw.set_window_should_close(window, True)

    glfw.set_key_callback(env.window, on_key)

    while not glfw.window_should_close(env.window):
        dt = rate.dt

        # 自动检测任务完成并重置
        if is_recording and (env.check_completion() or manual_save_trigger):
            # 增加最小帧数限制 (例如 100 帧)，防止重置后因状态未完全清除导致的连击
            if len(episode_states) > 100:
                print(f"Task completed! Saving episode {episode_count}...")
                save_episode_to_zarr(args.dataset_path, episode_states, episode_actions, chunk_size=args.chunk_size)
                episode_count += 1
            else:
                print(f"Discarding short episode ({len(episode_states)} frames) - likely reset artifact.")
            
            episode_states = []
            episode_actions = []
            manual_save_trigger = False
            env.reset()

        # 1. 记录当前状态 (State) - 在应用动作之前
        if is_recording:
            episode_states.append(env.get_obs())

        # 2. 获取并应用输入 (生成 Action)
        action = env.get_action_from_input(dt)
        
        if is_recording:
            episode_actions.append(action)

        # 3. 执行控制 (IK + Step)
        env.step(dt)
        
        # 4. 渲染
        env.render(is_recording, episode_count, len(episode_states))
        
        rate.sleep()

    env.close()

if __name__ == "__main__":
    main()