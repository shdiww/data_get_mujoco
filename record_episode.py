import argparse

import glfw
import click
import numpy as np
from loop_rate_limiters import RateLimiter
from mujoco_env import MujocoEnv
from diffusion_policy.common.replay_buffer import ReplayBuffer

def main():
    parser = argparse.ArgumentParser(description="Record Mujoco episode to Zarr")
    parser.add_argument("--dataset_path", type=str, default="episode.zarr", help="Output path for the zarr dataset")
    args = parser.parse_args()

    # 初始化环境
    env = MujocoEnv()

    # 初始化数据集
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=args.dataset_path, mode='a')

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
    episode_count = replay_buffer.n_episodes

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
                        replay_buffer.drop_episode()
                        print("Dropped last episode.")
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
                
                # 将 episode_states (list of dicts) 转换为 dict of arrays
                # 例如: [{'state': s1, 'cam': c1}, {'state': s2, 'cam': c2}] -> {'state': [s1, s2], 'cam': [c1, c2]}
                keys = episode_states[0].keys()
                data = {
                    k: np.stack([x[k] for x in episode_states]) for k in keys
                }
                data['action'] = np.array(episode_actions)
                
                # 不传入 chunks 参数，让 ReplayBuffer 自动计算最优切片大小 (默认约 2MB/块)
                replay_buffer.add_episode(data, compressors='disk')
                print(f"Saved. Total frames in dataset: {replay_buffer.n_steps}")
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