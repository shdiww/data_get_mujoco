import click
import pathlib
import numpy as np
import time
import mujoco
from diffusion_policy.common.replay_buffer import ReplayBuffer
from mujoco_diffusion.mujoco_env import MujocoEnv
from scipy.spatial.transform import Rotation as R

@click.command()
@click.option('-p', '--path', default='data/mujoco_demo/replay_buffer.zarr', help='Path to zarr dataset')
@click.option('-f', '--frequency', default=10, type=int, help='Control frequency (Hz)')
@click.option('-e', '--episode_idx', default=0, type=int, help='Episode index to replay')
def main(path, frequency, episode_idx):
    # -------------------------------------------------------------------------
    # 1. 加载数据集
    # -------------------------------------------------------------------------
    print(f"正在加载数据集: {path} ...")
    try:
        replay_buffer = ReplayBuffer.create_from_path(path, mode='r')
    except Exception as e:
        print(f"无法加载数据集: {e}")
        return

    print(f"总回合数: {replay_buffer.n_episodes}")
    if episode_idx >= replay_buffer.n_episodes:
        print(f"错误: 回合索引 {episode_idx} 超出范围 (0-{replay_buffer.n_episodes-1})")
        return

    # 获取指定回合的数据
    episode_data = replay_buffer.get_episode(episode_idx)
    # actions shape: (T, 7) -> [x, y, z, rx, ry, rz, gripper]
    actions = episode_data['action'] 
    print(f"正在回放回合 {episode_idx}, 步数: {len(actions)}")

    # -------------------------------------------------------------------------
    # 2. 初始化环境
    # -------------------------------------------------------------------------
    xml_path = "/home/blzgz/data_get_mujoco/model/franka_emika_panda/mjx_scene.xml"
    # 注意：这里不加载相机以提高回放流畅度，专注于观察物理行为
    env = MujocoEnv(xml_path, frequency=frequency, camera_names=[])
    
    # 重置环境
    env.reset()
    
    # -------------------------------------------------------------------------
    # [新增] 自动计算 Home Keyframe
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("正在计算适合当前数据的 Home Keyframe (QPOS)...")
    
    # 1. 获取第一帧的目标位姿
    first_action = actions[0]
    target_pos = first_action[:3]
    target_rotvec = first_action[3:6]
    target_gripper = first_action[6]
    
    # 2. 设置 Mocap 并预热 IK
    env.data.mocap_pos[0] = target_pos
    r = R.from_rotvec(target_rotvec)
    quat_xyzw = r.as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    env.data.mocap_quat[0] = quat_wxyz
    
    print("正在执行 IK 预热 (让机械臂稳定在起始位置)...")
    for _ in range(200): # 运行 200 步让物理系统稳定
        env.step(target_gripper)
        # env.render() # 如果需要看过程可以取消注释
        
    # 3. 获取并打印 QPOS
    current_qpos = env.data.qpos[:9] # 7 arm + 2 gripper
    qpos_str = " ".join([f"{x:.5f}" for x in current_qpos])
    
    print("\n[CRITICAL FIX] 请复制以下行替换 mjx_panda.xml 中的 <keyframe>:")
    print(f'    <key name="home" qpos="{qpos_str}"/>')
    print("="*60 + "\n")
    
    print("环境已重置，开始 Open-loop Replay...")
    print("请观察机械臂是否平稳运动。如果此时抽搐/掉落，说明是环境控制接口问题。")

    # -------------------------------------------------------------------------
    # 3. 回放循环
    # -------------------------------------------------------------------------
    for i, action in enumerate(actions):
        start_time = time.time()
        
        # 解析动作 (与 collect_data_sim.py 写入的格式一致)
        target_pos = action[:3]
        target_rotvec = action[3:6]
        target_gripper = action[6]
        
        # --- 关键：模拟 eval_mujoco_robot.py 的控制逻辑 ---
        
        # 1. 更新 Mocap 位置
        env.data.mocap_pos[0] = target_pos
        
        # 2. 更新 Mocap 旋转 (RotVec -> Quat wxyz)
        r = R.from_rotvec(target_rotvec)
        quat_xyzw = r.as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        env.data.mocap_quat[0] = quat_wxyz
        
        # 3. 步进环境 (触发 IK 和 物理步进)
        env.step(target_gripper)
        
        # 4. 渲染
        env.render()
        
        # 频率控制
        dt = 1.0 / frequency
        elapsed = time.time() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)
            
        if i % 10 == 0:
            print(f"Step {i}/{len(actions)} | Target Pos: {target_pos}")

    print("回放结束。")
    env.close()

if __name__ == "__main__":
    main()