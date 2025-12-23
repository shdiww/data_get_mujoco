import pathlib
import cv2
import time
import numpy as np
import mujoco
import glfw
from scipy.spatial.transform import Rotation as R
from diffusion_policy.common.replay_buffer import ReplayBuffer # This is correct
from mujoco_diffusion.mujoco_env import MujocoEnv # Correctly imports from the package
from mujoco_diffusion.gamepad_controller import GamepadController # Correctly imports from the package
from mujoco_diffusion.video_recorder import VideoRecorder

def main():
    # 配置
    xml_path = "/home/blzgz/data_get_mujoco/model/franka_emika_panda/mjx_scene.xml"
    # 根据 XML 文件修改相机名称: "overview" (场景相机) 和 "hand_camera" (手眼相机)
    camera_names = ["overview", "hand_camera"]
    
    # 数据集根目录
    dataset_root = pathlib.Path("data/mujoco_demo")
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    # 视频目录
    video_dir = dataset_root.joinpath("videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    # 1. 初始化环境 (替代 RealEnv)
    env = MujocoEnv(xml_path, camera_names=camera_names, frequency=30)
    
    # 2. 初始化手柄 (替代 SpaceMouse)
    # max_pos_speed: m/s, max_rot_speed: rad/s
    controller = GamepadController(max_pos_speed=0.1)
    
    print("准备就绪！请使用左摇杆/扳机键移动。旋转功能已禁用。")

    obs = env.reset()
    
    # 初始化夹爪目标 (0.04 为张开)
    gripper_target = 0.04
    GRIPPER_SPEED = 0.15 # 夹爪开合速度
    
    # 限制 Mocap 的移动范围 (可选)
    POS_LIMITS = np.array([[-0.5, 0.8], [-0.5, 0.5], [0.0, 0.8]]) 

    # --- 录制相关初始化 ---
    zarr_path = str(dataset_root.joinpath("replay_buffer.zarr"))
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode='a')
    print(f"数据集将保存至: {dataset_root}")
    
    is_recording = False
    last_time = time.time()
    fps = 0.0

    # 临时缓存 (只存低维数据)
    episode_low_dim = {
        'state': [],
        'action': [],
        'timestamp': []
    }

    # 初始化视频录制器
    recorder = VideoRecorder(video_dir, camera_names, fps=30.0, resolution=(env.obs_width, env.obs_height))

    def start_recording():
        nonlocal is_recording, episode_low_dim
        is_recording = True
        print("\n[录制] 开始")
        
        # 重置缓存
        episode_low_dim['state'] = []
        episode_low_dim['action'] = []
        episode_low_dim['timestamp'] = []
        
        # 准备视频录制
        episode_idx = replay_buffer.n_episodes
        recorder.start_recording(episode_idx)

    def stop_recording(save=True):
        nonlocal is_recording, episode_low_dim
        is_recording = False
        print(f"\n[录制] 停止。")

        # 检查数据长度，如果太短则不保存
        if save and len(episode_low_dim['timestamp']) <= 10:
            print("回合太短，已丢弃。")
            save = False

        recorder.stop_recording(save=save)

        if save:
            print(f"正在保存 {len(episode_low_dim['timestamp'])} 步数据...")
            
            data = {
                'state': np.array(episode_low_dim['state'], dtype=np.float32),
                'action': np.array(episode_low_dim['action'], dtype=np.float32),
                'timestamp': np.array(episode_low_dim['timestamp'], dtype=np.float64)
            }
            
            replay_buffer.add_episode(data, compressors='disk')
            print(f"已保存至 Zarr。总回合数: {replay_buffer.n_episodes}")

    # 主循环 (使用 env.window 判断退出)
    while not glfw.window_should_close(env.window):
            start_time = time.time()

            # --- 获取动作 ---
            # 在真实环境中，这里读取 SpaceMouse
            # 在仿真中，我们读取手柄
            # dpos: [dx, dy, dz], drot: [drx, dry, drz]
            # gripper_cmd: -1 (闭), 1 (开), 0 (不动). 忽略 drot 以禁用旋转
            dpos, _, gripper_cmd, reset_cmd = controller.get_action()

            if reset_cmd:
                print("已通过手柄重置！")
                if is_recording:
                    stop_recording(save=True)
                env.reset()
                gripper_target = 0.04

            # --- 录制控制 ---
            if controller.is_record_toggled():
                if not is_recording:
                    start_recording()
                else:
                    stop_recording(save=True)

            # --- 更新 Mocap 目标 (Target) ---
            # 我们直接修改 data.mocap_pos，Mink IK 会在 env.step 中让机械臂去追这个点
            
            # 平移
            env.data.mocap_pos[0] += dpos * env.dt
            # 限制范围
            np.clip(env.data.mocap_pos[0], POS_LIMITS[:, 0], POS_LIMITS[:, 1], out=env.data.mocap_pos[0])

            # --- 环境步进 (执行 IK) ---
            # 连续更新夹爪目标
            gripper_target += gripper_cmd * GRIPPER_SPEED * env.dt
            gripper_target = np.clip(gripper_target, 0.0, 0.04)
            
            # --- 收集数据 (在 Step 之前) ---
            if is_recording:
                # 1. 写入视频帧
                recorder.write_frame(obs['images'])
                
                # 2. 记录低维数据 (转为 7维: XYZ + RotVec + Gripper)
                # State: 实际末端位姿
                eef_pos = obs['robot_eef_pose'][:3]
                eef_quat = obs['robot_eef_pose'][3:] # wxyz
                r = R.from_quat(eef_quat[[1, 2, 3, 0]]) # xyzw
                
                # 获取夹爪实际宽度 (假设最后两个关节是手指)
                gripper_width = np.mean(obs['qpos'][-2:])
                state_7d = np.concatenate([eef_pos, r.as_rotvec(), [gripper_width]])
                
                # Action: 目标位姿 (Mocap)
                mocap_pos = env.data.mocap_pos[0].copy()
                mocap_quat = env.data.mocap_quat[0].copy() # wxyz
                r_target = R.from_quat(mocap_quat[[1, 2, 3, 0]])
                action_7d = np.concatenate([mocap_pos, r_target.as_rotvec(), [gripper_target]])

                episode_low_dim['state'].append(state_7d)
                episode_low_dim['action'].append(action_7d)
                episode_low_dim['timestamp'].append(time.time())

            obs = env.step(gripper_target)

            # --- 任务检查 ---
            if env.task.check_completion(env.model, env.data):
                print("任务完成！正在重置...")
                if is_recording:
                    stop_recording(save=True)
                env.reset()
                gripper_target = 0.04 # 重置后夹爪默认张开

            # --- 同步 Viewer ---
            # 计算 FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - last_time)
            last_time = curr_time
            
            env.render(fps=fps, gripper_val=gripper_target, is_recording=is_recording)

            # --- 频率控制 ---
            # 确保循环频率与 env.frequency 一致
            elapsed = time.time() - start_time
            sleep_time = env.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    env.close()

if __name__ == "__main__":
    main()
