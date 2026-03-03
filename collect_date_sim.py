import pathlib
import time
import numpy as np
import glfw
from diffusion_policy.common.replay_buffer import ReplayBuffer # This is correct
from mujoco_diffusion.mujoco_env import MujocoEnv # Correctly imports from the package
from mujoco_diffusion.gamepad_controller import GamepadController # Correctly imports from the package
from mujoco_diffusion.video_recorder import VideoRecorder
from mujoco_diffusion.episode_recorder import EpisodeRecorder

def main():
    # 配置
    # [修改] 将采集频率设置为 10Hz, 与你的评估频率保持一致
    FREQUENCY = 20
    xml_path = "/home/blzgz/data_get_mujoco/model/franka_emika_panda/mjx_scene.xml"
    # 根据 XML 文件修改相机名称: "overview" (场景相机) 和 "hand_camera" (手眼相机)
    camera_names = ["overview", "hand_camera"]
    
    print(f"XML Path: {xml_path}")
    print("!!! 警告: 请确保此 XML 文件与 eval_mujoco_robot.py 完全一致，且在采集/训练/推理全过程中不再修改任何几何参数 !!!")
    
    # 数据集根目录
    dataset_root = pathlib.Path("data/mujoco_demo")
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    # 视频目录
    video_dir = dataset_root.joinpath("videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    # 1. 初始化环境 (替代 RealEnv)
    # [修改] 使用定义的频率，并增加 IK 迭代次数(ik_iterations=20)以保证大步长下的解算精度
    env = MujocoEnv(xml_path, camera_names=camera_names, frequency=FREQUENCY,ik_iterations=20)
    
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
    
    last_time = time.time()
    fps = 0.0

    # 录制管理器（低维 + 视频 + zarr）
    recorder = VideoRecorder(video_dir, camera_names, fps=float(FREQUENCY),resolution=(env.obs_width, env.obs_height))
    episode_recorder = EpisodeRecorder(replay_buffer=replay_buffer, video_recorder=recorder, min_steps=10)

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
                if episode_recorder.is_recording:
                    saved = episode_recorder.stop(save=True)
                    print(f"重置前录制保存: {saved}")
                env.reset()
                gripper_target = 0.04

            # --- 录制控制 ---
            if controller.is_record_toggled():
                if not episode_recorder.is_recording:
                    episode_recorder.start()
                    print("\n[录制] 开始")
                else:
                    saved = episode_recorder.stop(save=True)
                    print(f"\n[录制] 停止，保存: {saved}")

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
            episode_recorder.append(
                obs=obs,
                mocap_pos=env.data.mocap_pos[0].copy(),
                mocap_quat_wxyz=env.data.mocap_quat[0].copy(),
                gripper_target=gripper_target
            )

            obs = env.step(gripper_target)

            # --- 任务检查 ---
            if env.task.check_completion(env.model, env.data):
                print("任务完成！正在重置...")
                if episode_recorder.is_recording:
                    saved = episode_recorder.stop(save=True)
                    print(f"重置前录制保存: {saved}")
                env.reset()
                gripper_target = 0.04 # 重置后夹爪默认张开

            # --- 同步 Viewer ---
            # 计算 FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - last_time)
            last_time = curr_time
            
            env.render(fps=fps, gripper_val=gripper_target, is_recording=episode_recorder.is_recording)

            # --- 频率控制 ---
            # 确保循环频率与 env.frequency 一致
            elapsed = time.time() - start_time
            sleep_time = env.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    env.close()

if __name__ == "__main__":
    main()
