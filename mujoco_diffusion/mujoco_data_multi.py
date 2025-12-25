import numpy as np
import time
from diffusion_policy.common.replay_buffer import ReplayBuffer
from mujoco_diffusion.video_recorder import VideoRecorder

class MujocoDataMulti:
    def __init__(self, dataset_root, video_dir, camera_names, env_obs_width, env_obs_height):
        self.dataset_root = dataset_root
        self.video_dir = video_dir
        self.camera_names = camera_names
        
        # Zarr 路径
        zarr_path = str(dataset_root.joinpath("replay_buffer.zarr"))
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode='a')
        print(f"数据集将保存至: {dataset_root}")

        # 初始化视频录制器
        self.recorder = VideoRecorder(video_dir, camera_names, fps=30.0, resolution=(env_obs_width, env_obs_height))
        
        self.is_recording = False
        self.episode_low_dim = {
            'state': [],
            'action': [],
            'timestep': []
        }

    def start_episode(self):
        """开始一个新的回合：重置缓存，启动视频录制"""
        self.is_recording = True
        print("\n[录制] 开始")
        
        # 1. 重置低维数据缓存
        self.episode_low_dim['state'] = []
        self.episode_low_dim['action'] = []
        self.episode_low_dim['timestep'] = []
        
        # 2. 启动视频录制 (传入当前的 episode index)
        episode_idx = self.replay_buffer.n_episodes
        self.recorder.start_recording(episode_idx)

    def stop_episode(self, save=True):
        """结束当前回合：停止视频录制，保存数据到 Zarr"""
        self.is_recording = False
        print(f"\n[录制] 停止。")

        # 1. 检查数据长度，如果太短则强制不保存
        if save and len(self.episode_low_dim['timestamp']) <= 10:
            print("回合太短，已丢弃。")
            save = False

        # 2. 停止视频录制 (如果 save=False，recorder 会自动删除刚才录的视频文件)
        self.recorder.stop_recording(save=save)

        # 3. 保存低维数据到 Zarr
        if save:
            print(f"正在保存 {len(self.episode_low_dim['timestep'])} 步数据...")
            
            # 构造符合 ReplayBuffer 要求的数据字典
            data = {
                'state': np.array(self.episode_low_dim['state'], dtype=np.float32),
                'action': np.array(self.episode_low_dim['action'], dtype=np.float32),
                'timestep': np.array(self.episode_low_dim['timestep'], dtype=np.float64)
            }
            
            try:
                self.replay_buffer.add_episode(data, compressors='disk')
                print(f"已保存至 Zarr。总回合数: {self.replay_buffer.n_episodes}")
            except Exception as e:
                print(f"保存 Zarr 失败: {e}")

    def add_data(self, state, action, timestamp, images):
        if not self.is_recording:
            return

        self.episode_low_dim['state'].append(state)
        self.episode_low_dim['action'].append(action)
        self.episode_low_dim['timestep'].append(timestamp)
        
        self.recorder.write_frame(images)