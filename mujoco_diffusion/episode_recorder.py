import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class EpisodeBuffer:
    state: List[np.ndarray] = field(default_factory=list)
    action: List[np.ndarray] = field(default_factory=list)
    timestamp: List[float] = field(default_factory=list)

    def clear(self):
        self.state.clear()
        self.action.clear()
        self.timestamp.clear()


class EpisodeRecorder:
    def __init__(self, replay_buffer, video_recorder, min_steps: int = 10):
        self.replay_buffer = replay_buffer
        self.video_recorder = video_recorder
        self.min_steps = min_steps
        self.buffer = EpisodeBuffer()
        self.is_recording = False

    def start(self):
        self.is_recording = True
        self.buffer.clear()
        episode_idx = self.replay_buffer.n_episodes
        self.video_recorder.start_recording(episode_idx)

    def append(self, obs: Dict, mocap_pos: np.ndarray, mocap_quat_wxyz: np.ndarray, gripper_target: float):
        if not self.is_recording:
            return

        self.video_recorder.write_frame(obs['images'])

        eef_pos = obs['robot_eef_pose'][:3]
        eef_quat = obs['robot_eef_pose'][3:]  # wxyz
        r = R.from_quat(eef_quat[[1, 2, 3, 0]])
        gripper_width = np.mean(obs['qpos'][-2:])
        state_7d = np.concatenate([eef_pos, r.as_rotvec(), [gripper_width]])

        r_target = R.from_quat(mocap_quat_wxyz[[1, 2, 3, 0]])
        action_7d = np.concatenate([mocap_pos, r_target.as_rotvec(), [gripper_target]])

        self.buffer.state.append(state_7d)
        self.buffer.action.append(action_7d)
        self.buffer.timestamp.append(time.time())

    def stop(self, save: bool = True):
        self.is_recording = False

        if save and len(self.buffer.timestamp) <= self.min_steps:
            save = False

        self.video_recorder.stop_recording(save=save)

        if save:
            data = {
                'state': np.array(self.buffer.state, dtype=np.float32),
                'action': np.array(self.buffer.action, dtype=np.float32),
                'timestamp': np.array(self.buffer.timestamp, dtype=np.float64)
            }
            self.replay_buffer.add_episode(data, compressors='disk')

        self.buffer.clear()
        return save
