import random
from pathlib import Path
from typing import Dict

import cv2
import imageio.v2 as imageio

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class MujocoImageRunner(BaseImageRunner):
    """Lightweight preview runner.

    This runner does not execute policy in simulator yet.
    It periodically exports one recorded camera stream as GIF so W&B can
    show a visual heartbeat during training.
    """

    def __init__(self, output_dir: str, dataset_path: str, camera_idx: int = 0, max_frames: int = 80):
        super().__init__(output_dir)
        self.dataset_path = Path(dataset_path)
        self.camera_idx = camera_idx
        self.max_frames = max_frames
        self._counter = 0

    def run(self, policy: BaseImagePolicy) -> Dict[str, object]:
        _ = policy
        videos_root = self.dataset_path.joinpath("videos")
        if not videos_root.exists():
            return {}

        episode_dirs = sorted([p for p in videos_root.iterdir() if p.is_dir()])
        if not episode_dirs:
            return {}

        episode_dir = random.choice(episode_dirs)
        mp4_path = episode_dir.joinpath(f"{self.camera_idx}.mp4")
        if not mp4_path.exists():
            return {}

        cap = cv2.VideoCapture(str(mp4_path))
        frames = []
        while len(frames) < self.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            return {}

        artifact_dir = Path(self.output_dir).joinpath("media")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        gif_path = artifact_dir.joinpath(f"preview_ep{episode_dir.name}_{self._counter}.gif")
        self._counter += 1

        imageio.mimsave(gif_path, frames, duration=0.05)
        return {
            "preview/episode": int(episode_dir.name),
            "preview/gif_path": str(gif_path)
        }
