import cv2
import pathlib
import shutil

class VideoRecorder:
    def __init__(self, video_dir, camera_names, fps, resolution):
        self.video_dir = pathlib.Path(video_dir)
        self.camera_names = camera_names
        self.fps = fps
        self.resolution = resolution
        self.writers = {}
        self.is_recording = False
        self.current_episode_idx = -1

    def start_recording(self, episode_idx):
        self.current_episode_idx = episode_idx
        episode_video_dir = self.video_dir.joinpath(str(episode_idx))
        episode_video_dir.mkdir(parents=True, exist_ok=True)
        
        self.writers = {}
        for i, name in enumerate(self.camera_names):
            # 视频文件名: 0.mp4, 1.mp4 ... 对应 camera_names 的顺序
            video_path = str(episode_video_dir.joinpath(f'{i}.mp4'))
            # 使用 mp4v 编码
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writers[name] = cv2.VideoWriter(
                video_path, fourcc, self.fps, self.resolution)
        
        self.is_recording = True

    def stop_recording(self, save=True):
        self.is_recording = False
        # 关闭视频流
        for writer in self.writers.values():
            writer.release()
        self.writers = {}

        if not save:
            # 清理未保存的视频
            episode_video_dir = self.video_dir.joinpath(str(self.current_episode_idx))
            if episode_video_dir.exists():
                shutil.rmtree(str(episode_video_dir))
                print("已删除未保存的视频文件。")

    def write_frame(self, images):
        if not self.is_recording:
            return
            
        for name, img in images.items():
            if name in self.writers:
                # MuJoCo RGB -> OpenCV BGR
                bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.writers[name].write(bgr_img)