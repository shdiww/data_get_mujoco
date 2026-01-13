import click
import pathlib
import numpy as np
import cv2
import torch
import dill
import hydra
from omegaconf import OmegaConf
from collections import deque
import torchvision.transforms as transforms
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from scipy.spatial.transform import Rotation as R

# 注册 eval 解析器
OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('-c', '--checkpoint', required=True, help='Path to checkpoint')
@click.option('-d', '--dataset_path', default='data/mujoco_demo/replay_buffer.zarr', help='Path to zarr dataset')
@click.option('-e', '--episode_idx', default=0, type=int, help='Episode index to test')
@click.option('--device', default='cuda:0', help='Device to use')
def main(checkpoint, dataset_path, episode_idx, device):
    # 解决 cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # [System] 检查显存
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"[System] GPU Memory: Free={free_mem/1024**3:.2f}GB, Total={total_mem/1024**3:.2f}GB")
        if free_mem < 1 * 1024**3: # Less than 1GB
            print("[Warning] 显存极低！建议先关闭其他占用显存的进程 (如训练脚本、其他 Eval 窗口)。")

    # -------------------------------------------------------------------------
    # 1. 加载模型
    # -------------------------------------------------------------------------
    print(f"正在加载模型: {checkpoint} ...")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    print("\n" + "="*80)
    print("[DEBUG] Model Shape Meta (模型期望的输入格式):")
    print(OmegaConf.to_yaml(cfg.task.shape_meta))
    print("="*80 + "\n")
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()
    
    # [Patch] 强制使用 CenterCrop，与 eval_mujoco_robot.py 保持一致
    if hasattr(policy, 'obs_encoder'):
        for key, transform in policy.obs_encoder.key_transform_map.items():
            if isinstance(transform, torch.nn.Sequential):
                for i, mod in enumerate(transform):
                    if isinstance(mod, CropRandomizer):
                        print(f"[Patch] Replacing CropRandomizer with CenterCrop for {key}")
                        transform[i] = transforms.CenterCrop((mod.crop_height, mod.crop_width))

    # -------------------------------------------------------------------------
    # [新增] 打印模型 Transforms 结构
    # -------------------------------------------------------------------------
    print("="*80)
    print("[DEBUG] Model Transforms (请检查是否有 Resize):")
    if hasattr(policy, 'obs_encoder'):
        print(policy.obs_encoder.key_transform_map)
    print("="*80)

    # -------------------------------------------------------------------------
    # 2. 加载数据集 (Zarr + Video)
    # -------------------------------------------------------------------------
    print(f"正在加载数据集: {dataset_path}")
    try:
        replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r')
    except Exception as e:
        print(f"无法加载数据集: {e}")
        return
        
    episode_data = replay_buffer.get_episode(episode_idx)
    print(f"[DEBUG] Zarr Episode Keys: {list(episode_data.keys())}")
    if 'state' in episode_data:
        print(f"[DEBUG] Zarr 'state' shape: {episode_data['state'].shape}")
        print(f"[DEBUG] Zarr 'state' sample (Step 0): {episode_data['state'][0]}")
        print("  (请确认: 这是 7维 EEF Pose [x,y,z,rx,ry,rz,g] 还是 关节角度?)")
        print("  (如果数值 > 3.14 或 < -3.14，或者是 9维，那很可能是关节角度)")
    
    # 确定视频路径
    dataset_root = pathlib.Path(dataset_path).parent
    video_dir = dataset_root / "videos"
    
    # 映射 Config Key 到 视频文件名 (逻辑与 eval_mujoco_robot.py 一致)
    model_obs_keys = list(cfg.task.shape_meta.obs.keys())
    key_map = {} 
    proprio_key = 'state' # 默认
    
    for key in model_obs_keys:
        shape = cfg.task.shape_meta.obs[key].shape
        if len(shape) == 3:
            if 'camera_0' in key or 'overview' in key or 'agent' in key:
                key_map[key] = 'overview'
            elif 'camera_1' in key or 'hand' in key or 'wrist' in key:
                key_map[key] = 'hand_camera'
        elif len(shape) == 1:
            proprio_key = key
            
    print(f"[Debug] Key Map: {key_map}")
    print(f"[Debug] Proprio Key: {proprio_key}")

    # 打开视频读取器
    video_readers = {}
    for model_key, video_stem in key_map.items():
        # 推断索引 (overview -> 0, hand_camera -> 1)
        cam_idx = -1
        if video_stem == 'overview':
            cam_idx = 0
        elif video_stem == 'hand_camera':
            cam_idx = 1

        # 尝试几种常见的路径结构
        candidates = [
            video_dir / str(episode_idx) / f"{video_stem}.mp4", # videos/0/overview.mp4
            video_dir / f"{episode_idx}_{video_stem}.mp4",     # videos/0_overview.mp4
        ]
        if cam_idx >= 0:
            candidates.append(video_dir / str(episode_idx) / f"{cam_idx}.mp4") # videos/0/0.mp4
            candidates.append(video_dir / f"{episode_idx}_{cam_idx}.mp4")      # videos/0_0.mp4
        candidates.append(video_dir / f"{episode_idx}.mp4")                   # videos/0.mp4 (如果只有单相机)
        
        cap = None
        for p in candidates:
            if p.exists():
                print(f"Found video for {model_key}: {p}")
                cap = cv2.VideoCapture(str(p))
                break
        
        if cap is None or not cap.isOpened():
            print(f"[Error] 找不到 {model_key} 对应的视频文件! 请检查 {video_dir}")
            return
        video_readers[model_key] = cap

    # -------------------------------------------------------------------------
    # 3. 逐帧推理对比
    # -------------------------------------------------------------------------
    gt_states = episode_data['state']
    gt_actions = episode_data['action']
    T = len(gt_actions)
    
    n_obs_steps = cfg.n_obs_steps
    obs_history = deque(maxlen=n_obs_steps)
    
    # [Fix] 强制使用训练时的分辨率 240x320
    # 即使 shape_meta 写的是 480x640，我们也强制 Resize 到 240x320 并 Patch 模型
    TRAIN_H, TRAIN_W = 240, 320
    print(f"\n[Fix] Force Resize to ({TRAIN_W}, {TRAIN_H}) to match Training View.")
    
    # Patch ObsEncoder to accept 240x320 (Bypass AssertionError)
    if hasattr(policy, 'obs_encoder'):
        for key in policy.obs_encoder.key_shape_map:
            shape = policy.obs_encoder.key_shape_map[key]
            if len(shape) == 3: # is image
                policy.obs_encoder.key_shape_map[key] = (shape[0], TRAIN_H, TRAIN_W)
                print(f"[Patch] Updated ObsEncoder expectation for {key} to {(shape[0], TRAIN_H, TRAIN_W)}")

    print(f"[Fix] Using RAW state (Euler) because Normalizer Scale (~0.318) implies 2*PI range.")
    
    policy.reset()

    # -------------------------------------------------------------------------
    # [新增] 检查 Normalizer 统计量
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("检查模型 Normalizer 统计量...")
    print("(请对比下方的 Mean Val 和上面的 Zarr Sample，如果数量级不同，说明单位/坐标系错了)")
    if hasattr(policy, 'normalizer'):
        try:
            if hasattr(policy.normalizer, 'params_dict'):
                for key, val in policy.normalizer.params_dict.items():
                    print(f"Group: {key}") # 'obs' or 'action'
                    try:
                        # 安全获取 keys
                        val_keys = list(val.keys()) if hasattr(val, 'keys') else []
                        
                        # 策略 1: 包含 input_stats (最常见的情况)
                        if 'input_stats' in val_keys:
                            stats = val['input_stats']
                            print(f"  Key: {key}")
                            if 'mean' in stats:
                                print(f"    Mean Val: {stats['mean'].detach().cpu().numpy()}")
                            if 'scale' in val_keys:
                                print(f"    Scale Val: {val['scale'].detach().cpu().numpy()}")
                            continue

                        # 策略 2: 扁平结构 (直接包含 mean/scale)
                        if 'mean' in val_keys and 'scale' in val_keys:
                            mean = val['mean']
                            scale = val['scale']
                            print(f"  Key: {key} (Direct)")
                            print(f"    Mean Val: {mean.detach().cpu().numpy()}")
                            print(f"    Scale Val: {scale.detach().cpu().numpy()}")
                            continue

                        # 策略 3: 嵌套结构 (例如 obs -> state, camera...)
                        if hasattr(val, 'items'):
                            for sub_key, stats in val.items():
                                sub_keys = list(stats.keys()) if hasattr(stats, 'keys') else []
                                
                                # 检查子项是否有 input_stats
                                if 'input_stats' in sub_keys:
                                    input_stats = stats['input_stats']
                                    print(f"  Key: {sub_key}")
                                    if 'mean' in input_stats:
                                        print(f"    Mean Val: {input_stats['mean'].detach().cpu().numpy()}")
                                    continue

                                if 'mean' in sub_keys and 'scale' in sub_keys:
                                    mean = stats['mean']
                                    scale = stats['scale']
                                    print(f"  Key: {sub_key}")
                                    print(f"    Mean Val: {mean.detach().cpu().numpy()}")
                                    print(f"    Scale Val: {scale.detach().cpu().numpy()}")
                                else:
                                    # 忽略 offset/scale 等非容器节点
                                    if len(sub_keys) > 0:
                                        print(f"  Key: {sub_key} (Skipping, keys={sub_keys})")
                        else:
                            print(f"  Val is not dict-like: {type(val)}")
                    except Exception as e:
                        print(f"  Error processing group {key}: {e}")
        except Exception as e:
            print(f"无法读取 Normalizer 统计量: {e}")
    print("="*80 + "\n")
    
    print(f"\n开始 Overfitting Test (Episode {episode_idx}, Steps {T})...")
    print(f"{'Step':<6} | {'MSE':<10} | {'Pred Pos':<25} | {'GT Pos':<25}")
    print("-" * 80)
    
    mse_list = []
    
    for t in range(T):
        # 1. 读取图像
        current_imgs = {}
        for model_key, cap in video_readers.items():
            ret, frame = cap.read()
            if not ret:
                print(f"[Warning] Video ended early at step {t}")
                break
            # OpenCV 读入是 BGR，转为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_imgs[model_key] = frame
            
        if len(current_imgs) < len(video_readers):
            break
            
        # 2. 预处理 (Resize & Construct Obs)
        current_step = {}
        for model_key, img in current_imgs.items():
            # [Fix] Resize to 320x240 (Width, Height)
            resized_img = cv2.resize(img, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
            current_step[model_key] = resized_img
            
        # [Fix] 状态处理: 直接使用 Zarr 原始数据 (Euler/RotVec + Gripper)
        # Zarr: [x, y, z, rx, ry, rz, gripper]
        current_step[proprio_key] = gt_states[t].astype(np.float32)
        
        # 3. 更新历史
        obs_history.append(current_step)
        # 填充初始历史 (Padding)
        if t == 0:
            while len(obs_history) < n_obs_steps:
                obs_history.append(current_step)
        
        # 4. 构造 Batch
        obs_dict = {}
        for key in model_obs_keys:
            if key in key_map:
                # Image: (T, H, W, C) -> (B, T, C, H, W) + Normalize
                imgs = np.stack([x[key] for x in obs_history])
                
                # [DEBUG] 操作 2: 可视化模型真正的输入
                if t == 0:
                    debug_img = imgs[-1] # (H, W, C) RGB
                    save_path = f"debug_input_step_{t}_{key}.png"
                    # RGB -> BGR for OpenCV
                    cv2.imwrite(save_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                    print(f"[DEBUG] Saved model input image (Check Crop/Resize): {save_path}")
                
                imgs = np.moveaxis(imgs, -1, 1).astype(np.float32) / 255.0
                imgs_tensor = torch.from_numpy(imgs)
                # [Note] 模型内部 obs_encoder 已包含 ImageNet Normalization，此处不应重复
                obs_dict[key] = imgs_tensor.unsqueeze(0).contiguous().to(device)
            elif key == proprio_key:
                # State: (T, D) -> (B, T, D)
                states = np.stack([x[key] for x in obs_history])
                obs_dict[key] = torch.from_numpy(states).unsqueeze(0).contiguous().to(device)
        
        # 5. 推理
        with torch.no_grad():
            result = policy.predict_action(obs_dict)
            # 取第一步动作
            pred_action = result['action'][0, 0].cpu().numpy()
            
        # 6. 对比
        gt_action = gt_actions[t]
        # 计算 MSE (均方误差)
        mse = np.mean((pred_action - gt_action)**2)
        mse_list.append(mse)
        
        if t % 10 == 0:
            pred_str = f"[{pred_action[0]:.3f}, {pred_action[1]:.3f}, {pred_action[2]:.3f}]"
            gt_str = f"[{gt_action[0]:.3f}, {gt_action[1]:.3f}, {gt_action[2]:.3f}]"
            print(f"{t:<6} | {mse:<10.6f} | {pred_str:<25} | {gt_str:<25}")
        
        # [Fix] 显式释放显存，防止 OOM
        del obs_dict
        del result
        torch.cuda.empty_cache()

    print("-" * 80)
    avg_mse = np.mean(mse_list)
    print(f"Average MSE: {avg_mse:.6f}")
    
    # 判定标准 (MSE < 0.01 通常认为拟合良好)
    if avg_mse < 0.01:
        print("✅ 测试通过: 模型能够准确拟合训练数据。")
        print("   结论: 模型权重正常，预处理逻辑正常。问题出在仿真环境的实时 Observation (如相机参数/光照/归一化)。")
    else:
        print("❌ 测试失败: 模型无法拟合训练数据。")
        print("   结论: 推理代码的预处理逻辑 (Resize/Crop/Normalization) 与训练时不一致，或者模型未训练收敛。")

if __name__ == '__main__':
    main()