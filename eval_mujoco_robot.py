import torch
import dill
import hydra
import pathlib
import numpy as np
import cv2
import time
import click
import glfw
from collections import deque
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf
from mujoco_diffusion.mujoco_env import MujocoEnv
import mujoco
from diffusion_policy.common.pytorch_util import dict_apply
import torchvision.transforms as transforms
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer

# 注册 eval 解析器 (配置文件中可能用到)
OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('-c', '--checkpoint', required=True, help='Path to checkpoint')
@click.option('-d', '--device', default='cuda:0', help='Device to use')
@click.option('-f', '--frequency', default=30, type=int, help='Control frequency (Hz)')
@click.option('--steps', default=16, type=int, help='Diffusion inference steps (lower is faster)')
def main(checkpoint, device, frequency, steps):
    # -------------------------------------------------------------------------
    # 0. 预加载配置 (为了获取正确的环境分辨率)
    # -------------------------------------------------------------------------
    print(f"正在预加载 Config 从 {checkpoint} ...")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # [Fix] 自动从 Config 检测频率
    # 避免用户忘记加 -f 10 参数导致 10Hz 模型在 30Hz 环境下运行
    try:
        config_fps = OmegaConf.select(cfg, "task.env_runner.fps")
        if config_fps is None:
            config_fps = OmegaConf.select(cfg, "task.dataset.fps")
        
        if config_fps is not None:
            config_fps = int(config_fps)
            print(f"[Config] 检测到模型训练频率: {config_fps} Hz")
            if frequency == 30 and config_fps != 30:
                print(f"[Config] 自动切换控制频率: 30 Hz -> {config_fps} Hz")
                frequency = config_fps
            elif frequency != config_fps:
                print(f"[Warning] 命令行频率 ({frequency} Hz) 与 Config ({config_fps} Hz) 不一致!")
    except Exception as e:
        print(f"[Config] 频率检测失败: {e}")

    # -------------------------------------------------------------------------
    # 1. 环境设置
    # -------------------------------------------------------------------------
    # MuJoCo XML 模型路径 (与 collect_data_sim.py 保持一致)
    xml_path = "/home/blzgz/data_get_mujoco/model/franka_emika_panda/mjx_scene.xml"
    
    # 相机名称 (需要与训练时的配置对应)
    # 假设 camera_0 -> overview, camera_1 -> hand_camera
    camera_names = ["overview", "hand_camera"]
    
    # [Fix] 从 Config 获取正确的分辨率
    # cfg.task.image_shape 是 [C, H, W] -> [3, 640, 480] (Portrait)
    # MuJoCo 默认可能是 Landscape (640, 480)，导致宽高比不匹配 (Aspect Ratio Trap)
    target_h = cfg.task.image_shape[1]
    target_w = cfg.task.image_shape[2]
    print(f"[Fix] 强制设置 MuJoCo 渲染分辨率: W={target_w}, H={target_h}")

    # 初始化 MuJoCo 环境
    # 使用 30Hz 以匹配数据采集频率
    env = MujocoEnv(xml_path, camera_names=camera_names, frequency=frequency, obs_resolution=(target_w, target_h))
    
    # 获取对象 ID 用于评分
    red_box_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
    target_zone_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")

    # -------------------------------------------------------------------------
    # 2. 推理变量初始化
    # -------------------------------------------------------------------------
    policy = None
    # cfg = None # Already loaded
    n_obs_steps = 1 # 默认值，加载模型后会更新
    obs_history = deque(maxlen=n_obs_steps)
    
    running = False
    
    print("========================================================")
    print("环境已就绪！")
    print(f"检查点路径: {checkpoint}")
    print("在 OpenCV 窗口中按 's' 加载模型并开始评估。")
    print("按 'q' 退出。")
    print("========================================================")
    
    # 重置环境获取初始观测
    obs = env.reset()
    
    def get_score():
        # 获取位置
        box_pos = env.data.xpos[red_box_id]
        zone_pos = env.data.xpos[target_zone_id]
        
        # 计算 XY 平面距离
        xy_dist = np.linalg.norm(box_pos[:2] - zone_pos[:2])
        
        # 评分规则:
        # 距离分: 距离 < 0.01 满分, > 0.2 零分
        dist_score = np.clip(1.0 - (xy_dist - 0.01) / (0.2 - 0.01), 0.0, 1.0)
        
        # 高度分: > 0.05 满分
        height_score = 1.0 if box_pos[2] > 0.05 else 0.0
        
        return (dist_score + height_score) / 2.0, xy_dist, box_pos[2]

    # 主循环
    last_loop_time = time.time()
    while not glfw.window_should_close(env.window):
        start_time = time.time()
        
        # 计算实际循环频率 (Real FPS)
        loop_dt = start_time - last_loop_time
        actual_fps = 1.0 / loop_dt if loop_dt > 0 else 0.0
        last_loop_time = start_time

        # ---------------------------------------------------------------------
        # 可视化与交互
        # ---------------------------------------------------------------------
        # 获取用于显示的图像 (Overview 相机)
        img = obs['images']['overview'] # RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 计算当前分数
        score, dist, height = get_score()

        # 叠加状态文本
        status = "Policy Control" if running else "Idle (Press 's' to start)"
        color = (0, 255, 0) if running else (0, 0, 255)
        cv2.putText(img_bgr, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img_bgr, f"Score: {score:.2f} | FPS: {actual_fps:.1f}/{env.frequency}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("MuJoCo Eval", img_bgr)
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        elif key == ord('s') and not running:
            # -----------------------------------------------------------------
            # 加载模型 (按 's' 触发)
            # -----------------------------------------------------------------
            print(f"正在实例化模型...")
            try:
                # payload 和 cfg 已经在开头加载了，无需重复加载
                # payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
                
                # 实例化 workspace 和 policy
                cls = hydra.utils.get_class(cfg._target_)
                workspace = cls(cfg)
                workspace.load_payload(payload)
                
                # 获取 policy
                policy = workspace.model
                if cfg.training.use_ema:
                    policy = workspace.ema_model
                
                policy.to(device)
                policy.eval()
                
                # [FIX] 覆盖推理步数 (Inference Steps)
                # 训练时通常用 100 步，但推理时 DDIM 只需要 10-20 步即可
                if hasattr(policy, 'num_inference_steps'):
                    policy.num_inference_steps = steps
                    print(f"[Config] 推理步数已从默认值调整为: {steps} (加速推理)")

                # [FIX] Patch CropRandomizer -> CenterCrop
                # 强制模型在推理时使用中心裁剪，消除 RandomCrop 带来的系统性偏差 (Systematic Bias)
                # 同时保持输入形状与 shape_meta 一致，避免 assert 错误
                if hasattr(policy, 'obs_encoder'):
                    for key, transform in policy.obs_encoder.key_transform_map.items():
                        if isinstance(transform, torch.nn.Sequential):
                            for i, mod in enumerate(transform):
                                if isinstance(mod, CropRandomizer):
                                    print(f"[Patch] Replacing CropRandomizer with CenterCrop for {key}")
                                    transform[i] = transforms.CenterCrop((mod.crop_height, mod.crop_width))

                # 更新观测参数
                n_obs_steps = cfg.n_obs_steps
                obs_history = deque(maxlen=n_obs_steps)
                
                print("模型加载成功！")
                print(f"观测历史步数 (n_obs_steps): {n_obs_steps}")
                
                # 清空历史并重置环境
                obs_history.clear()
                obs = env.reset()
                running = True
                
            except Exception as e:
                print(f"加载模型出错: {e}")
                import traceback
                traceback.print_exc()
        
        # ---------------------------------------------------------------------
        # 策略控制循环
        # ---------------------------------------------------------------------
        if running:
            # 1. 处理观测数据
            # 构造当前步的字典，需匹配 shape_meta
            current_step = {}
            
            # 1. Resize 到 shape_meta 定义的尺寸 (模型期望的输入尺寸)
            c, h_in, w_in = cfg.task.shape_meta.obs.camera_0.shape
            img_overview = obs['images']['overview']
            
            # # [DEBUG] 终极 Debug: 保存原始图像
            # # 检查点: 1. Shape 是否为 (480, 640, 3)? 2. 内容是否为全局视角?
            # if not hasattr(policy, 'debug_raw_saved'):
            #     policy.debug_raw_saved = True
            #     print(f"Raw Image Shape: {img_overview.shape}")
            #     cv2.imwrite("debug_raw_overview.png", cv2.cvtColor(img_overview, cv2.COLOR_RGB2BGR))

            current_step['camera_0'] = cv2.resize(img_overview, (w_in, h_in), interpolation=cv2.INTER_AREA)

            c, h_in, w_in = cfg.task.shape_meta.obs.camera_1.shape
            img_hand = obs['images']['hand_camera']
            
            # [DEBUG] 终极 Debug: 保存原始图像
            # 检查点: 1. Shape 是否为 (480, 640, 3)? 2. 内容是否为全局视角?
            if not hasattr(policy, 'debug_raw_saved'):
                policy.debug_raw_saved = True
                print(f"Raw Image Shape: {img_hand.shape}")
                cv2.imwrite("debug_raw_hand.png", cv2.cvtColor(img_hand, cv2.COLOR_RGB2BGR))

            current_step['camera_1'] = cv2.resize(img_hand, (w_in, h_in), interpolation=cv2.INTER_AREA)
            
            # 状态: 7D (eef_pos + eef_rotvec + gripper_width)
            eef_pos = obs['robot_eef_pose'][:3]
            eef_quat = obs['robot_eef_pose'][3:] # wxyz
            # 转换 wxyz -> xyzw (scipy 需要 xyzw)
            r = R.from_quat(eef_quat[[1, 2, 3, 0]])
            
            # 夹爪宽度 (取两个手指的平均值)
            gripper_width = np.mean(obs['qpos'][-2:])
            
            state_7d = np.concatenate([eef_pos, r.as_rotvec(), [gripper_width]])
            current_step['state'] = state_7d.astype(np.float32)
            
            # 添加到历史记录
            obs_history.append(current_step)
            
            # 处理回合开始 (如果历史记录不足，进行填充)
            if len(obs_history) < n_obs_steps:
                # 用第一帧填充历史
                while len(obs_history) < n_obs_steps:
                    obs_history.append(current_step)
            
            # 2. 准备推理 Batch
            # 堆叠历史: (T, ...)
            imgs_0 = np.stack([x['camera_0'] for x in obs_history])
            imgs_1 = np.stack([x['camera_1'] for x in obs_history])
            states = np.stack([x['state'] for x in obs_history])
            
            # 处理图像: (T, H, W, C) -> (B, T, C, H, W)
            # 归一化到 [0, 1]
            imgs_0 = np.moveaxis(imgs_0, -1, 1).astype(np.float32) / 255.0
            imgs_1 = np.moveaxis(imgs_1, -1, 1).astype(np.float32) / 255.0
            
            # 添加 Batch 维度并移动到设备
            imgs_0_t = torch.from_numpy(imgs_0).unsqueeze(0).to(device)
            imgs_1_t = torch.from_numpy(imgs_1).unsqueeze(0).to(device)
            states_t = torch.from_numpy(states).unsqueeze(0).to(device)
            
            obs_dict = {
                'camera_0': imgs_0_t,
                'camera_1': imgs_1_t,
                'state': states_t
            }
            
            # 3. 预测动作
            with torch.no_grad():
                # [DEBUG] Step 1: 保存模型看到的图像 (用于验证 Crop 是否正确)
                # 这张图展示了进入 ResNet 之前的真实像素，包含了 Resize 和 Crop 的结果
                if not hasattr(policy, 'debug_saved'):
                    policy.debug_saved = True
                    print("[DEBUG] 正在保存推理视图 (debug_inference_view.png)...")
                    try:
                        # 遍历所有相机 (camera_0, camera_1)
                        for cam_name in ['camera_0', 'camera_1']:
                            if hasattr(policy, 'obs_encoder') and cam_name in policy.obs_encoder.key_transform_map:
                                transform = policy.obs_encoder.key_transform_map[cam_name]
                                # 取第一帧 (B=1, T, C, H, W) -> (C, H, W)
                                img_tensor = obs_dict[cam_name][0, 0] 
                                # 增加 Batch 维以便 transform 处理: (1, C, H, W)
                                img_tensor = img_tensor.unsqueeze(0)
                                
                                # 应用变换，但在 Normalize 之前停止 (为了看到原始像素)
                                debug_img = img_tensor
                                if isinstance(transform, torch.nn.Sequential):
                                    for module in transform:
                                        if isinstance(module, torchvision.transforms.Normalize):
                                            break
                                        debug_img = module(debug_img)
                                
                                # 转回 numpy 图片: (1, C, H, W) -> (H, W, C)
                                debug_img_np = debug_img[0].cpu().numpy().transpose(1, 2, 0)
                                debug_img_np = (np.clip(debug_img_np, 0, 1) * 255).astype(np.uint8)
                                debug_img_bgr = cv2.cvtColor(debug_img_np, cv2.COLOR_RGB2BGR)
                                filename = f"debug_inference_view_{cam_name}.png"
                                cv2.imwrite(filename, debug_img_bgr)
                                print(f"[DEBUG] 已保存 '{filename}'. Shape: {debug_img_np.shape}")
                    except Exception as e:
                        print(f"[DEBUG] 保存调试图像失败: {e}")

                result = policy.predict_action(obs_dict)
                # result['action'] 形状为 (B, T_pred, D_action)
                # 我们取第一个 batch
                action = result['action'][0].detach().cpu().numpy()
            
            # 4. 执行动作
            # 我们执行预测序列中的第一个动作 (闭环控制)
            target_action = action[0] # 7 dim: pos(3), rotvec(3), gripper(1)
            
            target_pos = target_action[:3]
            target_rotvec = target_action[3:6]
            target_gripper = target_action[6] 
            
            # 更新 Mocap 位置
            env.data.mocap_pos[0] = target_pos
            
            # 更新 Mocap 旋转 (RotVec -> Quat wxyz)
            r_target = R.from_rotvec(target_rotvec)
            quat_xyzw = r_target.as_quat()
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            env.data.mocap_quat[0] = quat_wxyz
            
            # 步进环境 (包含夹爪控制)
            obs = env.step(target_gripper)
            
            # 更新 MuJoCo Viewer
            env.render()
            
        else:
            # 空闲模式: 仅渲染场景
            # 不步进物理，所以机器人保持静止
            env.render()
        
        # 频率控制: 确保循环不会跑得比设定频率更快，并稳定帧率
        elapsed = time.time() - start_time
        if elapsed < env.dt:
            time.sleep(env.dt - elapsed)
            
    env.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()