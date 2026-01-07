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

# 注册 eval 解析器 (配置文件中可能用到)
OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('-c', '--checkpoint', required=True, help='Path to checkpoint')
@click.option('-d', '--device', default='cuda:0', help='Device to use')
@click.option('-f', '--frequency', default=30, type=int, help='Control frequency (Hz)')
def main(checkpoint, device, frequency):
    # -------------------------------------------------------------------------
    # 1. 环境设置
    # -------------------------------------------------------------------------
    # MuJoCo XML 模型路径 (与 collect_data_sim.py 保持一致)
    xml_path = "/home/blzgz/data_get_mujoco/model/franka_emika_panda/mjx_scene.xml"
    
    # 相机名称 (需要与训练时的配置对应)
    # 假设 camera_0 -> overview, camera_1 -> hand_camera
    camera_names = ["overview", "hand_camera"]
    
    # 初始化 MuJoCo 环境
    # 使用 30Hz 以匹配数据采集频率
    env = MujocoEnv(xml_path, camera_names=camera_names, frequency=frequency)
    
    # 获取对象 ID 用于评分
    red_box_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
    target_zone_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")

    # -------------------------------------------------------------------------
    # 2. 推理变量初始化
    # -------------------------------------------------------------------------
    policy = None
    cfg = None
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
    while not glfw.window_should_close(env.window):
        start_time = time.time()
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
        cv2.putText(img_bgr, f"Score: {score:.2f} (Dist: {dist:.3f}, H: {height:.3f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("MuJoCo Eval", img_bgr)
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        elif key == ord('s') and not running:
            # -----------------------------------------------------------------
            # 加载模型 (按 's' 触发)
            # -----------------------------------------------------------------
            print(f"正在从 {checkpoint} 加载模型...")
            try:
                # 加载 payload
                payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
                cfg = payload['cfg']
                
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
            
            # 图像: HWC -> 稍后转换为 BCHW
            # 根据模型配置调整图像尺寸
            h, w = cfg.task.shape_meta.obs.camera_0.shape[1:]
            img_overview = obs['images']['overview']
            current_step['camera_0'] = cv2.resize(img_overview, (w, h), interpolation=cv2.INTER_AREA)

            h, w = cfg.task.shape_meta.obs.camera_1.shape[1:]
            img_hand = obs['images']['hand_camera']
            current_step['camera_1'] = cv2.resize(img_hand, (w, h), interpolation=cv2.INTER_AREA)
            
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