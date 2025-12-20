from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # 传给 scheduler.step 的额外参数
            **kwargs):
        super().__init__()

        # 解析动作维度
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # 观测编码后的特征维度
        obs_feature_dim = obs_encoder.output_shape()[0]

        # 创建扩散模型
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            # 视觉特征作为全局条件，U-Net 输入只放动作
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        # mask 生成器在训练/推理中锁定观测槽位，避免被噪声覆盖
        # 负责观测/动作的归一化与反归一化
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
 
    def conditional_sample(self, 
            condition_data, 
            condition_mask,
            local_cond=None, 
            global_cond=None,
            generator=None,
            # 传给 scheduler.step 的关键字参数
            **kwargs
            ):
        # 通过扩散反演生成完整轨迹，condition_mask 位置需与已知数据一致
        model = self.model
        scheduler = self.noise_scheduler
        #初始化高斯噪声 
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # 设置反推步数
        scheduler.set_timesteps(self.num_inference_steps)
        #开始反向去噪
        for t in scheduler.timesteps:
            # 1. 应用条件 某些维度/时间点是已知条件，扩散变量不能乱改
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. 预测噪声/残差
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. 计算上一时刻：x_t -> x_t-1
            #    对应 DDPM 反向更新，模型输出噪声梯度
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # 结束时再次强制条件
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    # 用最近 To 步观测作为条件，采样未来 T 步动作，然后只执行其中 n_action_steps 步（receding horizon）
    # 被我理解为类MPC风格化窗口控制

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: 必须包含 "obs" 键
        result: 必须包含 "action" 键
        """
        assert 'past_action' not in obs_dict # 还未实现 past_action
        # 输入归一化
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        # 闭环设置：只取最近 To 步观测，预测 horizon 长度动作

        # 构造输入
        device = self.device
        dtype = self.dtype

        # 根据是否全局条件选择输入形式
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # 用全局视觉特征条件，动作槽位全零
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape 回 B, Do，视觉只编码一次，推理各步复用
            global_cond = nobs_features.reshape(B, -1)
            # 动作部位初始化为零
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # 修补式条件：把视觉特征拼接在动作后作为已知部分
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape 回 B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # 执行采样
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # 反归一化得到真实尺度动作
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # 取出将要执行的窗口动作
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= 训练阶段  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # 归一化输入
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 根据条件方式组装模型输入
        local_cond = None
        global_cond = None
        trajectory = nactions  #x0干净序列
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... 到 B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape 回 B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... 到 B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape 回 B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # 生成修补掩码：哪些位置作为条件保持不变
        condition_mask = self.mask_generator(trajectory.shape)

        # 采样要加到轨迹上的噪声
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # 为每个样本采样随机扩散步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # 随机扩散步 k，模拟前向加噪
        # 按各步噪声方差加噪（即前向扩散）
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # 计算损失的掩码
        loss_mask = ~condition_mask

        # 应用条件，保证条件位被固定
        # 被 mask 标记的位置保持观测/条件不被噪声学习
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # 预测噪声残差（训练目标为噪声/样本残差）
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype) # 只在非条件位计算损失，对应闭环条件
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
