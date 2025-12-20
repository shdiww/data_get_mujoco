from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        # 从zarr文件中复制数据到ReplayBuffer，只加载'img', 'state', 'action'这几个键
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        # 生成验证集掩码
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        # 训练集掩码是验证集掩码的取反
        train_mask = ~val_mask
        # 根据max_train_episodes下采样训练集
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 初始化序列采样器
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        # 创建一个副本作为验证集
        val_set = copy.copy(self)
        # 为验证集创建一个新的采样器，使用验证集掩码
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # 准备用于归一化的数据
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        # 创建线性归一化器
        normalizer = LinearNormalizer()
        # 拟合数据
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # 为图像数据添加归一化器
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        # 返回采样器的长度
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # 从样本中提取agent_pos，数据类型为float32
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        # 调整图像数据维度并归一化到[0,1]
        image = np.moveaxis(sample['img'],-1,1)/255

        # 组织成模型需要的数据格式
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 根据索引采样一个序列
        sample = self.sampler.sample_sequence(idx)
        # 将样本转换为模型需要的数据格式
        data = self._sample_to_data(sample)
        # 将numpy数组转换为torch张量
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    # 测试代码，加载zarr数据
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
