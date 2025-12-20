import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 步长为 2 的卷积做下采样，等价于可学习的「卷积 + stride」，替代 MaxPool
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # ConvTranspose1d = 可学习的上采样（常被称作反卷积/反池化），stride=2 放大时间轴
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        # nn.Sequential：把若干层按顺序打包成一个可调用模块
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            # GroupNorm：按通道分组做归一化，小 batch 时更稳定
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            # Mish：平滑版 ReLU/Swish 的激活，避免硬截断
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1,256,16))
    o = cb(x)
