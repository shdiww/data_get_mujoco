# 环境对齐说明（diffusion_policy 上游）

本仓库的 MuJoCo 图像训练环境，参考了上游仓库：
- https://github.com/real-stanford/diffusion_policy
- 重点参考文件：`conda_environment.yaml`

## 为什么要单独给一份环境模板

上游环境偏“全任务全依赖”，包含 robomimic / robosuite / pybullet 等大量可选组件。
当前仓库的主要目标是：
1. MuJoCo 数据采集（Xbox + Mink IK）
2. diffusion_policy 的 image workspace 训练

因此这里给出 `environment/mujoco_image_conda.yaml`，优先保证核心链路可跑，降低安装失败率和环境体积。

## 关键包分组

- 训练核心：`torch`、`torchvision`、`diffusers`、`accelerate`、`hydra-core`
- 数据处理：`zarr`、`numcodecs`、`av`、`opencv`
- 仿真控制：`mujoco`、`glfw`、`mink`、`pygame`
- 日志可视化：`wandb`、`tensorboard`

## 验证策略

安装后执行：

```bash
python scripts/verify_env.py
```

检查要点：
- 关键模块都能 import
- `torch.cuda.is_available()` 为 True（GPU 训练场景）
