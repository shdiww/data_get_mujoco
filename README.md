# MuJoCo Panda Pick and Place with Mink

这是一个基于 MuJoCo 物理引擎和 Mink (Inverse Kinematics Library) 的 Franka Emika Panda 机械臂抓取放置任务仿真项目。

## 环境配置 (Installation)

本项目基于 Python 开发。建议使用 Conda 创建独立的虚拟环境。

### 1. 创建 Conda 环境

```bash
conda create -n mujoco_env python=3.10
conda activate mujoco_env
```

### 2. 安装依赖

主要依赖库包括 `mink`、`mujoco` 以及用于控制循环频率的工具。

推荐直接安装带有示例依赖的 `mink`，这会自动安装本项目所需的大部分依赖（包括 `mujoco`, `glfw`, `loop_rate_limiters` 等）：

```bash
pip install "mink[examples]"
```

如果你发现缺少 `loop_rate_limiters` 或其他特定库，可以单独安装：

```bash
pip install loop_rate_limiters glfw numpy
```

## 运行 (Usage)

在项目根目录下，运行主程序启动仿真：

```bash
python main.py
```

## 操作说明 (Controls)

仿真启动后，你可以通过键盘和鼠标控制机械臂末端的目标位置（Mocap Target）以及相机视角。

### 机械臂控制
*   **W / S**: 沿 X 轴移动末端目标 (前后)
*   **A / D**: 沿 Y 轴移动末端目标 (左右)
*   **Q / E**: 沿 Z 轴移动末端目标 (上下)
*   **Z**: 闭合夹爪 (抓取)
*   **X**: 张开夹爪 (释放)
*   **Shift (按住)**: 加快移动速度和夹爪开合速度

### Xbox 手柄控制
*   **左摇杆**: 前后左右移动 (XY轴)
*   **右摇杆 (上下)**: 上下移动 (Z轴)
*   **A 键**: 闭合夹爪
*   **B 键**: 张开夹爪
*   **Start 键**: 重置仿真

### 系统控制
*   **R**: 重置仿真环境，并随机刷新红色方块的位置

### 任务目标

1.  控制机械臂移动到**红色方块**上方。
2.  抓取方块，将其移动到地板上的**绿色半透明目标区域**。
3.  松开夹爪并抬起机械臂。当方块完全进入目标区域且机械臂抬起后，任务完成，环境会自动重置。
   
## 数据录制与回放 (Recording & Replay)

本项目提供了录制和回放操作轨迹的功能，数据保存为 Zarr 格式。

### 录制 (Recording)

运行 `record_episode.py` 脚本开始录制。操作方式与主程序一致，支持三个相机视角。

```bash
python record_episode.py --dataset_path my_episode.zarr --num_episodes 20 --min_episode_steps 80 --save_partial
```

*   **多回合录制**: 会持续收集成功回合，直到达到 `--num_episodes` 后自动保存并退出。
*   **质量过滤**: 成功回合长度小于 `--min_episode_steps` 会被自动丢弃；超过 `--max_episode_steps` 的回合会被自动丢弃并重置。
*   **重置**: 在录制过程中按 `R` 重置，只会丢弃当前回合，不影响之前已经成功的回合。
*   **动作维度**: 默认保存 4 维动作 `[x, y, z, gripper]`。
*   **数据格式**: 保存为 ReplayBuffer 兼容的 Zarr 结构：`data/state`、`data/action` 和 `meta/episode_ends`，可直接用于后续 `diffusion_policy` 数据管线。

### 回放 (Replay)

运行 `replay_episode.py` 脚本回放录制的数据集。

```bash
python replay_episode.py --dataset_path my_episode.zarr --episode_idx 0
```

*   **初始化**: 脚本会读取目标 episode 的第一帧状态，强制设置环境（包括机械臂和方块位置），确保回放初始条件一致。
*   **执行**: 脚本会逐帧读取 `action` 并应用到控制器中，复现录制时的运动轨迹。
*   **兼容性**: 同时兼容新格式（多回合 ReplayBuffer 风格）和旧格式（单回合 `state/action`）。
