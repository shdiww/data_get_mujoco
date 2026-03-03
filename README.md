# MuJoCo Data Collection for Diffusion Policy

这是一个用于在 MuJoCo 仿真环境中采集机械臂示教数据的项目。该项目整合了 `mujoco` 物理引擎、`mink` 逆运动学求解器和 `pygame` 手柄控制，生成的数据格式（Zarr）可直接用于 Diffusion Policy 的训练。

## 项目结构

*   **`collect_date_sim.py`**: 仿真数据采集的主入口脚本。
*   **`mujoco_diffusion/`**: 仿真核心功能包。
    *   `mujoco_env.py`: 封装了 MuJoCo 环境、渲染逻辑以及基于 Mink 的 IK 控制器。
    *   `gamepad_controller.py`: 封装了 Xbox 手柄的输入处理逻辑。
    *   `task.py`: 定义了“抓取与放置”任务的逻辑（如物体随机化、成功判定）。
*   **`model/`**: 存放 MuJoCo XML 模型文件（如 Franka Emika Panda 机械臂）。
*   **`data/`**: 默认的数据集输出目录。
*   **`diffusion_policy/`**: 包含 ReplayBuffer 等数据处理工具。

## 依赖安装

请确保你的 Python 环境中安装了以下核心依赖：

```bash
pip install mujoco mink glfw pygame zarr numpy scipy
```

*(注：还需要安装 `diffusion_policy` 及其相关依赖)*

## 使用说明

### 1. 启动采集

连接 Xbox 手柄，然后在项目根目录下运行：

```bash
python collect_date_sim.py
```

### 2. 手柄操作指南

| 按键/摇杆 | 功能 |
| :--- | :--- |
| **左摇杆** | 控制机械臂末端在 **XY 平面** 移动 |
| **右摇杆 (上下)** | 控制机械臂末端在 **Z 轴** (高度) 移动 |
| **A 键** | **闭合**夹爪 |
| **B 键** | **张开**夹爪 |
| **Back 键** (左侧小按钮) | **开始 / 停止录制** |
| **Start 键** (右侧小按钮) | **重置环境** (若正在录制，会自动保存当前回合) |

> **注意**: 当前配置下，为了简化抓取任务，末端执行器的**旋转功能已禁用**，机械臂将始终保持垂直向下的姿态。

### 3. 数据录制与保存

*   **开始录制**: 按下 **Back 键**，界面左上角会显示 `RECORDING`。
*   **保存数据**:
    *   再次按下 **Back 键** 停止录制。
    *   或者按下 **Start 键** 手动重置环境。
    *   或者完成任务（方块放入目标区并抬起），环境自动重置。
    *   以上操作都会触发数据保存，数据将追加写入到 `data/mujoco_demo.zarr` 文件中。
*   **无效数据**: 步数过短（<10步）的回合会被自动丢弃。


## Franka + Xbox 新功能规划

已新增一份分阶段落地文档，覆盖可行性判断、模块重构优先级与建议目录结构：

- `docs/franka_xbox_mujoco_plan.md`


## 环境配置（对齐 diffusion_policy）

已根据上游仓库 `real-stanford/diffusion_policy` 的 `conda_environment.yaml`，提供一个更贴近当前 MuJoCo 图像任务的环境模板：

- `environment/mujoco_image_conda.yaml`
- `scripts/setup_env_mujoco_image.sh`
- `scripts/verify_env.py`
- `environment/mujoco_image_requirements.txt`（无 conda 时的 venv 方案）

### 一键创建（推荐）

```bash
bash scripts/setup_env_mujoco_image.sh mujoco-image
```

### 手动方式

```bash
conda env create -f environment/mujoco_image_conda.yaml -n mujoco-image
conda activate mujoco-image
python scripts/verify_env.py
```

> 如果你服务器 CUDA / 驱动版本与 `cudatoolkit=11.8` 不一致，优先调整 `environment/mujoco_image_conda.yaml` 中 `pytorch`/`cudatoolkit` 对应版本再安装。


### 无 conda 时（venv）

```bash
bash scripts/setup_env_mujoco_image.sh
source .venv/bin/activate
```
