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
