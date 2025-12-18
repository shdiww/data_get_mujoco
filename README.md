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

### 系统控制
*   **R**: 重置仿真环境，并随机刷新红色方块的位置

### 任务目标

1.  控制机械臂移动到**红色方块**上方。
2.  抓取方块，将其移动到地板上的**绿色半透明目标区域**。
3.  松开夹爪并抬起机械臂。当方块完全进入目标区域且机械臂抬起后，任务完成，环境会自动重置。