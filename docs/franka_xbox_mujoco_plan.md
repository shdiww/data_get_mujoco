# Franka + Xbox + MuJoCo 数据采集流程可行性与重构建议

## 结论

该流程完全可实现：
1. 在 MuJoCo 中加载 Franka（或其他机器人）XML；
2. 用 Xbox 手柄仅控制末端位姿目标（mocap target）；
3. 用 Mink（或同类 IK）将末端目标映射到关节命令；
4. 同步录制低维状态（EEF 位姿/夹爪）与多视角相机图像（视频）；
5. 输出为可直接用于 diffusion policy 的训练数据（Zarr + videos）。

## 与当前代码的映射

- 入口脚本 `collect_date_sim.py` 已覆盖手柄输入、录制开关、回合保存。
- `mujoco_diffusion/mujoco_env.py` 已覆盖 MuJoCo 加载、IK 控制、相机渲染、观测输出。
- `mujoco_diffusion/video_recorder.py` 已支持多相机视频写入。

## 还需要补齐的关键点（按优先级）

### P0（必须先做）

1. **统一配置中心**
   - 把 XML 路径、相机名、频率、IK 迭代、位置限制、输出目录集中到配置文件（YAML/JSON）。
   - 避免把硬编码散落在脚本里（当前 `collect_date_sim.py` 存在硬编码路径）。

2. **动作与观测协议固定**
   - 明确 `state`/`action` 的语义（世界系还是基坐标系、四元数顺序、夹爪单位）。
   - 将协议文档化，保证训练、回放、评估一致。

3. **多机器人适配层**
   - 引入 `RobotAdapter`：定义 `eef_site_name`、`mocap_body_name`、`gripper_actuator_ids`、`home_keyframe`。
   - 不同机器人（Franka/UR5 等）只替换适配器，不改主流程。

### P1（强烈建议）

4. **拆分脚本职责**
   - `collect_date_sim.py` 拆成：
     - `runner`（主循环/频率控制）；
     - `teleop`（Xbox 输入映射）；
     - `recording`（低维+视频+zarr）；
     - `task_logic`（重置/成功判定）。

5. **加最小可回归测试**
   - 单元测试：输入映射、姿态转换（wxyz/xyzw）、episode flush 逻辑。
   - 冒烟测试：无手柄时可用脚本动作源跑 100 step 并生成一段 demo 数据。

6. **统一时钟与时间戳**
   - 录制时间戳建议使用控制循环时间基准，避免不同系统调用导致漂移。

### P2（体验优化）

7. **在线数据质量检查**
   - 实时显示 episode 长度、EEF 速度、夹爪开度范围、相机掉帧率。

8. **导出桥接工具**
   - 提供脚本将 Zarr + mp4 转为训练管线所需格式，便于复现实验。

## 建议目录结构（示例）

```text
mujoco_diffusion/
  configs/
    franka_xbox_collect.yaml
  robots/
    base_adapter.py
    franka_adapter.py
    ur5_adapter.py
  control/
    ik_controller.py
    teleop_mapping.py
  env/
    mujoco_env.py
  data/
    recorder.py
    replay_writer.py
    video_writer.py
  runners/
    collect_runner.py
scripts/
  collect_sim.py
  replay_dataset.py
```

## 实施顺序建议（两周版）

- 第 1-2 天：抽离配置 + RobotAdapter（先支持 Franka）。
- 第 3-4 天：拆 runner/recording，保证功能不变。
- 第 5-6 天：补 3 个核心测试（映射、姿态、保存）。
- 第 7-8 天：加无手柄冒烟脚本 + 文档。
- 第 9-10 天：联调训练入口并冻结数据协议。

