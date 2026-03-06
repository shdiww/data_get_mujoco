from pathlib import Path
from dataclasses import dataclass

import mujoco
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

from keyboard import InputListener
from renderer import MujocoRenderer
from success import TaskSuccessEvaluator
from task import PickAndPlaceTask

_HERE = Path(__file__).parent
_XML = _HERE / "model/franka_emika_panda/mjx_scene.xml"

# IK 求解器参数
SOLVER = "daqp"  # 使用的二次规划求解器
POS_THRESHOLD = 1e-4  # 位置误差收敛阈值 (米)
ORI_THRESHOLD = 1e-4  # 方向误差收敛阈值 (弧度)
MAX_ITERS = 20  # 单步IK的最大迭代次数


@dataclass
class AppContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    task: PickAndPlaceTask
    success_evaluator: TaskSuccessEvaluator
    configuration: mink.Configuration
    end_effector_task: mink.FrameTask
    tasks: list
    renderer: MujocoRenderer
    input_listener: InputListener
    rate: RateLimiter
    start_qpos: np.ndarray
    initial_pos: np.ndarray


def converge_ik(
    configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters
):
    """
    运行最多 'max_iters' 次IK计算。

    Args:
        configuration: 机器人构型。
        tasks: IK任务列表。
        dt: 时间步长。
        solver: QP求解器名称。
        pos_threshold: 位置误差收敛阈值。
        ori_threshold: 方向误差收敛阈值。
        max_iters: 最大迭代次数。

    Returns:
        如果位置和姿态误差都低于阈值，则返回 True，否则返回 False。
    """
    for _ in range(max_iters):
        # 调用mink求解IK，得到关节速度
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
        # 将计算出的速度积分到机器人构型中
        configuration.integrate_inplace(vel, dt)

        # 这里只检查第一个任务（末端执行器任务）的误差。
        # 如果有多个任务，你可能需要组合它们的误差。
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        # 如果位置和姿态都达到要求，提前返回
        if pos_achieved and ori_achieved:
            return True
    return False


def _print_controls():
    print("\n---")
    print("控制说明:")
    print("  鼠标左键拖动: 旋转视角")
    print("  鼠标右键拖动: 平移视角")
    print("  鼠标滚轮: 缩放视角")
    print("---\n")
    print("控制说明 (手柄):")
    print("  左摇杆: 前后左右移动 (XY轴)")
    print("  右摇杆 (上下): 上下移动 (Z轴)")
    print("  A 键: 闭合夹爪")
    print("  B 键: 张开夹爪")
    print("  Start 键: 重置仿真")
    print("任务: 将方块移动到地板上的红色圆形区域")
    print("目标点已被限制在安全工作区域内。")


def initialize_app():
    # 加载 MuJoCo 模型和数据
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # 任务与成功判定
    task = PickAndPlaceTask(model)
    success_evaluator = TaskSuccessEvaluator(model)

    # IK 相关初始化
    configuration = mink.Configuration(model)
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    # 渲染初始化
    renderer = MujocoRenderer(model)
    renderer.init()

    # 仿真和 mocap 初始化
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    start_qpos = data.qpos.copy()
    task.randomize_box(model, data)
    mujoco.mj_forward(model, data)
    mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
    initial_pos = data.mocap_pos[0].copy()
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)

    # 输入监听初始化
    pos_limits = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.6, 0.5]])
    abs_pos_limits = pos_limits + initial_pos[:, np.newaxis]
    input_listener = InputListener(model, data, renderer.scene, renderer.cam, task.box_id, abs_pos_limits)
    renderer.register_mouse_callbacks(input_listener)

    # 主循环频率
    rate = RateLimiter(frequency=30.0, warn=False)

    return AppContext(
        model=model,
        data=data,
        task=task,
        success_evaluator=success_evaluator,
        configuration=configuration,
        end_effector_task=end_effector_task,
        tasks=tasks,
        renderer=renderer,
        input_listener=input_listener,
        rate=rate,
        start_qpos=start_qpos,
        initial_pos=initial_pos,
    )


def main():
    app = initialize_app()
    _print_controls()

    try:
        while not app.renderer.should_close():
            dt = app.rate.dt
            app.input_listener.update(dt)
            T_wt = mink.SE3.from_mocap_name(app.model, app.data, "target")
            app.end_effector_task.set_target(T_wt)
            converge_ik(app.configuration, app.tasks, dt, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS)

            # 夹爪控制: 覆盖IK计算出的夹爪关节角度
            gripper_target = app.input_listener.gripper_target
            app.configuration.q[7] = gripper_target
            if app.configuration.q.shape[0] > 8:
                app.configuration.q[8] = gripper_target

            # 将控制信号应用到致动器
            if app.model.nu > 0:
                if app.task.gripper_act1 != -1:
                    app.data.ctrl[app.task.gripper_act1] = gripper_target
                if app.task.gripper_act2 != -1:
                    app.data.ctrl[app.task.gripper_act2] = gripper_target
                for i in range(app.model.nu):
                    if i != app.task.gripper_act1 and i != app.task.gripper_act2:
                        app.data.ctrl[i] = app.configuration.q[i]
            else:
                if app.configuration.q.shape[0] > 8:
                    app.data.qpos[7] = gripper_target
                    app.data.qpos[8] = gripper_target

            mujoco.mj_step(app.model, app.data)
            app.renderer.render(app.data)

            if app.success_evaluator.is_success(app.model, app.data):
                app.task.reset(
                    app.model,
                    app.data,
                    app.start_qpos,
                    app.configuration,
                    app.initial_pos,
                    app.input_listener,
                )

            app.renderer.refresh()
            app.rate.sleep()
    finally:
        app.renderer.close()


if __name__ == "__main__":
    main()
