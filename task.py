import mujoco
import numpy as np


class PickAndPlaceTask:
    def __init__(self, model):
        # 在初始化时统一获取所有需要的ID
        self.box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
        self.gripper_act1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_joint1")
        self.gripper_act2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_joint2")

    def randomize_box(self, model, data):
        """随机化方块位置"""
        if self.box_id != -1:
            jnt_id = model.body_jntadr[self.box_id]
            if jnt_id != -1:
                q_addr = model.jnt_qposadr[jnt_id]
                data.qpos[q_addr:q_addr+7] = model.qpos0[q_addr:q_addr+7]
                data.qpos[q_addr] += np.random.uniform(-0.05, 0.25) # X轴: 偏前/上
                data.qpos[q_addr+1] += np.random.uniform(-0.1, 0.2)  # Y轴: 偏左

    def reset(self, model, data, start_qpos, configuration, initial_pos, input_listener):
        """重置仿真环境"""
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        # 恢复到预计算的低位姿态
        data.qpos[:] = start_qpos
        self.randomize_box(model, data)
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos) # 同步IK求解器的状态
        # mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
        data.mocap_pos[0] = initial_pos  # 重置时恢复到 XML 定义的初始位置
        input_listener.gripper_target = 0.04
