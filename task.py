import mujoco
import numpy as np


class PickAndPlaceTask:
    def __init__(self, model):
        # 在初始化时统一获取所有需要的ID
        self.box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
        self.target_zone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        self.target_zone_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone_geom")
        self.attachment_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.gripper_act1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_joint1")
        self.gripper_act2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_joint2")

    def randomize_box(self, model, data):
        """随机化方块位置"""
        if self.box_id != -1:
            jnt_id = model.body_jntadr[self.box_id]
            if jnt_id != -1:
                q_addr = model.jnt_qposadr[jnt_id]
                data.qpos[q_addr:q_addr+7] = model.qpos0[q_addr:q_addr+7]
                data.qpos[q_addr] += np.random.uniform(0.02, 0.28) # X轴: 偏前/上
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

    def check_completion(self, model, data):
        """检查任务是否完成：方块是否落地且完全在目标区域内，且机械臂已抬起。"""
        if self.box_id == -1 or self.target_zone_body_id == -1:
            return False

        geom_id = model.body_geomadr[self.box_id]
        box_half_size = model.geom_size[geom_id]
        box_pos = data.xpos[self.box_id]
        box_mat = data.xmat[self.box_id].reshape(3, 3)

        # 计算方块AABB (XY平面)
        aabb_half_size = np.abs(box_mat).dot(box_half_size)
        min_b = box_pos[:2] - aabb_half_size[:2]
        max_b = box_pos[:2] + aabb_half_size[:2]

        # 从模型中动态获取目标区域位置和大小
        target_zone_pos = data.xpos[self.target_zone_body_id]
        target_zone_size = model.geom_size[self.target_zone_geom_id]
        min_t = target_zone_pos[:2] - target_zone_size[:2]
        max_t = target_zone_pos[:2] + target_zone_size[:2]

        overlap_min = np.maximum(min_b, min_t)
        overlap_max = np.minimum(max_b, max_t)
        overlap_dims = np.maximum(0, overlap_max - overlap_min)
        overlap_area = overlap_dims[0] * overlap_dims[1]
        box_area = (max_b[0] - min_b[0]) * (max_b[1] - min_b[1])

        # 检查方块是否落地 (Z轴高度接近半高，允许 5mm 误差)
        is_on_ground = box_pos[2] <= box_half_size[2] + 0.005

        # 检查机械臂是否抬起 (高度 > 0.2m)
        is_arm_lifted = data.site_xpos[self.attachment_site_id][2] > 0.2

        return is_on_ground and overlap_area >= box_area - 1e-5 and is_arm_lifted