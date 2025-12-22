import mujoco
import numpy as np

class PickAndPlaceTask:
    def __init__(self, model):
        # 获取必要的对象 ID
        self.box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
        self.target_zone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        self.target_zone_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone_geom")
        self.attachment_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        
        # 夹爪致动器 ID
        self.gripper_act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_joint1"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_joint2")
        ]
        # 如果找不到名字，回退到硬编码索引 (假设最后两个是夹爪)
        if -1 in self.gripper_act_ids:
             self.gripper_act_ids = [model.nu - 2, model.nu - 1]

    def randomize_box(self, model, data):
        """随机化方块位置"""
        if self.box_id != -1:
            jnt_id = model.body_jntadr[self.box_id]
            if jnt_id != -1:
                q_addr = model.jnt_qposadr[jnt_id]
                # 重置为初始高度
                data.qpos[q_addr:q_addr+7] = model.qpos0[q_addr:q_addr+7]
                # 在桌面上随机平移
                data.qpos[q_addr] += np.random.uniform(0.02, 0.28)   # X轴
                data.qpos[q_addr+1] += np.random.uniform(-0.1, 0.2)  # Y轴

    def check_completion(self, model, data):
        """检查任务是否完成"""
        if self.box_id == -1 or self.target_zone_body_id == -1:
            return False

        # 获取方块位置
        box_pos = data.xpos[self.box_id]
        
        # 获取目标区域位置
        target_pos = data.xpos[self.target_zone_body_id]
        
        # 简单判断：方块在目标区域 xy 范围内，且高度较低（落地）
        dist_xy = np.linalg.norm(box_pos[:2] - target_pos[:2])
        is_close = dist_xy < 0.05
        is_grounded = box_pos[2] < 0.05 
        
        # 机械臂抬起 (可选)
        is_arm_lifted = data.site_xpos[self.attachment_site_id][2] > 0.2

        return is_close and is_grounded and is_arm_lifted