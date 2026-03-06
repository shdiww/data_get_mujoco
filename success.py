import mujoco
import numpy as np


class TaskSuccessEvaluator:
    """任务成功条件定义与判定。"""

    def __init__(
        self,
        model,
        box_body_name="red_box",
        target_zone_body_name="target_zone",
        target_zone_geom_name="target_zone_geom",
        attachment_site_name="attachment_site",
        ground_tolerance=0.005,
        arm_lift_height=0.2,
        overlap_epsilon=1e-5,
    ):
        self.box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_body_name)
        self.target_zone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_zone_body_name)
        self.target_zone_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_zone_geom_name)
        self.attachment_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, attachment_site_name)

        self.ground_tolerance = ground_tolerance
        self.arm_lift_height = arm_lift_height
        self.overlap_epsilon = overlap_epsilon

    def is_success(self, model, data):
        if self.box_id == -1 or self.target_zone_body_id == -1:
            return False

        geom_id = model.body_geomadr[self.box_id]
        box_half_size = model.geom_size[geom_id]
        box_pos = data.xpos[self.box_id]
        box_mat = data.xmat[self.box_id].reshape(3, 3)

        # 计算方块在 XY 平面的 AABB
        aabb_half_size = np.abs(box_mat).dot(box_half_size)
        min_b = box_pos[:2] - aabb_half_size[:2]
        max_b = box_pos[:2] + aabb_half_size[:2]

        # 目标区域边界
        target_zone_pos = data.xpos[self.target_zone_body_id]
        target_zone_size = model.geom_size[self.target_zone_geom_id]
        min_t = target_zone_pos[:2] - target_zone_size[:2]
        max_t = target_zone_pos[:2] + target_zone_size[:2]

        overlap_min = np.maximum(min_b, min_t)
        overlap_max = np.minimum(max_b, max_t)
        overlap_dims = np.maximum(0, overlap_max - overlap_min)
        overlap_area = overlap_dims[0] * overlap_dims[1]
        box_area = (max_b[0] - min_b[0]) * (max_b[1] - min_b[1])

        # 成功条件
        is_on_ground = box_pos[2] <= box_half_size[2] + self.ground_tolerance
        is_arm_lifted = data.site_xpos[self.attachment_site_id][2] > self.arm_lift_height
        is_overlap_ok = overlap_area >= box_area - self.overlap_epsilon

        return is_on_ground and is_overlap_ok and is_arm_lifted
