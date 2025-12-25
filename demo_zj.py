import mujoco as mj
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import time
import os

# 加载模型
model = mj.MjModel.from_xml_path("/home/blzgz/data_get_mujoco/model/WA1_D11/urdf/scene.xml")  # 替换为你的模型文件路径
data = mj.MjData(model)


# 启动被动查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
        
    # 调整相机视角
    viewer.cam.lookat[:] = [0, 0, 0.5]
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20

    # 设置初始位置
    key_name = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    init_qpos = model.key_qpos[key_name]
    data.qpos = init_qpos
    mujoco.mj_forward(model, data)
    
    # 仿真循环
    t = 0.0
    dt = 0.01
    while viewer.is_running():
        data.qpos = init_qpos
        data.qpos[4] = init_qpos[4] + 0.05*np.sin(2*np.pi*t)
        data.qvel = np.zeros_like(data.qvel)
        
        # 物理仿真步进
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
        t += dt
