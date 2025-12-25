import mujoco.viewer
 
def main():
    xml_path ="/home/blzgz/data_get_mujoco/model/franka_emika_panda/mjx_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    # model = mujoco.MjModel.from_xml_path('/home/blzgz/mujoco_diffusion/franka_ros/franka_description/robots/panda_arm_hand.urdf.xacro')
    data = mujoco.MjData(model)
 
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
 
if __name__ == "__main__":
    main()