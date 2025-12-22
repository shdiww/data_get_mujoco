#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
左/右手臂单臂 MoveJ 控制客户端
通过 Uplimb_MoveJ 服务控制手臂 7 关节运动到指定位置
"""

import rospy
from navi_types.srv import Uplimb_MoveJ

def move_arm(arm_side, joint_angles, wait=True):
    """
    控制单个手臂运动到指定关节角度
    
    参数:
        arm_side (str): 'left' 或 'right' - 选择左臂或右臂
        joint_angles (list): 7个关节角度 [Shoulder_Y, Shoulder_X, Shoulder_Z, 
                                          Elbow, Wrist_Z, Wrist_Y, Wrist_X]
        wait (bool): True 表示等待执行完成,False 表示立即返回
    
    返回:
        bool: 服务是否执行成功
    """
    
    # 构造服务名称
    service_name = f"/{arm_side}_arm_movej_service"
    
    try:
        # 等待服务可用
        rospy.wait_for_service(service_name, timeout=5)
        
        # 创建服务代理
        movej_client = rospy.ServiceProxy(service_name, Uplimb_MoveJ)
        
        # 调用服务（使用位置参数，避免同时使用位置和关键字）
        # Uplimb_MoveJ 的字段为: jnt_angle, not_wait
        # 这里 `wait=True` 表示等待完成，所以传给服务的 `not_wait` 要取反
        response = movej_client(joint_angles, (not wait))
        
        rospy.loginfo(f"{arm_side.upper()} arm MoveJ service returned: {response.finish}")
        return response.finish.data
        
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False
    except rospy.ROSException as e:
        rospy.logerr(f"Service {service_name} not available: {e}")
        return False


def move_both_arms(left_angles, right_angles, wait=True):
    """
    同时控制左右手臂运动
    
    参数:
        left_angles (list): 左臂 7 个关节角度
        right_angles (list): 右臂 7 个关节角度
        wait (bool): 是否等待执行完成
    """
    
    rospy.loginfo("Moving both arms...")
    
    left_success = move_arm('left', left_angles, wait)
    right_success = move_arm('right', right_angles, wait)
    
    return left_success and right_success


def main():
    """
    示例：控制手臂运动
    """
    rospy.init_node('arm_movej_controller', anonymous=True)
    
    # 示例 1: 控制左臂到初始位置 (单位：弧度)
    print("\n=== 示例 1: 左臂运动到初始位置 ===")
    left_home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    success = move_arm('left', left_home, wait=True)
    if success:
        rospy.loginfo("Left arm reached home position")
    
    rospy.sleep(1)
    
    # 示例 2: 控制右臂到初始位置
    print("\n=== 示例 2: 右臂运动到初始位置 ===")
    right_home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    success = move_arm('right', right_home, wait=True)
    if success:
        rospy.loginfo("Right arm reached home position")
    
    rospy.sleep(1)
    
    # 示例 3: 控制左臂到自定义位置 (单位：弧度)
    print("\n=== 示例 3: 左臂运动到自定义位置 ===")
    left_custom = [0.5, 0.3, -0.2, 1.0, 0.5, 0.0, 0.0]  # 示例角度
    success = move_arm('left', left_custom, wait=True)
    if success:
        rospy.loginfo("Left arm reached custom position")
    
    rospy.sleep(1)
    
    # 示例 4: 同时控制左右臂
    print("\n=== 示例 4: 左右臂同时运动 ===")
    left_angles = [0.2, 0.1, -0.1, 0.5, 0.2, 0.0, 0.0]
    right_angles = [-0.2, -0.1, 0.1, 0.5, -0.2, 0.0, 0.0]
    success = move_both_arms(left_angles, right_angles, wait=True)
    if success:
        rospy.loginfo("Both arms moved successfully")
    
    rospy.loginfo("All tests completed")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
