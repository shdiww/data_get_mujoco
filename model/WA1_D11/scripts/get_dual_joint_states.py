#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import JointState

# 回调函数处理接收到的消息
def joint_state_callback(msg):
    # 获取所有关节的位置
    joint_positions = msg.position

    # 假设关节顺序是：左臂7个，右臂7个，脖子2个，腰部1个
    left_arm = joint_positions[0:7]  # 左臂7个关节
    right_arm = joint_positions[7:14]  # 右臂7个关节
    neck = joint_positions[14:16]  # 脖子的2个关节
    waist = joint_positions[16]  # 腰部的1个关节

    # 打印关节位置
    rospy.loginfo("Left Arm Joint Positions: %s", left_arm)
    rospy.loginfo("Right Arm Joint Positions: %s", right_arm)
    rospy.loginfo("Neck Joint Positions: %s", neck)
    rospy.loginfo("Waist Joint Position: %s", waist)

def listener():
    # 初始化 ROS 节点
    rospy.init_node('joint_state_listener', anonymous=True)

    # 订阅 /joint_states 话题
    rospy.Subscriber("/joint_states", JointState, joint_state_callback)

    # 保持节点运行
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
