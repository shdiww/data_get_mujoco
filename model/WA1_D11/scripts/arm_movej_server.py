#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时服务端：把 /left_arm_movej_service 和 /right_arm_movej_service
请求转为 FollowJointTrajectory 发送到 /left_arm_controller 和 /right_arm_controller。

服务类型: navi_types/Uplimb_MoveJ
请求: float32[] jnt_angle (7), bool not_wait
响应: std_msgs/Bool finish

使用方法（在已经启动 roscore 和机器人控制器/仿真之后）:
$ chmod +x scripts/arm_movej_server.py
$ rosrun WA1_D11 arm_movej_server.py

注意：该脚本只是一个桥接，如果控制器/action server 不存在，
会记录错误并返回 finish=False。
"""

from __future__ import print_function
import rospy
import actionlib
from navi_types.srv import Uplimb_MoveJ, Uplimb_MoveJResponse
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from gazebo_msgs.srv import SetModelConfiguration, SetModelConfigurationRequest


LEFT_JOINT_NAMES = [
    "Shoulder_Y_L",
    "Shoulder_X_L",
    "Shoulder_Z_L",
    "Elbow_L",
    "Wrist_Z_L",
    "Wrist_Y_L",
    "Wrist_X_L",
]

RIGHT_JOINT_NAMES = [
    "Shoulder_Y_R",
    "Shoulder_X_R",
    "Shoulder_Z_R",
    "Elbow_R",
    "Wrist_Z_R",
    "Wrist_Y_R",
    "Wrist_X_R",
]


class MoveJServer(object):
    def __init__(self):
        rospy.init_node('arm_movej_server', anonymous=False)

        # action clients
        self.left_ac = actionlib.SimpleActionClient('/left_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.right_ac = actionlib.SimpleActionClient('/right_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        rospy.loginfo('Waiting for action servers (5s)...')
        left_ok = self.left_ac.wait_for_server(rospy.Duration(5.0))
        right_ok = self.right_ac.wait_for_server(rospy.Duration(5.0))
        if not left_ok or not right_ok:
            rospy.logwarn('One or both FollowJointTrajectory action servers not available; will fall back to Gazebo SetModelConfiguration if needed.')

        # services
        self.left_srv = rospy.Service('/left_arm_movej_service', Uplimb_MoveJ, self.left_cb)
        self.right_srv = rospy.Service('/right_arm_movej_service', Uplimb_MoveJ, self.right_cb)

        rospy.loginfo('MoveJ service servers started: /left_arm_movej_service, /right_arm_movej_service')

    def _send_trajectory(self, ac, joint_names, angles, not_wait):
        # angles: list of 7 floats
        if len(angles) != len(joint_names):
            rospy.logerr('angles length %d != joint_names length %d', len(angles), len(joint_names))
            return False

        traj = JointTrajectory()
        traj.joint_names = joint_names
        pt = JointTrajectoryPoint()
        pt.positions = angles
        pt.velocities = [0.0] * len(angles)
        pt.time_from_start = rospy.Duration(1.5)
        traj.points = [pt]

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj

        # 如果 action server 可用，优先使用 action
        try:
            # check whether action server is available immediately
            if ac.wait_for_server(rospy.Duration(0.0)):
                ac.send_goal(goal)
                if not not_wait:
                    finished = ac.wait_for_result(rospy.Duration(10.0))
                    if not finished:
                        rospy.logwarn('Action did not finish in time')
                        return False
                    result = ac.get_result()
                    rospy.loginfo('Action result: %s', str(result))
                else:
                    rospy.loginfo('Sent goal (not waiting)')
                return True
            else:
                # 回退：如果是 Gazebo 仿真，直接调用 /gazebo/set_model_configuration 设置模型关节（即时生效）
                try:
                    rospy.wait_for_service('/gazebo/set_model_configuration', timeout=2.0)
                    set_cfg = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
                    req = SetModelConfigurationRequest()
                    req.model_name = 'WA1_D11'
                    # use the robot_description param where the URDF was loaded
                    req.urdf_param_name = 'robot_description'
                    req.joint_names = joint_names
                    req.joint_positions = angles
                    set_cfg(req)
                    rospy.loginfo('Gazebo set_model_configuration called for %s', req.model_name)
                    return True
                except Exception as e2:
                    rospy.logerr('Fallback to Gazebo set_model_configuration failed: %s', e2)
                    return False
        except Exception as e:
            rospy.logerr('Failed to send goal: %s', e)
            return False

    def left_cb(self, req):
        rospy.loginfo('Received left arm MoveJ request: %s not_wait=%s', req.jnt_angle, req.not_wait)
        ok = self._send_trajectory(self.left_ac, LEFT_JOINT_NAMES, list(req.jnt_angle), req.not_wait)
        return Uplimb_MoveJResponse(finish=Bool(data=ok))

    def right_cb(self, req):
        rospy.loginfo('Received right arm MoveJ request: %s not_wait=%s', req.jnt_angle, req.not_wait)
        ok = self._send_trajectory(self.right_ac, RIGHT_JOINT_NAMES, list(req.jnt_angle), req.not_wait)
        return Uplimb_MoveJResponse(finish=Bool(data=ok))


if __name__ == '__main__':
    try:
        server = MoveJServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
