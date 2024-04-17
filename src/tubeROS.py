import rospy
import numpy as np
from geometry_msgs.msg import Twist, Point, PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion


import os
import sys
sys.path.append(os.path.dirname(__file__))
from Navigation.tubeMPPI import TubeMPPIRacetrack
from transformation_utils import get_matrix_pose_from_quat

class TubeMPPIROSNode:
    def __init__(self):
        rospy.init_node('tube_mppi_path_planner')
        self.path_planner = TubeMPPIRacetrack()
        # Initialize your path planner
        self.current_pose = np.zeros(3) # x, y, theta
        self.update_goal_ = False

        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        self.goal_subscriber = rospy.Subscriber(
            '/goal_update', 
            PoseStamped, 
            self.goal_update_callback
        )
        
        self.pose_subscriber = rospy.Subscriber(
            '/turtle_pose', 
            PoseStamped if rospy.get_param('~pose_stamped') else Pose,
            self.pose_callback
        )
        
        self.map_subscriber = rospy.Subscriber(
            '/occupancy_map', 
            OccupancyGrid, 
            self.map_callback
        )

    def goal_update_callback(self, msg):
        self.update_goal_ = True
        new_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.path_planner.update_goal(new_goal)
        rospy.loginfo(f"Updated goal to: {self.update_goal_} {new_goal}")

    def map_callback(self, msg):
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.path_planner.update_static_map(grid)

    def pose_callback(self, msg):
        if rospy.get_param('~pose_stamped'):
            msg = msg.pose
        self.current_pose = get_matrix_pose_from_quat(msg, return_matrix=False)
        self.current_pose = np.array([self.current_pose])
        self.plan_and_execute()

    def plan_and_execute(self):
        if not self.update_goal_:
            return
        rospy.loginfo(f"PLANNING AND EXECUTING")
        control_actions = self.path_planner.get_action(self.current_pose)
        rospy.loginfo("Desired Controls: {}".format(control_actions))
        twist_msg = Twist()
        twist_msg.linear.x  = control_actions[0][0]
        twist_msg.angular.z = control_actions[0][1]
        self.velocity_publisher.publish(twist_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = TubeMPPIROSNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo('Tube MPPI path planner node shut down.')
