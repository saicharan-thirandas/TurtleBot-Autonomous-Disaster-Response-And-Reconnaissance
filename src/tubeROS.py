import rospy
import numpy as np
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion


import os
import sys
sys.path.append(os.path.dirname(__file__))
from Navigation.tubeMPPI import TubeMPPIRacetrack

class TubeMPPIROSNode:
    def __init__(self):
        rospy.init_node('tube_mppi_path_planner')
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle_pose', PoseStamped, self.pose_callback)
        self.goal_subscriber = rospy.Subscriber('/goal_update', Point, self.goal_update_callback)
        self.map_subscriber = rospy.Subscriber('/occupancy_map', OccupancyGrid, self.map_callback)

        # Initialize your path planner
        self.path_planner = TubeMPPIRacetrack()
        self.current_pose = np.zeros(3)  # x, y, theta
        self.rate = rospy.Rate(10)  # Hz

    def map_callback(self, msg):
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.path_planner.update_static_map(grid)
        rospy.loginfo("Map updated in the path planner.")

    def pose_callback(self, msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_pose = np.array([position.x, position.y, yaw])
        self.plan_and_execute()

    def goal_update_callback(self, msg):
        new_goal = np.array([msg.x, msg.y])
        self.path_planner.update_goal(new_goal)
        rospy.loginfo("Updated goal to: {}".format(new_goal))

    def plan_and_execute(self):
        control_actions = self.path_planner.get_action(self.current_pose)
        twist_msg = Twist()
        twist_msg.linear.x = control_actions[0][0]
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
