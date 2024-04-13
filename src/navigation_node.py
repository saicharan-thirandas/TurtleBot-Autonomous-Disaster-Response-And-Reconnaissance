#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class SimplePathPlanner:
    def __init__(self, goal_x, goal_y, tolerance=0.1):
        rospy.init_node('simple_path_planner')

        # Publisher to send commands to the robot
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscriber to get the robot's odometry
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Goal and tolerance
        self.goal = np.array([goal_x, goal_y])
        self.tolerance = tolerance

        # Robot's current pose
        self.current_pose = Pose()

        # Rate
        self.rate = rospy.Rate(10)  # 10 Hz

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def get_heading_to_goal(self):
        current_orientation = self.current_pose.orientation
        current_position = np.array([self.current_pose.position.x, self.current_pose.position.y])
        
        # Convert quaternion to euler
        _, _, yaw = euler_from_quaternion([
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
            current_orientation.w
        ])

        # Calculate angle to the goal
        direction_to_goal = np.arctan2(self.goal[1] - current_position[1], self.goal[0] - current_position[0])
        angle_to_goal = direction_to_goal - yaw

        # Normalize angle to be between -pi to pi
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi

        return angle_to_goal, np.linalg.norm(self.goal - current_position)

    def run(self):
        while not rospy.is_shutdown():
            angle_to_goal, distance_to_goal = self.get_heading_to_goal()
            
            # Create Twist message
            command = Twist()

            if distance_to_goal > self.tolerance:
                # Rotate to face the goal
                if abs(angle_to_goal) > 0.1:
                    command.angular.z = 0.5 if angle_to_goal > 0 else -0.5
                # Move towards the goal
                else:
                    command.linear.x = 0.5

            self.cmd_pub.publish(command)
            self.rate.sleep()

if __name__ == '__main__':
    planner = SimplePathPlanner(goal_x=2.0, goal_y=2.0)
    planner.run()
