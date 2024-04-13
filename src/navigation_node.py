#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from collections import defaultdict

class AstarPathPlanner:
    def __init__(self, goal_x, goal_y, tolerance=0.1):
        rospy.init_node('astar_path_planner')

        # Publisher to send commands to the robot
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscribers to get the robot's odometry, laser scan, and occupancy map
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.map_sub = rospy.Subscriber('/occupancy_map', OccupancyGrid, self.map_callback)
        
        # Goal and tolerance
        self.goal = np.array([goal_x, goal_y])
        self.tolerance = tolerance

        # Robot's current pose and occupancy map
        self.current_pose = PoseStamped()
        self.occupancy_map = None
        self.laser_scan = None

        # Rate
        self.rate = rospy.Rate(10)  # 10 Hz

        # Path planning storage
        self.open_set = set()
        self.came_from = {}
        self.g_score = defaultdict(lambda: float('inf'))
        self.f_score = defaultdict(lambda: float('inf'))

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def map_callback(self, msg):
        self.occupancy_map = msg

    def scan_callback(self, msg):
        self.laser_scan = msg

    def heuristic(self, node_a, node_b):
        return np.linalg.norm(np.array(node_a) - np.array(node_b))

    def get_neighbors(self, node):
        if not self.occupancy_map:
            return []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        neighbors = []
        node_x, node_y = int(node[0]), int(node[1])
        for dx, dy in directions:
            neighbor = (node_x + dx, node_y + dy)
            if 0 <= neighbor[0] < self.occupancy_map.info.width and 0 <= neighbor[1] < self.occupancy_map.info.height:
                index = neighbor[1] * self.occupancy_map.info.width + neighbor[0]
                if self.occupancy_map.data[index] == 0:
                    neighbors.append(neighbor)
        return neighbors

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def plan_path(self):
        start_node = (self.current_pose.pose.position.x, self.current_pose.pose.position.y)
        goal_node = (self.goal[0], self.goal[1])
        self.open_set.add(start_node)
        self.g_score[start_node] = 0
        self.f_score[start_node] = self.heuristic(start_node, goal_node)

        while self.open_set:
            current = min(self.open_set, key=lambda node: self.f_score[node])
            if current == goal_node:
                return self.reconstruct_path(self.came_from, current)
            self.open_set.remove(current)
            for neighbor in self.get_neighbors(current):
                tentative_g_score = self.g_score[current] + self.distance(current, neighbor)
                if tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)
                    if neighbor not in self.open_set:
                        self.open_set.add(neighbor)
        return None  # No path found if we reach this point

    def avoid_obstacles(self):
        if not self.laser_scan:
            return Twist()  # If no scan data is available, return an empty twist

        command = Twist()
        MIN_DISTANCE = 0.5  # meters
        if min(self.laser_scan.ranges) < MIN_DISTANCE:
            command.angular.z = 1.0  # Turn in place
        else:
            command.linear.x = 0.5  # Move forward
        return command

    def run(self):
        while not rospy.is_shutdown():
            if self.laser_scan and self.occupancy_map:
                command = self.avoid_obstacles()  # Modify the control based on obstacles
                self.cmd_pub.publish(command)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        planner = AstarPathPlanner(goal_x=2.0, goal_y=2.0)
        planner.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation node shut down.")
