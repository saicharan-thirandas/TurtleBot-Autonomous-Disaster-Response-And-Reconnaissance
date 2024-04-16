#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose 
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry, OccupancyGrid 
from std_msgs.msg import Bool
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import os
import sys
sys.path.append(os.path.dirname(__file__))
from transformation_utils import get_matrix_pose_from_quat, get_quat_pose, calculate_yaw
from mapping import Mapping


class GoalUpdater(Mapping):
    def __init__(self):
        super(GoalUpdater, self).__init__()
        rospy.init_node('update_goal', anonymous=True)
        self.reset_goal = False

        rospy.Subscriber(
            name=rospy.get_param('~occupancy_map_topic'),
            data_class=OccupancyGrid,
            callback=self.occupancy_grid_callback_lidar,
            queue_size=10
            )

        rospy.Subscriber(
            name=rospy.get_param('~occupancy_map_cam_topic'),
            data_class=OccupancyGrid,
            callback=self.occupancy_grid_callback_camera,
            queue_size=10
            )
        
        rospy.Subscriber(
            name=rospy.get_param('~pose_topic'),
            data_class=PoseStamped if rospy.get_param('~pose_stamped') else Pose,
            callback=self.turtle_pose_callback,
            queue_size=10
        )

        rospy.Subscriber(
            name=rospy.get_param('~goal_reset'),
            data_class=Bool,
            callback=self.goal_reset_callback,
            queue_size=10
        )

        self.goal_publisher = rospy.Publisher(
            rospy.get_param("~goal_update"), 
            PoseStamped if rospy.get_param('~pose_stamped') else Pose, 
            queue_size=10
        )
        
        self.goal_reset_publisher = rospy.Publisher(
            rospy.get_param("~goal_reset"), 
            Bool, 
            queue_size=10
        )
  
    def occupancy_grid_callback_lidar(self, data):
        # Extracting map data from OccupancyGrid message
        width = data.info.width
        height = data.info.height
        map_data = data.data

        map_array = np.array(map_data).reshape((height, width))
        image = cv2.normalize(map_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.lidar_map = image
    
    def occupancy_grid_callback_camera(self, data):
        # Extracting map data from OccupancyGrid message
        width = data.info.width
        height = data.info.height
        map_data = data.data

        map_array = np.array(map_data).reshape((height, width))
        image = cv2.normalize(map_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.camera_map = image
    
    def turtle_pose_callback(self, pose_msg):
        if rospy.get_param('~pose_stamped'):
            pose_msg = pose_msg.pose
        self.turtle_pose = get_matrix_pose_from_quat(pose_msg, return_matrix=False) # [x, y, yaw]

    def goal_reset_callback(self, msg):
        self.reset_goal = msg.data
        rospy.loginfo(f"Reset the goal: {self.reset_goal}")

    def filter_frontier(self , image):
        _, thresholded = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(thresholded)        
        return inverted
    
    def find_closest_pixel(self, current_position, diff_image):
        """ 
        Given the current position in the occupancy map and the 
        occupancy map corresponding to the ~camera_pov...
        """

        # In diff_image, the white pixels are the fronties outside the camera's pov
        non_black_pixels_indices = np.argwhere(diff_image > 10)
        if len(non_black_pixels_indices) == 0:
            return None
        
        # Randomly sample 10% of non-black pixels
        num_samples = max(1, len(non_black_pixels_indices) // 10)  # Ensure at least one sample
        sampled_indices = np.random.choice(len(non_black_pixels_indices), num_samples, replace=False)
        sampled_pixels = non_black_pixels_indices[sampled_indices]
        
        sampled_pixels = np.float32(sampled_pixels)
        distances = np.linalg.norm(sampled_pixels - current_position, axis=1)
        min_distance_index = np.argmin(distances)        
        closest_pixel = sampled_pixels[min_distance_index]
        
        return closest_pixel.astype(int)  # Return as integer coordinates
    
    def find_goal_position(self, current_position, lidar_map, camera_map):
        """ 
        Given the turtle's position in the occupancy map, occupancy map from the
        lidar and the occupancy from the camera's POV...
        """
        
        filtered_image1 = self.filter_frontier(lidar_map)
        filtered_image2 = self.filter_frontier(camera_map)
        
        diff_image = cv2.absdiff(filtered_image1, filtered_image2)
        diff_image = cv2.bitwise_xor(filtered_image1, filtered_image2)
        
        closest_pixel = self.find_closest_pixel(current_position, diff_image)
        new_target_position = closest_pixel if closest_pixel is not None else None

        return new_target_position, filtered_image1, filtered_image2, diff_image

    def create_pose_msg(x, y, yaw):
        return get_quat_pose(x, y, yaw, stamped=rospy.get_param('~pose_stamped'))
    
    def main(self):
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            if self.reset_goal and hasattr(self, 'turtle_pose'):
                turtle_pose = self.turtle_pose
                cor_x, cor_y, _  = super()._coords_to_grid_indicies(*turtle_pose, sign=-1)
                rospy.loginfo(f"Recieved a request to update the goal... current turtle coords: {turtle_pose}, grid: {cor_x, cor_x}")
                turtle_grid_pose = np.array([cor_x, cor_y])
                new_target_position, _, _, _ = self.find_goal_position(turtle_grid_pose, self.lidar_map, self.camera_map)

                if new_target_position is not None:
                    # Get new Pose
                    yaw  = calculate_yaw(new_target_position[0], new_target_position[1], *turtle_pose[:2])
                    new_target_position = super()._grid_indices_to_coords(*new_target_position, yaw, sign=1)
                    pose = get_quat_pose(new_target_position[0], new_target_position[1], yaw, stamped=rospy.get_param('~pose_stamped'))
                    rospy.loginfo(f"New pose: {pose}")
                    self.goal_publisher.publish(pose)
                    
                    # Do not reset goal
                    reset = Bool()
                    reset.data = False
                    self.goal_reset_publisher.publish(reset)

            rate.sleep()

if __name__ == '__main__':
    try:
        goal_updater=GoalUpdater()
        goal_updater.main()
        #data recieved is None
    except rospy.ROSInterruptException:
        pass
