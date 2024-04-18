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
import matplotlib.pyplot as plt


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

    def filter_frontier(self , image):
        _, thresholded = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(thresholded)        
        return inverted
    
    def find_closest_pixel(self, current_position, diff_image, lidar_map):
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
    
    def find_closest_pixel_(self, current_position, diff_image, lidar_map):
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

        # Find the closest pixel not present in lidar_map
        min_distance = np.inf
        closest_pixel = None
        for i, pixel in enumerate(sampled_pixels):
            if tuple(pixel) not in lidar_map:  # Check if pixel is not present in lidar_map
                if distances[i] < min_distance:
                    min_distance = distances[i]
                    closest_pixel = pixel
        
        if closest_pixel is None:
            return None  # Return None if all sampled pixels are present in lidar_map

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
        
        closest_pixel = self.find_closest_pixel(current_position, diff_image, filtered_image1)
        new_target_position = closest_pixel if closest_pixel is not None else None

        return new_target_position, filtered_image1, filtered_image2, diff_image

    def create_pose_msg(x, y, yaw):
        return get_quat_pose(x, y, yaw, stamped=rospy.get_param('~pose_stamped'))
    
    def display_pose_and_goal(self,filtered_image1,filtered_image2,diff_image , image1_with_positions , current_position,new_target_position):
        
        cv2.circle(image1_with_positions, (current_position[1], current_position[0]), 5, (0, 0, 255), -1)  # Red circle for current position
        if new_target_position is not None:
            cv2.circle(image1_with_positions, (new_target_position[1], new_target_position[0]), 5, (0, 255, 0), -1)  # Green circle for final position
        
        # Display both filtered images side by side
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(filtered_image1, cmap='gray')
        axes[0].set_title("Current Lidar Map")
        axes[0].axis('off')
        axes[1].imshow(filtered_image2, cmap='gray')
        axes[1].set_title("Frontier seen by Camera")
        axes[1].axis('off')
        axes[2].imshow(diff_image, cmap='gray')
        axes[2].set_title("Frontier to explore")
        axes[2].axis('off')
        axes[3].imshow(cv2.cvtColor(image1_with_positions, cv2.COLOR_BGR2RGB))
        axes[3].set_title("Current - Red vs Goal - Green ")
        axes[3].axis('off')
        plt.show()
    

    def main(self):
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            if self.reset_goal and hasattr(self, 'turtle_pose'):
                turtle_pose = self.turtle_pose
                cor_x, cor_y, _  = super()._world_coordinates_to_map_indices(turtle_pose[:2])
                rospy.loginfo(f"Recieved a request to update the goal... current turtle pose: {turtle_pose}, grid coords: {cor_x, cor_y}")
                turtle_grid_pose = np.array([cor_x, cor_y])
                new_target_position, _f1, _f2, _d  = self.find_goal_position(turtle_grid_pose, self.lidar_map, self.camera_map)
                #_f1, _f2, _d = self.lidar_map, self.camera_map ,self.lidar_map
                if new_target_position is not None:
                    # Get new Pose
                    self.display_pose_and_goal(_f1, _f2, _d ,  self.lidar_map , turtle_grid_pose,new_target_position)
                    yaw  = calculate_yaw(new_target_position[0], new_target_position[1], *turtle_pose[:2])
                    new_target_coords = super()._grid_indices_to_coords(*new_target_position, yaw, sign=-1)
                    pose = get_quat_pose(new_target_coords[0], new_target_coords[1], yaw, stamped=rospy.get_param('~pose_stamped'))
                    rospy.loginfo(f"NEW TARGET POSE: {new_target_coords}, NEW TARGET COORDS: {new_target_position}")
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
    except rospy.ROSInterruptException:
        pass
