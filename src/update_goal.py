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
        cv2.namedWindow("Frontiers")

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
        self.lidar_map = cv2.normalize(map_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    def occupancy_grid_callback_camera(self, data):
        # Extracting map data from OccupancyGrid message
        width = data.info.width
        height = data.info.height
        map_data = data.data

        map_array = np.array(map_data).reshape((height, width))
        self.camera_map = cv2.normalize(map_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    def turtle_pose_callback(self, pose_msg):
        if rospy.get_param('~pose_stamped'):
            pose_msg = pose_msg.pose
        self.turtle_pose = get_matrix_pose_from_quat(pose_msg, return_matrix=False) # [x, y, yaw]

    def goal_reset_callback(self, msg):
        self.reset_goal = msg.data

    def filter_frontier(self, image):
        _, img_thesholded = cv2.threshold(image, thresh=180, maxval=255, type=cv2.THRESH_BINARY_INV)
        return img_thesholded
    
    def find_next_pixel(self, current_position, diff_image, lidar_map):
        """ 
        Given the current position in the occupancy map and the 
        occupancy map corresponding to the ~camera_pov...
        """

        # In diff_image, the white pixels are the walls outside the camera's pov
        white_pixels_indices = np.argwhere(diff_image > 200)
        if len(white_pixels_indices) == 0:
            return None
        
        # Randomly sample n% of white pixels
        num_samples = max(1, int(len(white_pixels_indices) // 20))  # Ensure at least one sample
        sampled_indices = np.random.choice(len(white_pixels_indices), num_samples, replace=False)
        sampled_pixels = white_pixels_indices[sampled_indices].astype(np.float32)

        distances = np.linalg.norm(sampled_pixels - current_position, axis=1)
        index = np.argsort(distances)[int(len(distances) // 2)]
        # index = np.argmin(distances)
        closest_pixel = sampled_pixels[index]
        
        return closest_pixel.astype(int), sampled_pixels.astype(int)  # Return as integer coordinates

    def find_goal_position(self, current_position):
        """ 
        Given the turtle's position in the occupancy map, occupancy map from the
        lidar and the occupancy from the camera's POV...
        """
        
        filtered_image1 = self.filter_frontier(self.lidar_map)
        filtered_image2 = self.filter_frontier(self.camera_map)
        diff_image = cv2.bitwise_xor(filtered_image1, filtered_image2)
        
        closest_pixel, sampled_pixels = self.find_next_pixel(current_position, diff_image, filtered_image1)
        new_target_position = closest_pixel if closest_pixel is not None else None

        return new_target_position, filtered_image1, filtered_image2, diff_image, sampled_pixels

    def create_pose_msg(x, y, yaw):
        return get_quat_pose(x, y, yaw, stamped=rospy.get_param('~pose_stamped'))
    
    def display_pose_and_goal(self, filtered_image1, filtered_image2, diff_image, image1_with_positions, current_position, new_target_position, sampled_pixels):
        
        image1_with_positions = cv2.cvtColor(image1_with_positions, cv2.COLOR_GRAY2RGB).astype(float)
        filtered_image1 = cv2.cvtColor(filtered_image1, cv2.COLOR_GRAY2RGB).astype(float)
        filtered_image2 = cv2.cvtColor(filtered_image2, cv2.COLOR_GRAY2RGB).astype(float)
        diff_image = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2RGB).astype(float)

        image1_with_positions = cv2.circle(image1_with_positions, (current_position[1], current_position[0]), 3, (0, 0, 255), -1)  # Red circle for current position
        for sampled_grid_ids in sampled_pixels:
            image1_with_positions = cv2.circle(image1_with_positions, (sampled_grid_ids[1], sampled_grid_ids[0]), 3, (255, 0, 0), -1)  # Blue circle for all sampled frontiers
        if new_target_position is not None:
            image1_with_positions = cv2.circle(image1_with_positions, (new_target_position[1], new_target_position[0]), 3, (0, 255, 0), -1)  # Green circle for final position
        
        # cv2.imshow("Frontiers", image1_with_positions) @SAI, you can display just the frontires images, or all
        cv2.imshow("Frontiers", np.hstack([filtered_image1, filtered_image2, diff_image, image1_with_positions]))
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
        if key == ord('p'):
            cv2.waitKey(-1)
    
    def main(self):
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            if self.reset_goal and hasattr(self, 'turtle_pose'):
                turtle_pose = self.turtle_pose
                cor_x, cor_y, _  = super()._world_coordinates_to_map_indices(turtle_pose[:2])
                
                rospy.loginfo(f"Recieved a request to update the goal... current turtle pose: {turtle_pose}, grid coords: {cor_x, cor_y}")
                turtle_grid_pose = np.array([cor_x, cor_y])
                new_target_position, _f1, _f2, _d, sampled_pixels = self.find_goal_position(turtle_grid_pose)

                if new_target_position is not None:
                    # Get new Pose
                    self.display_pose_and_goal(_f1, _f2, _d, self.lidar_map, turtle_grid_pose, new_target_position, sampled_pixels)
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
