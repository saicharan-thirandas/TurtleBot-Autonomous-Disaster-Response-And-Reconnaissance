#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped , Pose 
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry, OccupancyGrid 
from std_msgs.msg import Bool
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import os
import sys
sys.path.append(os.path.dirname(__file__))
from transformation_utils import get_matrix_pose_from_quat , get_quat_pose , calculate_yaw
from mapping import Mapping


class GoalUpdater:
    def __init__(self):
        rospy.init_node('goal_updater')
        # Publisher to send commands to the robo                # Subscribe to lidar map topic
        self.reset_goal =False
        rospy.Subscriber(
                name='/occupancy_map',
                data_class=OccupancyGrid,
                callback=self.occupancy_grid_callback_lidar,
                queue_size=10
            )

        rospy.Subscriber(
                name='/occupancy_map_camera',
                data_class=OccupancyGrid,
                callback=self.occupancy_grid_callback_camera,
                queue_size=10
            )
        
        rospy.Subscriber(
            name='/turtle_pose',
            data_class=Pose,
            callback=self.turtle_pose_callback,
            queue_size=10
        )

        rospy.Subscriber(
            name='/update_goal/reset',
            data_class=Bool,
            callback=self.goal_reset_callback,
            queue_size=10
        )

    def occupancy_grid_callback_lidar(self,data):
        # Extracting map data from OccupancyGrid message
        width = data.info.width
        height = data.info.height
        resolution = data.info.resolution
        map_data = data.data

        # Reshape map data into 2D array
        # Reshape map data into 2D array
        map_array = np.array(map_data).reshape((height, width))

        # Create an image using OpenCV
        # Map the values from map_array directly to the image
        # Normalize the values to fit in the range [0, 255]
        image = cv2.normalize(map_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


        # Optionally, you can resize the image for better visualization
        # image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        # Display or save the image
        self.lidar_map = image
        cv2.imshow("Occupancy Grid", image)
        cv2.waitKey(0) 
    
    def occupancy_grid_callback_camera(self,data):
        # Extracting map data from OccupancyGrid message
        width = data.info.width
        height = data.info.height
        resolution = data.info.resolution
        map_data = data.data

        map_array = np.array(map_data).reshape((height, width))

        image = cv2.normalize(map_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        self.camera_map = image
        cv2.imshow("Occupancy Grid", image)
        cv2.waitKey(0) 
    
    #TODO - Lidar to recheck the function.
    def turtle_pose_callback(self, pose_msg):
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y

        rot = R.from_quat([
            pose_msg.pose.pose.orientation.x,
            pose_msg.pose.pose.orientation.y,
            pose_msg.pose.pose.orientation.z,
            pose_msg.pose.pose.orientation.w
        ])

        _, _, yaw = rot.as_euler('xyz', degrees=False)
        self.turtle_pose = [x, y, yaw]

    def goal_reset_callback(self, msg):
        self.reset_goal = msg


    def filter_frontier(self , image):
        # Threshold the grayscale image to find black regions
        _, thresholded = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        print(str(image))
        
        # Invert the thresholded image
        inverted = cv2.bitwise_not(thresholded)
        
        return inverted
    
    
    def find_closest_pixel(self,current_position, image):
        # Find all non-black pixels (pixel value > 10) in the image
        non_black_pixels_indices = np.argwhere(image > 10)
        
        if len(non_black_pixels_indices) == 0:
            # If there are no non-black pixels, return None
            return None
        
        print("total pixels" +str(non_black_pixels_indices))
        # Randomly sample 10% of non-black pixels
        num_samples = max(1, len(non_black_pixels_indices) // 10)  # Ensure at least one sample
        sampled_indices = np.random.choice(len(non_black_pixels_indices), num_samples, replace=False)
        sampled_pixels = non_black_pixels_indices[sampled_indices]
        
        # Reshape sampled pixels to fit the k-means input format
        sampled_pixels = np.float32(sampled_pixels)
        
        # Perform pairwise distance calculation
        distances = np.linalg.norm(sampled_pixels - current_position, axis=1)
        
        # Find the index of the minimum distance
        min_distance_index = np.argmin(distances)
        
        # Get the closest pixel coordinates
        closest_pixel = sampled_pixels[min_distance_index]
        
        return closest_pixel.astype(int)  # Return as integer coordinates
    
    def find_goal_position(self,current_position, image1, image2):
        # Filter out black values in both images
        filtered_image1 = self.filter_frontier(image1)
        filtered_image2 = self.filter_frontier(image2)
        print("value "+str(filtered_image1))
        
        # Calculate the absolute difference between filtered images
        diff_image = cv2.absdiff(filtered_image1, filtered_image2)
        diff_image = cv2.bitwise_xor(filtered_image1, filtered_image2)
        #print

        
        # Find the closest pixel in the difference image
        closest_pixel = self.find_closest_pixel(current_position, diff_image)
        
        if closest_pixel is not None:
            # Set the closest non-black pixel as the new target position
            new_target_position = closest_pixel
        else:
            # If no non-black pixels found, set new_target_position to None
            new_target_position = None
        
        return new_target_position, filtered_image1, filtered_image2 ,diff_image


    def create_pose_msg(x, y, yaw):
        pose_msg = Pose()
        pose_msg.position.x = x
        pose_msg.position.y = y
        pose_msg.position.z = 0.0  # Assuming z-coordinate is 0 for 2D pose
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = np.sin(yaw / 2)  # Convert yaw to quaternion
        pose_msg.orientation.w = np.cos(yaw / 2)  # Convert yaw to quaternion
        return pose_msg
    
    def main(self):


        # Publisher for updated goal
        goal_publisher = rospy.Publisher("/update_goal/current_goal", PoseStamped, queue_size=10)
        goal_reset_publisher = rospy.Publisher("/update_goal/reset", PoseStamped, queue_size=10)

        # Create a CvBridge
        lidar_map_cv2=self.lidar_map
        camera_map_cv2=self.camera_map

        rate = rospy.Rate(1)  # 10Hz loop rate
        turtle_pose=self.turtle_pose

        while not rospy.is_shutdown():
            if lidar_map_cv2 is not None and camera_map_cv2 is not None and turtle_pose is not None and self.reset_goal is True:
                try:
                    cor_x, cor_y = Mapping._coords_to_grid_indicies(turtle_pose.pose.position.x, turtle_pose.pose.position.y, sign=-1)
                    turtle_position = np.array([cor_x, cor_y])

                    new_target_position, _, _, _ = self.find_goal_position(turtle_position, lidar_map_cv2, camera_map_cv2)

                    # Publish updated goal
                    if new_target_position is not None:
                        #new_goal_msg = PoseStamped()
                        #new_goal_msg.pose.position.x = new_target_position[0]
                        #new_goal_msg.pose.position.y = new_target_position[1]
                        yaw = calculate_yaw(new_target_position[0], new_target_position[1], turtle_pose.pose.position.x, turtle_pose.pose.position.y)
                        new_goal_pose_msg= get_quat_pose(new_target_position[0],new_target_position[1],yaw)
                        rospy.loginfo(" upating new goal info " + str(new_goal_pose_msg))

                        goal_publisher.publish(new_goal_pose_msg)
                        goal_reset_publisher.publish(False)

                except CvBridgeError as e:
                    rospy.logerr(e)

            rate.sleep()

if __name__ == '__main__':
    try:
        goal_updater=GoalUpdater()
        goal_updater.main()
    except rospy.ROSInterruptException:
        pass
