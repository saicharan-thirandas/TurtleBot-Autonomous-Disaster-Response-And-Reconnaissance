import numpy as np
import rclpy
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge


class SensorFusion(Node):

    def __init__(self):

        super().__init__('sensor_fusion')
        self.bridge = CvBridge()

        self.intrinsics = np.array([[2.68e3,     0., 1.64e3], 
                                    [    0., 2.49e3, 1.21e3], 
                                    [    0.,     0.,     1.]])
        
        self.rectification_mat = np.array([[1, 0, 0], 
                                           [0, 1, 0], 
                                           [0, 0, 1]])
        
        self.lidar_to_camera = np.array([[1, 0, 0,   0],  # TODO: Measure this properly
                                         [0, 1, 0, -10], 
                                         [0, 0, 1,   0]])
        
        self.transformation = self.intrinsics @ self.rectification_mat @ self.lidar_to_camera
        # uv = self.transformation @ [n, 3]

        # Based on RPI camera parameters
        self.theta_min = 62.2
        self.theta_max = 121.1
    
    def fusion(self):
        pass