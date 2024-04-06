#!/usr/bin/env python

import gtsam
import numpy as np
from dataclasses import dataclass
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from apriltag_ros.msg import AprilTagDetectionArray


@dataclass
class Grid:
    grid_resolution = 0.1 # meters per grid cell
    grid_size_x = 100  # number of grid cells in the x direction
    grid_size_y = 100  # number of grid cells in the y direction
    grid_origin_x = -5.0  # origin of the grid in the x direction
    grid_origin_y = -5.0  # origin of the grid in the y direction


class Mapping(Grid):
    def __init__(self, 
                 p_free: float=0.2, 
                 p_occ: float=0.8, 
                 p_prior: float=0.5):
        
        # Initialize occupancy grid with p_prior
        self.occupancy_grid = np.zeros((
            self.grid_size_x, 
            self.grid_size_y
            ))
        
        self.log_odds_free  = self._prob_to_log_odds(p_free)
        self.log_odds_occ   = self._prob_to_log_odds(p_occ)
        self.log_odds_prior = self._prob_to_log_odds(p_prior)
    
    def _log_odds_to_prob(self, log_odds):
        return 1 - 1 / (1 + np.exp(log_odds))

    def _prob_to_log_odds(self, prob):
        return np.log(prob / (1 - prob))


class Lidar(Mapping):

    def __init__(self):

        super(Lidar, self).__init__()
        self.lidar_sub = rospy.Subscriber(
            name='/scan', 
            data_class=LaserScan, 
            callback=self.update_map
        )

        self.lidar_sub = rospy.Subscriber(
            name='/odom', 
            data_class=Odometry, 
            callback=self.update_odom
        )

        # Publish processed occupancy map
        self.occ_map_pub = rospy.Publisher(
            name='/occupancy_map',
            dataclass=OccupancyGrid,
            queue_size=10
        )
        self.odom = [0., 0., 0.]

    def get_probability_map(self):
        return super()._log_odds_to_prob(self.occupancy_grid)

    def _get_free_grids_from_beam(self, obs_pt_inds, hit_pt_inds):
        diff = hit_pt_inds - obs_pt_inds
        j = np.argmax( np.abs(diff) )
        D = np.abs( diff[j] )
        return obs_pt_inds + ( np.outer( np.arange(D + 1), diff ) + (D // 2) ) // D

    def _coords_to_grid_indicies(self, x, y, w):
        grid_x = int((x - self.grid_origin_x) / self.grid_resolution)
        grid_y = int((y - self.grid_origin_y) / self.grid_resolution)
        grid_w = int(w / self.grid_resolution)
        return np.array([grid_x, grid_y, grid_w])
    
    def update_odom(self, odom_msg):
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        w = odom_msg.pose.pose.orientation # TODO: Get heading angle using odom.txt as example
        self.odom = [x, y, w]

    def update_map(self, lidar_msg):

        ranges = np.asarray(lidar_msg.ranges)
        x, y, w = self.odom
        obs_pt_inds = self._coords_to_grid_indicies(x, y, w)

        for i in range(len(ranges)):
            # Get angle of range
            angle_rad = i * lidar_msg.angle_increment

            if ranges[i] <= lidar_msg.range_min or ranges[i] == np.inf:
                continue

            # Get x,y position of laser beam in the map
            beam_angle_rad = w + angle_rad
            hit_x = x + np.cos(beam_angle_rad) * ranges[i]
            hit_y = y + np.sin(beam_angle_rad) * ranges[i]

            hit_pt_inds  = self._coords_to_grid_indicies(hit_x, hit_y, beam_angle_rad)
            self.occupancy_grid[hit_pt_inds[0], hit_pt_inds[1]] = self.occupancy_grid[hit_pt_inds[0], hit_pt_inds[1]] + self.log_odds_occ - self.log_odds_prior

            free_pt_inds = self._get_free_grids_from_beam(obs_pt_inds[:2], hit_pt_inds[:2])
            self.occupancy_grid[free_pt_inds[:, 0], free_pt_inds[:, 1]] = self.occupancy_grid[free_pt_inds[:, 0], free_pt_inds[:, 1]] + self.log_odds_free - self.log_odds_prior

        self.publish()

    def publish(self):

        # TODO: Figure out attributes to pubish map
         # Conver the map to a 1D array
        self.map_update = OccupancyGrid()
        self.map_update.header.frame_id = "occupancy_grid"
        self.map_update.header.stamp = rospy.Time.now()
        # self.map_update.info = self.map_laser.info
        # self.map_update.data = self.laser_arr.flatten('F')
        # # Convert the map to type int8
        # self.map_update.data = self.map_update.data.astype(np.int8)
        # # Publish the map
        # self.pub.publish(self.map_update)


class GTSAM(Lidar):
    """ Encapsulates functions for GTSAM. """

    def __init__(self):

        rospy.init_node('slam_node', anonymous=True)
        super(GTSAM, self).__init__()

        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1])
        )

        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1])
        )

        self.tag_sub = rospy.Subscriber(
            name='/tag_detections', 
            data_class=AprilTagDetectionArray, 
            callback=self.add_tag_factors
        )

        # Publish predicted pose
        self.pose_pub = rospy.Publisher(
            name='/turtle_pose',
            dataclass=None, # TODO: Figure out dataclass to use for pose
            queue_size=10
        )

        # Define occupancy grid factor noise model
        self.occupancy_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]))

        # Create a factor graph
        self.graph = gtsam.NonlinearFactorGraph()

        # Create an initial estimate for the robot's pose (e.g., based on odometry)
        self.initial_estimate = gtsam.Values()
        self.current_pose = 0

        self._initialize_graph()

    def _initialize_graph(self):

        assert self.current_pose == 0
        priorMean   = gtsam.Pose2(0.0, 0.0, 0.0)
        priorFactor = gtsam.PriorFactorPose2(X(self.current_pose), priorMean, self.prior_noise) # TODO: Use appropriate shorthand variables 
        self.graph.add(priorFactor)
        self.initial_estimate.insert(X(self.current_pose), priorMean) # TODO: Use appropriate shorthand variables
        # self.current_pose += 1 # Publish should take care of this += 1

    def add_tag_factors(self, apriltag_msg):
        
        # TODO: Process AprilTag detections/tracking and add observation factors to the factor graph
        # Should this be from the original poses (T_CA -> T_OA via T_OA = T_OR @ T_RC @ T_CA) or from the most recent output of april_tag_tracker.py?
        for detection in apriltag_msg.detections:
            tag_id = detection.id[0]

            if tag_id not in self.landmark_set:
                self.landmark_set.add(tag_id)
                self.initial_estimates.insert(L(tag_id), gtsam.Pose2())

            relative_pose = gtsam.Pose2(detection.pose.pose.pose.position.x,
                                        detection.pose.pose.pose.position.y,
                                        detection.pose.pose.pose.orientation.z)  # Assuming yaw angle only

            self.graph.add(gtsam.BetweenFactor(X(self.current_pose), L(tag_id), relative_pose, self.apriltag_noise))

    def add_grid_factors(self):

        # TODO: Create occupancy grid factors for GTSAM. Is this needed?
        grid_factors = []
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                if self.occupancy_grid[x, y] != 0:  # TODO: What should the conditional be to use occupied cells?
                    # Create factor for occupied cell
                    factor = gtsam.RangeFactor(
                        1,  # TODO: Use a shorthand variable
                        gtsam.Point2((self.grid_origin_x + x * self.grid_resolution), (self.grid_origin_y + y * self.grid_resolution)),
                        1.0,  # Range (distance to obstacle)
                        self.occupancy_noise
                    )
                    grid_factors.append(factor)

        # Add occupancy grid factors to the factor graph
        for factor in grid_factors:
            self.graph.add(factor)

    def add_odem_factors(self):

        # TODO: Use snippets from HW_5, to add odometry factor
        assert self.current_pose >= 1
        odometry = gtsam.Pose2(self.odom)

        self.initial_estimate.insert(
            X(self.current_pose), 
            self.initial_estimate.atPose2( X(self.current_pose-1) ).compose(odometry)
        )

        self.graph.add(
            gtsam.BetweenFactor(
                X(self.current_pose-1), 
                X(self.current_pose), 
                odometry, 
                self.odometry_noise
            )
        )

    def optimize(self):

        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, params)
        result = optimizer.optimize()
        marginals = gtsam.Marginals(self.graph, result)
        return result, marginals
    
    def publish_poses(self, result):

        curr_pose = result.atPose2( X(self.current_pose) )
        self.current_pose += 1
        # TODO: Publish poses with self.pose_pub.publish(...)
        
    def run(self):
        # TODO: Finish this function

        rate = rospy.Rate(1)  # 1 Hz
        # Get initial pose
        # Publish updated pose
        while not rospy.is_shutdown():
            # add_odem_factors(...)
            # Get msgs from /scan (and publish occupancy map - update_map(...))
            # add_grid_factors(...)?
            # Get msgs from /tag_detections (and add_tag_factors(...))
            # Optimize SLAM graph to get pose
            # Publish updated pose
            rate.sleep()

if __name__ == '__main__':
    
    try:
        slam = GTSAM()
        slam.run()
    except rospy.ROSInterruptException: 
    	rospy.loginfo("Shutting slam_node down ...")