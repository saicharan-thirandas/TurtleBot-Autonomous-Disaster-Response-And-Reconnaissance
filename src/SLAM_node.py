#!/usr/bin/env python3

import gtsam
from gtsam.symbol_shorthand import X, L
import numpy as np
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R
from apriltag_ros.msg import AprilTagDetectionArray

import os
import sys
sys.path.append(os.path.dirname(__file__))
from transformation_utils import get_quat_pose, get_matrix_pose_from_quat
from mapping import Mapping


class Lidar(Mapping):

    def __init__(self):

        super(Lidar, self).__init__()
        self._init_map()
        self.dx, self.dy, self.dw = [0., 0., 0.]
        self.in_cam_pov = lambda angle_rad: angle_rad >= np.deg2rad(360 + 62.2 - 90) or angle_rad <= np.deg2rad(121.1 - 90)

        # Subscribe to odometry
        self.lidar_sub = rospy.Subscriber(
            name=rospy.get_param('~odom_topic'), 
            data_class=Odometry, 
            callback=self.update_odom
        )

        self.occ_map_pub = rospy.Publisher(
            name=rospy.get_param('~occupancy_map_topic'),
            data_class=OccupancyGrid,
            queue_size=10
        )

        self.occ_map_pub_cam = rospy.Publisher(
            name=rospy.get_param('~occupancy_map_cam_topic'),
            data_class=OccupancyGrid,
            queue_size=10
        )

        # Subscribe to the lidar messages
        self.lidar_sub = rospy.Subscriber(
            name=rospy.get_param('~lidar_topic'), 
            data_class=LaserScan, 
            callback=self.update_map
        )

    def _get_free_grids_from_beam(self, obs_pt_inds, hit_pt_inds) -> np.ndarray:
        diff = hit_pt_inds - obs_pt_inds
        j = np.argmax( np.abs(diff) )
        D = np.abs( diff[j] )
        return obs_pt_inds + ( np.outer( np.arange(D + 1), diff ) + (D // 2) ) // D
    
    def update_odom(self, odom_msg):
        self.current_pose = get_matrix_pose_from_quat(odom_msg.pose.pose, return_matrix=False)
        twist_msg = odom_msg.twist.twist
        self.dx += twist_msg.linear.x
        self.dy += twist_msg.linear.y
        self.dw += twist_msg.angular.z

    def update_map(self, lidar_msg):

        ranges = np.asarray(lidar_msg.ranges)
        x, y, w = list(self.current_pose)
        obs_pt_inds = super()._coords_to_grid_indicies(x, y, w, sign=1)

        for i in range(len(ranges)):
            # Get angle of range
            angle_rad = i * lidar_msg.angle_increment
            beam_angle_rad = w + angle_rad

            if ranges[i] <= lidar_msg.range_min or ranges[i] == np.inf:
                continue

            # if ranges[i] == np.inf:
            #     ranges[i] = 3.5

            # Get x,y position of laser beam in the map
            hit_x = x + np.cos(beam_angle_rad) * ranges[i]
            hit_y = y + np.sin(beam_angle_rad) * ranges[i]

            hit_pt_inds  = super()._coords_to_grid_indicies(hit_x, hit_y, beam_angle_rad, sign=1)
            free_pt_inds = self._get_free_grids_from_beam(obs_pt_inds[:2], hit_pt_inds[:2])

            self.occupancy_grid_logodds[hit_pt_inds[0], hit_pt_inds[1]] = self.occupancy_grid_logodds[hit_pt_inds[0], hit_pt_inds[1]] + self.log_odds_occ - self.log_odds_prior
            self.occupancy_grid_logodds[free_pt_inds[:, 0], free_pt_inds[:, 1]] = self.occupancy_grid_logodds[free_pt_inds[:, 0], free_pt_inds[:, 1]] + self.log_odds_free - self.log_odds_prior

            if self.in_cam_pov(angle_rad):
                self.occupancy_grid_logodds_cam[hit_pt_inds[0], hit_pt_inds[1]] = self.occupancy_grid_logodds_cam[hit_pt_inds[0], hit_pt_inds[1]] + self.log_odds_occ - self.log_odds_prior
                self.occupancy_grid_logodds_cam[free_pt_inds[:, 0], free_pt_inds[:, 1]] = self.occupancy_grid_logodds_cam[free_pt_inds[:, 0], free_pt_inds[:, 1]] + self.log_odds_free - self.log_odds_prior

        self.publish(
            input_grid=super()._log_odds_to_prob(
                log_odds=self.occupancy_grid_logodds
                ) * 100, 
            cam_pub=self.occ_map_pub
        )
        
        self.publish(
            input_grid=super()._log_odds_to_prob(
                log_odds=self.occupancy_grid_logodds_cam
                ) * 100,
            cam_pub=self.occ_map_pub_cam
        )

    def _init_map(self):

        self.map_init = self._init_occupancy_map(
            input_grid=super()._log_odds_to_prob(
                log_odds=self.occupancy_grid_logodds
            ) * 100
        )

    def _init_occupancy_map(self, input_grid: np.ndarray):

        map_init = OccupancyGrid()
        map_init.info.width  = self.grid_size_x
        map_init.info.height = self.grid_size_y
        map_init.info.resolution = self.grid_resolution
        map_init.info.origin.position.x = self.grid_origin_x
        map_init.info.origin.position.y = self.grid_origin_y
        map_init.info.origin.position.z = 0
        map_init.info.origin.orientation.x = 0
        map_init.info.origin.orientation.y = 0
        map_init.info.origin.orientation.z = 0
        map_init.info.origin.orientation.w = 1
        map_init.data = input_grid.flatten().astype(np.int8)
        return map_init

    def publish(self, input_grid: np.ndarray, cam_pub: rospy.Publisher):

        # Conver the map to a 1D array
        map_update = OccupancyGrid()
        map_update.info = self.map_init.info
        map_update.header.frame_id = 'occupancy_grid' if cam_pub.name==rospy.get_param('~occupancy_map_topic') else 'occupancy_grid_camera'
        map_update.header.stamp = rospy.Time.now()
        map_update.data = input_grid.flatten().astype(np.int8)
        # Publish the map
        cam_pub.publish(map_update)


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

        self.apriltag_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1])
        )

        # Publish predicted pose
        self.pose_pub = rospy.Publisher(
            name=rospy.get_param('~pose_topic'),
            data_class=PoseStamped if rospy.get_param('~pose_stamped') else Pose,
            queue_size=10
        )

        # Define occupancy grid factor noise model
        self.occupancy_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]))

        # Create a factor graph
        self.graph = gtsam.NonlinearFactorGraph()

        # Create an initial estimate for the robot's pose (e.g., based on odometry)
        self.initial_estimate = gtsam.Values()
        self.current_pose_idx = 0

        # Subscribe to the detected tags
        self.landmark_set = set()
        self.tag_sub = rospy.Subscriber(
            name=rospy.get_param('~tags_topic'), 
            data_class=AprilTagDetectionArray, 
            callback=self.add_tag_factors
        )

    def _initialize_graph(self):

        assert self.current_pose_idx == 0
        priorMean   = gtsam.Pose2(*self.current_pose) if hasattr(self, 'current_pose') else gtsam.Pose2(0., 0., 0.)
        priorFactor = gtsam.PriorFactorPose2(X(self.current_pose_idx), priorMean, self.prior_noise)
        self.graph.add(priorFactor)
        self.initial_estimate.insert(X(self.current_pose_idx), priorMean)

    def add_tag_factors(self, apriltag_msg):
        
        added_factors = False
        for detection in apriltag_msg.detections:
            added_factors = True
            tag_id = detection.id[0]

            if tag_id not in self.landmark_set:
                self.landmark_set.add(tag_id)
                self.initial_estimate.insert(L(tag_id), gtsam.Pose2())

            relative_pose = gtsam.Pose2(detection.pose.pose.pose.position.x,
                                        detection.pose.pose.pose.position.y,
                                        detection.pose.pose.pose.orientation.z)

            self.graph.add(gtsam.BetweenFactorPose2(X(self.current_pose_idx), L(tag_id), relative_pose, self.apriltag_noise))
        rospy.loginfo(f"ADDED TAG FACTORS: {added_factors}")

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

        assert self.current_pose_idx >= 1
        rospy.loginfo(f"1. CURRENT ODOMETRY FOR SLAM: {[self.dx, self.dy, self.dw]}")
        odometry = gtsam.Pose2(self.dx, self.dy, self.dw)

        self.initial_estimate.insert(
            X(self.current_pose_idx), 
            self.initial_estimate.atPose2( X(self.current_pose_idx-1) ).compose(odometry)
        )

        rospy.loginfo(f"2. ADDING ODOMETRY: {odometry}")
        self.graph.add(
            gtsam.BetweenFactorPose2(
                X(self.current_pose_idx-1), 
                X(self.current_pose_idx), 
                odometry, 
                self.odometry_noise
            )
        )
        self.dx, self.dy, self.dw = [0., 0., 0.]

    def optimize(self):

        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, params)
        result = optimizer.optimize()
        marginals = gtsam.Marginals(self.graph, result)
        return result, marginals
    
    def publish_poses(self, result):

        curr_pose = result.atPose2( X(self.current_pose_idx) )
        pose_msg  = get_quat_pose(x=curr_pose.x(), y=curr_pose.y(), yaw=curr_pose.theta(), stamped=rospy.get_param('~pose_stamped'))
        self.pose_pub.publish(pose_msg)
        self.current_pose_idx += 1
        rospy.loginfo(f"3. SLAM POSE: {curr_pose.x(), curr_pose.y(), curr_pose.theta()}")
        
    def run(self):

        rate = rospy.Rate(1)  # 1 Hz
        self._initialize_graph()
        result, _ = self.optimize()
        self.publish_poses(result)
        while not rospy.is_shutdown():
            self.add_odem_factors()
            # add_grid_factors(...)?
            result, _ = self.optimize()
            self.publish_poses(result)
            rate.sleep()

if __name__ == '__main__':
    
    try:
        slam = GTSAM()
        slam.run()
    except rospy.ROSInterruptException: 
    	rospy.loginfo("Shutting slam_node down ...")