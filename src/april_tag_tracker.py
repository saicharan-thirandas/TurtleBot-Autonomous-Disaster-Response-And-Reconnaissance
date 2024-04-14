#!/usr/bin/env python

import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np

import rospy
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from geometry_msgs.msg import Pose


class AprilTagTracker():

    def __init__(self):
        rospy.init_node('april_tag_tracker', anonymous=True)

        # Subscribe to SLAM topic
        self.tag_sub = rospy.Subscriber(
            name='/turtle_pose',
            data_class=Pose,
            callback=self.turtle_pose_update,
            queue_size=10
        )

        # Subscribe to tag_detection topic
        self.tag_sub = rospy.Subscriber(
            name='/tag_detections',
            data_class=AprilTagDetectionArray,
            callback=self.tag_update_callback,
            queue_size=10
        )

        # Publish tracked AprilTag poses
        self.tag_track_pub = rospy.Publisher(
            name='/tag_tracker', 
            data_class=AprilTagDetectionArray, 
            queue_size=10
        )

        self.T_OR = np.eye(4)
        self.T_RO = np.eye(4)

        self.T_RC = np.array([[1., 0., 0., 0.], # TODO: Get Camera's frame w.r.t. Robot's base
                              [0., 1., 0., 0.], 
                              [0., 0., 1., 0.],  
                              [0., 0., 0., 1.]])
        self.T_CR = np.eye(4)
        self.T_CR[:3, :3] = self.T_RC.T
        self.T_CR[:3, -1] = -(self.T_RC.T) @ np.array(self.T_RC[:3, -1])

    def turtle_pose_update(self, turtle_pose_msg):
        
        # SLAM output will be T_OR. Or Robot w.r.t. to Origin
        t = [turtle_pose_msg.position.x, 
             turtle_pose_msg.position.y, 
             turtle_pose_msg.position.z]

        q = [turtle_pose_msg.orientation.x, 
             turtle_pose_msg.orientation.y, 
             turtle_pose_msg.orientation.z, 
             turtle_pose_msg.orientation.w]
        r = R.from_quat(q).as_matrix()

        self.T_OR[:3, :3] = r
        self.T_OR[:3, -1] = np.array(t)

        # Transform T_OR to T_RO
        self.T_RO[:3, :3] = r.T
        self.T_RO[:3, -1] = -(r.T) @ np.array(t)

    def tag_update_callback(self, tag_detections):
        
        for tag_detection in tag_detections.detections:
            tag_pose = tag_detection.pose.pose.pose

            t = [tag_pose.position.x, 
                 tag_pose.position.y, 
                 tag_pose.position.z]
            q = [tag_pose.orientation.x, 
                 tag_pose.orientation.y, 
                 tag_pose.orientation.z, 
                 tag_pose.orientation.w]
            r = R.from_quat(q).as_matrix()

            T_CA = np.eye(4)
            T_CA[:3, :3] = r
            T_CA[:3, -1] = np.array(t)

            T_OA = self.T_OR @ self.T_RC @ T_CA
            # self.update_goal_pub.publish()
        
        # Publish all tracked tags and their T_CA poses
        tag_msg = AprilTagDetectionArray()
        tag_msg.header = tag_detections.header

        # ---------------------------------------- #
        # for tag_id, T_OA_tag in :
        #     tag = AprilTagDetection()
        #     T_CA = self.T_CR @ self.T_RO @ T_OA_tag
                
        #     q = R.from_matrix(T_CA[:3, :3]).as_quat()
        #     t = T_CA[:3, -1]

        #     tag.id = tag_id
        #     tag.pose.pose.pose.position.x = t[0]
        #     tag.pose.pose.pose.position.y = t[1]
        #     tag.pose.pose.pose.position.z = t[2]
        #     tag.pose.pose.pose.orientation.x = q[0]
        #     tag.pose.pose.pose.orientation.y = q[1]
        #     tag.pose.pose.pose.orientation.z = q[2]
        #     tag.pose.pose.pose.orientation.w = q[3]
        #     tag_msg.detections.append(tag)
        # ---------------------------------------- #

        self.tag_track_pub.publish(tag_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    
    try:
        tag_tracker = AprilTagTracker()
        tag_tracker.run()
    except rospy.ROSInterruptException: 
    	rospy.loginfo("Shutting april_tag_detector down ...")