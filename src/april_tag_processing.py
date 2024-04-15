#!/usr/bin/env python

import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np

import rospy
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from pupil_apriltags import Detector
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2

import os
import sys
sys.path.append(os.path.dirname(__file__))
from image_processing import draw_bbox


# TODO: Confirm these; i.e. perform camera calibration
FX = 2.68e3
FY = 2.49e3
CX = 1.64e3
CY = 1.21e3


class AprilTagDetector():

    def __init__(self):

        rospy.init_node('april_tag_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # Initialize AprilTag detector
        self.detector = Detector(families="tag36h11")

        # Subscribe to camera image topic
        self.image_sub = rospy.Subscriber(
            name='/camera_rect/image_raw/compressed', # '/raspicam_node/image/compressed', /camera_rect/image_raw/compressed
            data_class=CompressedImage,
            callback=self.image_callback,
            queue_size=10
        )

        # Publish processed image
        self.image_pub = rospy.Publisher(
            name='/output_video',
            data_class=Image,
            queue_size=10
        )

        # Publish detected AprilTag poses
        self.tag_publisher = rospy.Publisher(
            name='/tag_detections', 
            data_class=AprilTagDetectionArray, 
            queue_size=10
        )

    def image_callback(self, msg):

        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        # Convert image to grayscale (AprilTag library requires grayscale images)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the image
        detections = self.detector.detect(
            gray_image, 
            estimate_tag_pose=True, 
            camera_params=(FX, 
                           FY, 
                           CX, 
                           CY), 
            tag_size=0.06
        )

        if len(detections) == 0:
            self.publish_image(cv_image)
            return
        
        rospy.loginfo(f"Sucessfully detected {len(detections)} tags")
        
        # Publish detected tag poses
        tag_msg = AprilTagDetectionArray()
        tag_msg.header = msg.header
        for detection in detections:
            tag = AprilTagDetection()

            tag_id   = detection.tag_id
            tag_fam  = detection.tag_family.decode("utf-8")

            cv_image = draw_bbox(cv_image, detection.corners, detection.center, tag_fam)

            rot = detection.pose_R
            t = detection.pose_t
            r = R.from_matrix(rot)
            q = r.as_quat()

            tag.id.append(tag_id)
            tag.pose.pose.pose.position.x = t[0]
            tag.pose.pose.pose.position.y = t[1]
            tag.pose.pose.pose.position.z = t[2]
            tag.pose.pose.pose.orientation.x = q[0]
            tag.pose.pose.pose.orientation.y = q[1]
            tag.pose.pose.pose.orientation.z = q[2]
            tag.pose.pose.pose.orientation.w = q[3]
            tag_msg.detections.append(tag)

        self.tag_publisher.publish(tag_msg)
        self.publish_image(cv_image)

    def publish_image(self, cv_image):
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image.astype(np.uint8), 'bgr8'))
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return
        
    def run(self):
        rospy.spin()
        
if __name__ == '__main__':
    
    try:
        tag_detector = AprilTagDetector()
        tag_detector.run()
    except rospy.ROSInterruptException: 
    	rospy.loginfo("Shutting april_tag_detector down ...")