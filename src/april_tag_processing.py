#!/usr/bin/env python

import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np

import rospy
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from pupil_apriltags import Detector
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


# TODO: Confirm these; i.e. perform camera calibration
FX = 2.68e3
FY = 2.49e3
CX = 1.64e3
CY = 1.21e3


def draw_bbox(image, corners, center, tag_fam):

    (ptA, ptB, ptC, ptD) = corners

    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))

    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)

    (cX, cY) = (int(center[0]), int(center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
    cv2.putText(
        image, 
        tag_fam, 
        (ptA[0], ptA[1] - 15), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 255, 0), 
        2
    )


class AprilTagDetector():

    def __init__(self):

        rospy.init_node('april_tag_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # Initialize AprilTag detector
        self.detector = Detector(families="tag36h11")

        # Subscribe to camera image topic
        self.image_sub = rospy.Subscriber(
            name='/camera_rect/image_raw',
            data_class=Image,
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
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
            return
        
        # Publish detected tag poses
        tag_msg = AprilTagDetectionArray()
        tag_msg.header = msg.header
        for detection in detections:
            tag = AprilTagDetection()

            tag_id   = detection.tag_id
            tag_fam  = detection.tag_family.decode("utf-8")

            draw_bbox(cv_image, detection.corners, detection.center, tag_fam)

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

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
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