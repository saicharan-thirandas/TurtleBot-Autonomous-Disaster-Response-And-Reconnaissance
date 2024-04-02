import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np

from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from pupil_apriltags import Detector
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class AprilTagTracker(Node):

    def __init__(self):
        super().__init__('april_tag_tracker')

        # Subscribe to SLAM topic
        self.tag_sub = self.create_subscription(
            msg_type=None,
            topic='/turtle_pose',
            callback=self.turtle_pose_update,
            queue_size=10
        )

        # Subscribe to tag_detection topic
        self.tag_sub = self.create_subscription(
            msg_type=AprilTagDetectionArray,
            topic='/tag_detections',
            callback=self.tag_update_callback,
            queue_size=10
        )

        self.tag_track_pub = self.create_publisher(
            msg_type=AprilTagDetectionArray, 
            topic="/tag_tracker", 
            queue_size=10
        )

        self.tag_tracker = {}
        self.smoothing_factor = 0.9
        self.T_RO = np.eye(4)
        self.T_CR = np.array([[1., 0., 0., 0.], # TODO: Get Robot's frame w.r.t. Camera
                              [0., 1., 0., 0.], 
                              [0., 0., 1., 0.],  
                              [0., 0., 0., 1.]])

    def turtle_pose_update(self, turtle_pose_msg):
        # TODO: Extract the correct pose matrix. Need to figure out the format to use for turtle_pose_msg. Temporary solution for now.
        
        # SLAM output will be T_OR. Or Robot w.r.t. to Origin
        turtle_pose = turtle_pose_msg.pose.pose.pose
        t = [turtle_pose.position.x, 
             turtle_pose.position.y, 
             turtle_pose.position.z]

        q = [turtle_pose.orientation.x, 
             turtle_pose.orientation.y, 
             turtle_pose.orientation.z, 
             turtle_pose.orientation.w]
        r = R.from_quat(q).as_matrix()

        # Transform T_OR to T_RO
        self.T_RO[:3, :3] = r.T
        self.T_RO[:3, -1] = -(r.T) @ np.array(t)

    def tag_update_callback(self, tag_detections):

        for tag_detection in tag_detections.detections:
            tag = AprilTagDetection()

            tag_id   = tag_detection.id[0]
            tag_pose = tag_detection.pose.pose.pose
            tag_size = tag_detection["tag_family"]
            tag_size = int(tag_size.replace("tag", ""))

            t = [tag_pose.position.x, 
                 tag_pose.position.y, 
                 tag_pose.position.z]
            q = [tag_pose.orientation.x, 
                 tag_pose.orientation.y, 
                 tag_pose.orientation.z, 
                 tag_pose.orientation.w]
            r = R.from_quat(q).as_matrix()

            T_OA = np.eye(4)
            T_OA[:3, :3] = r
            T_OA[:3, -1] = np.array(t)

            T_CA = self.T_CR @ self.T_RO @ T_OA

            if tag_id in self.tag_tracker.keys():
                self.tag_tracker[tag_id] = self.smoothing_factor * self.tag_tracker[tag_id] + (1 - self.smoothing_factor) * T_CA
            else:
                self.tag_tracker[tag_id] = T_CA
        
        # Publish all tracked tags and their T_CA poses
        tag_msg = AprilTagDetectionArray()
        tag_msg.header = tag_detections.header
        for tag_id, T_CA_tag in self.tag_tracker.items():

            # TODO: Using T_CA_tag, get (x, y, z) and <q>: [x, y, z, w]
            tag.id = tag_id
            tag.pose.pose.pose.position.x = None
            tag.pose.pose.pose.position.y = None
            tag.pose.pose.pose.position.z = None
            tag.pose.pose.pose.orientation.x = None
            tag.pose.pose.pose.orientation.y = None
            tag.pose.pose.pose.orientation.z = None
            tag.pose.pose.pose.orientation.w = None
            tag_msg.detections.append(tag)

        self.tag_track_pub.publish(tag_msg)

# TODO:
    # - In SLAM_node.py create a publisher '/turtle_pose' that publishes the robot's predicted pose