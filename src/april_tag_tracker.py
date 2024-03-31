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
        # Subscribe to tag_detection node
        self.image_sub = self.create_subscription(
            msg_type=AprilTagDetectionArray,
            topic='/tag_detections',
            callback=self.tag_update_callback,
            queue_size=10
        )
        self.tag_tracker = {}
        self.smoothing_factor = 0.9

    def tag_update_callback(self, tag_detections):
        if len(tag_detections.detections) == 0:
            return

        for tag_detection in tag_detections.detections:    
            tag_id   = tag_detection.id[0]
            tag_pose = tag_detection.pose.pose.pose

            t = [tag_pose.position.x, 
                 tag_pose.position.y, 
                 tag_pose.position.z]
            q = [tag_pose.orientation.x, 
                 tag_pose.orientation.y, 
                 tag_pose.orientation.z, 
                 tag_pose.orientation.w]
            r = R.from_quat(q).as_matrix()

            rot = np.eye(4)
            rot[:3, :3] = r

            if tag_id in self.tag_tracker.keys():
                L = 0.9
                self.tag_tracker[tag_id] = self.smoothing_factor * self.tag_tracker[tag_id] + (1 - self.smoothing_factor) * T_AO
            else:
                self.tag_tracker[tag_id] = T_AO


# TODO:
    # - In SLAM_node.py create a publisher '/turtle_pose' that publishes the robot's predicted pose
    # - Here, subscribe to the '/turtle_pose' topic.
