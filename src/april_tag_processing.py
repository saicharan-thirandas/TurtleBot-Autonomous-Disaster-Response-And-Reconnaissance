import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np

from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from pupil_apriltags import Detector
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class AprilTagDetector(Node):

    def __init__(self):
        super().__init__('april_tag_detector')
        self.bridge = CvBridge()

        # Subscribe to camera image topic
        self.image_sub = self.create_subscription(
            msg_type=Image,
            topic='/camera/image_raw',
            callback=self.image_callback,
            queue_size=10
        )
        self.image_sub  # Prevent unused variable warning

        # >>>>>>>>>>>>>> Subscribe to pose of robot and SLAM map estimation <<<<<<<<<<<<<<

        self.image_pub = self.create_publisher(
            msg_type=Image, 
            topic="/image_converter/output_video", 
            queue_size=10
        )

        # Publish detected AprilTag poses
        self.tag_publisher = self.create_publisher(
            msg_type=AprilTagDetectionArray, 
            topic='/tag_detections', 
            queue_size=10
        )

    def image_callback(self, msg):

        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # Convert image to grayscale (AprilTag library requires grayscale images)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Initialize AprilTag detector
        detector = Detector(families="tag36h11")

        # Detect AprilTags in the image
        detections = detector.detect(
            gray_image, 
            estimate_tag_pose=True, 
            camera_params=(499.11014636, 
                           498.6075723, 
                           316.14098243, 
                           247.3739291), 
            tag_size=0.06
        )

        if len(detections) == 0:
            return
        
        # Publish detected tag poses
        tag_msg = AprilTagDetectionArray()
        tag_msg.header = msg.header
        for detection in detections:
            tag = AprilTagDetection()

            tag_id   = detection["id"]
            tag_fam  = detection.tag_family.decode("utf-8")
            tag_size = detection["tag_family"]
            tag_size = int(tag_size.replace("tag", ""))

            rot = detection.pose_R
            t = detection.pose_t
            r = R.from_matrix(rot)
            q = r.as_quat()
            
            tag.id = tag_id
            tag.size = tag_size
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
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

# In subscriber: 
        # for tag_detection in tag_msgs:
            # # -----------------------------------------
            # tag_pose = tag_detection.pose.pose.pose
            # t = [tag_pose.position.x, 
            #      tag_pose.position.y, 
            #      tag_pose.position.z
            # ]
            # q = [tag_pose.orientation.x, 
            #      tag_pose.orientation.y, 
            #      tag_pose.orientation.z, 
            #      tag_pose.orientation.w
            # ]
            # r = R.from_quat(q).as_matrix()
            # # -----------------------------------------