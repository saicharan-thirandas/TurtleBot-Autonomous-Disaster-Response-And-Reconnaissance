import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose

T_RC = np.array([[1., 0., 0., 0.], # TODO: Get Camera's frame w.r.t. Robot's base
                 [0., 1., 0., 0.], 
                 [0., 0., 1., 0.],  
                 [0., 0., 0., 1.]])
T_CR = np.eye(4)
T_CR[:3, :3] = T_RC.T
T_CR[:3, -1] = -(T_RC.T) @ np.array(T_RC[:3, -1])


def get_matrix_pose_from_quat(pose, return_matrix=True):
    t = [pose.position.x, 
         pose.position.y, 
         pose.position.z]
    
    r = R.from_quat([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ])

    if not return_matrix:
        _, _, yaw = r.as_euler('xyz', degrees=False)
        return [t[0], t[1], yaw]

    r = r.as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, -1] = np.array(t)
    return T

def get_quat_pose(x, y, yaw):
    
    pose_msg = Pose()

    q = R.from_euler(
        seq='xyz', 
        angles=[0., 0., yaw], 
        degrees=False
    ).as_quat()
    
    pose_msg.position.x = x
    pose_msg.position.y = y
    pose_msg.position.z = 0
    pose_msg.orientation.x = q[0]
    pose_msg.orientation.y = q[1]
    pose_msg.orientation.z = q[2]
    pose_msg.orientation.w = q[3]
    return pose_msg