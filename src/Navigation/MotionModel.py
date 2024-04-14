#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, Pose
import numpy as np

class MotionModel:
    """Abstract class for modeling the motion of a 2D mobile robot within a ROS node."""
    
    def __init__(self):
        pass
    
    def step(self, current_pose, velocity_cmd, dt=0.1):
        """Move 1 timestep forward using a kinematic model."""
        raise NotImplementedError

class Unicycle(MotionModel):
    """Discrete-time unicycle kinematic model integrated with ROS for 2D robot simulation."""
    
    def __init__(self, v_min=0, v_max=1, w_min=-2 * np.pi, w_max=2 * np.pi):
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max
        super(Unicycle, self).__init__()
    
    def step(self, current_pose, velocity_cmd, dt=0.1):
        # Extract current state
        x, y, theta = current_pose.position.x, current_pose.position.y, current_pose.orientation.z
        
        # Extract commands
        vx = max(min(velocity_cmd.linear.x, self.v_max), self.v_min)
        vw = max(min(velocity_cmd.angular.z, self.w_max), self.w_min)
        
        # Compute next state
        x_next = x + vx * np.cos(theta) * dt
        y_next = y + vx * np.sin(theta) * dt
        theta_next = theta + vw * dt
        
        # Update pose
        next_pose = Pose()
        next_pose.position.x = x_next
        next_pose.position.y = y_next
        next_pose.orientation.z = theta_next
        
        return next_pose

class NoisyUnicycle(Unicycle):
    """Unicycle model with added process noise, integrated with ROS."""
    
    def __init__(self, process_noise_limits=np.array([0.02, 0.02, 0.1]) * 2, v_min=0, v_max=1, w_min=-2*np.pi, w_max=2*np.pi):
        self.process_noise_limits = process_noise_limits
        super(NoisyUnicycle, self).__init__(v_min=v_min, v_max=v_max, w_min=w_min, w_max=w_max)
    
    def step(self, current_pose, velocity_cmd, dt=0.1):
        next_pose = super(NoisyUnicycle, self).step(current_pose, velocity_cmd, dt)
        
        # Add noise
        noise = np.random.uniform(-self.process_noise_limits, self.process_noise_limits)
        next_pose.position.x += noise[0]
        next_pose.position.y += noise[1]
        next_pose.orientation.z += noise[2]
        
        return next_pose
