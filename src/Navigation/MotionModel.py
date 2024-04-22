#!/usr/bin/env python
import numpy as np
from gymnasium import spaces

DT = 0.2
V_MIN = -0.22
V_MAX = 0.22

class MotionModel:
    """Abstract class for modeling the motion of a 2D mobile robot within a ROS node."""
    
    def __init__(self):
        pass
    
    def step(self, current_pose, velocity_cmd, dt=DT):
        """Move 1 timestep forward using a kinematic model."""
        raise NotImplementedError

class Unicycle(MotionModel):
    """Discrete-time unicycle kinematic model for 2D robot simulator."""

    def __init__(self, v_min=V_MIN, v_max=V_MAX, w_min=-2 * np.pi, w_max=2 * np.pi):
        super().__init__()
        self.action_space = spaces.Box(
            np.array([v_min, w_min]),
            np.array([v_max, w_max]),
            shape=(2,),
            dtype=float,
        )


    def step(
        self, current_state: np.ndarray, action: np.ndarray, dt: float = DT
    ) -> np.ndarray:
        """Move 1 timestep forward w/ kinematic model, x_{t+1} = f(x_t, u_t)"""
        # current_state = np.array([x, y, theta])
        # action = np.array([vx, vw])

        # clip the action to be within the control limits
        clipped_action = np.clip(
            action, 
            self.action_space.low, 
            self.action_space.high
        )

        current_state = current_state.reshape((-1, 3))
        clipped_action = clipped_action.reshape((-1, 2))
        next_state = np.empty_like(current_state)

        next_state[:, 0] = current_state[:, 0] + dt * clipped_action[
            :, 0
        ] * np.cos(current_state[:, 2])
        next_state[:, 1] = current_state[:, 1] + dt * clipped_action[
            :, 0
        ] * np.sin(current_state[:, 2])
        next_state[:, 2] = current_state[:, 2] + dt * clipped_action[:, 1]

        next_state = next_state.squeeze()

        return next_state


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
