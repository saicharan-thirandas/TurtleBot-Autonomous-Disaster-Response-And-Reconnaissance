# tube_mppi_racetrack.py

import numpy as np
import rospy
import os
import sys
sys.path.append(os.path.dirname(__file__))

from ILQG import AncillaryILQG
from MotionModel import Unicycle
from MPPI import MPPIRacetrack

class TubeMPPIRacetrack:
    def __init__(self, static_map=None, num_timesteps=20):
        
        # MPPI Racetrack to score MPPI
        self.nominal = MPPIRacetrack(static_map=static_map, num_steps_per_rollout=num_timesteps)
        
        # 
        self.ancillary = AncillaryILQG(K=num_timesteps)
        
        # Motion model for turtlebot
        self.motion_model = Unicycle()
        
        # Initial static map
        self.static_map = static_map
        self.z = None

    def update_static_map(self, new_grid):
        self.static_map = new_grid
        self.nominal.static_map = new_grid

    def simulate(self, x0, u):
        xs = [x0]
        for i in range(len(u)):
            x1 = self.motion_model.step(xs[i].copy(), u[i].copy())
            xs.append(x1)
        xs = np.array(xs)
        return xs
    
    def solve_ancillary(self, x):
        x_traj, u, cost = self.ancillary.ilqg(x, target=None)
        return x_traj, u, cost

    def solve_nominal(self):
        v = self.nominal.get_action(self.z) # [1, 3] -> [20, 2]
        z_traj = self.simulate(self.z, v) # [1, 3], [20, 2]
        return z_traj, v

    def get_action(self, x0):
        x0 = x0.squeeze()
        # if self.z is None:
        self.z = x0
        z_traj, v = self.solve_nominal()
        self.ancillary.nominal_states = z_traj
        self.ancillary.nominal_actions = v
        x_traj, u, cost = self.solve_ancillary(x0)
        z_next = self.motion_model.step(self.z.copy(), v[0].copy())
        self.z = z_next
        return u, v

    def update_goal(self, new_goal):
        self.nominal.waypoint = np.array(new_goal).reshape((1, -1))
        self.ancillary.waypoint = np.array(new_goal).reshape((1, -1))
