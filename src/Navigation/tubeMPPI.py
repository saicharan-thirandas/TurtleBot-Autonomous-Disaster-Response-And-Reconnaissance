# tube_mppi_racetrack.py

import numpy as np
from ILQG import AncillaryILQG
from MotionModel import MotionModel, Unicycle, NoisyUnicycle
from MPPI import MPPIRacetrack

class TubeMPPIRacetrack:
    def __init__(self, static_map=None, num_timesteps=20):
        self.static_map = static_map
        self.nominal = MPPIRacetrack(static_map=static_map, num_steps_per_rollout=num_timesteps)
        self.ancillary = AncillaryILQG(static_map=static_map, K=num_timesteps)
        self.motion_model = Unicycle()
        self.z = None

    def simulate(self, x0, u):
        xs = [x0]
        for i in range(len(u)):
            xs.append(self.motion_model.step(xs[i].copy(), u[i].copy()))
        return np.array(xs)

    def solve_ancillary(self, x):
        x_traj, u, cost = self.ancillary.ilqg(x, target=None)
        return x_traj, u, cost

    def solve_nominal(self):
        v = self.nominal.get_action(self.z)
        z_traj = self.simulate(self.z, v)
        return z_traj, v

    def get_action(self, x0, waypoint):
        if self.z is None:
            self.z = x0
        self.nominal.waypoint = np.array(waypoint).reshape((1, -1))
        self.ancillary.waypoint = np.array(waypoint).reshape((1, -1))
        z_traj, v = self.solve_nominal()
        self.ancillary.nominal_states = z_traj
        self.ancillary.nominal_actions = v
        x_traj, u, cost = self.solve_ancillary(x0)
        z_next = self.motion_model.step(self.z.copy(), v[0].copy())
        self.z = z_next
        return u, v

    def update_goal(self, new_goal):
        self.nominal.update_goal(new_goal)
