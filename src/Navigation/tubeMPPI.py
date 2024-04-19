# tube_mppi_racetrack.py

import numpy as np
from ILQG import AncillaryILQG
from MotionModel import MotionModel, Unicycle, NoisyUnicycle
from MPPI import MPPIRacetrack

class TubeMPPIRacetrack:
    def __init__(self, static_map = None, num_timesteps = 25):
      #remember to set init state for nominal controller before starting
      self.static_map = static_map
      self.nominal = MPPIRacetrack(static_map = static_map, num_steps_per_rollout=num_timesteps)
      self.ancillary = AncillaryILQG(static_map = static_map, K = num_timesteps)
      self.motion_model = Unicycle()
      self.z = None
      self.threshold = 10
      self.times = []

    def simulate(self, x0, u):
      xs = [x0]
      for i in range(len(u)):
        xs.append(self.motion_model.step(xs[i], u[i]))

      return np.array(xs)

    def solve_ancillary(self, x):
      """The ancillary controller finds the control seq which minimizes the
      difference between the uncertain state and the nominal state"""
      x_traj, u, cost = self.ancillary.ilqg(x, target = None)
      return x_traj, u, cost

    def solve_nominal(self, x):
      """ Use MPPI to solve nominal problem"""
      v, score, z_traj = self.nominal.get_action(self.z.copy())
      v_actual, score_actual, z_traj_actual = self.nominal.get_action(x.copy())

      if score_actual <= score + self.threshold:
        self.z = x
        v = v_actual
        z_traj = z_traj_actual

      z_traj = self.simulate(self.z, v)
      return z_traj, v

    def get_action(self, x0):
      start = timeit.default_timer()
      if self.z is None:
        self.z = x0

      #solve the nominal system and get the states and controls
      z_traj, v = self.solve_nominal(x0)
      # plt.scatter(z_traj[:, 0], z_traj[:, 1])
      # plt.show()

      #solve ancillary problem and obtain control
      self.ancillary.nominal_states = z_traj
      self.ancillary.nominal_actions = v
      x_traj, u, cost = self.solve_ancillary(x0)
      # plt.scatter(x_traj[:, 0], x_traj[:, 1])
      # plt.show()

      #simulate the next state of nominal system
      z_next = self.motion_model.step(self.z.copy(), v[0])
      self.z = z_next

      self.times.append(timeit.default_timer() - start)

      return u, v