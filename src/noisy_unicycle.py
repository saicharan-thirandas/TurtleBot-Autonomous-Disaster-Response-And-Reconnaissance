"""Discrete-time unicycle kinematic model w/ process noise."""

import numpy as np

from .unicycle import Unicycle


class NoisyUnicycle(Unicycle):
    """Same as Unicycle but add process noise rather during step."""

    def __init__(
        self,
        process_noise_limits=np.array([0.02, 0.02, 0.1]),
        v_min=0,
        v_max=1,
        w_min=-2 * np.pi,
        w_max=2 * np.pi,
    ):
        self.process_noise_limits = process_noise_limits
        super().__init__(v_min=v_min, v_max=v_max, w_min=w_min, w_max=w_max)

    def step(self, current_state, action, dt=0.1) -> np.ndarray:
        """Add process noise to the parent kinematics model."""
        next_state = super().step(current_state, action, dt=dt)
        perturbed_next_state = next_state + np.random.uniform(
            low=-self.process_noise_limits, high=self.process_noise_limits
        )

        return perturbed_next_state
