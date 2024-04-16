import numpy as np
from dataclasses import dataclass


@dataclass
class Grid:
    grid_resolution = 0.1 # meters per grid cell
    grid_size_x = 384  # number of grid cells in the x direction
    grid_size_y = 384  # number of grid cells in the y direction
    grid_origin_x = -10.0  # origin of the grid in the x direction
    grid_origin_y = -10.0  # origin of the grid in the y direction


class Mapping(Grid):
    def __init__(self, 
                 p_free: float=0.1,
                 p_occ: float=1.0, 
                 p_prior: float=0.5):
        
        # Initialize occupancy grid with p_prior
        self.occupancy_grid_logodds = np.zeros((
            self.grid_size_x, 
            self.grid_size_y
        ))

        self.occupancy_grid_logodds_cam = np.zeros((
            self.grid_size_x, 
            self.grid_size_y
        ))
        
        self.log_odds_free  = self._prob_to_log_odds(p_free)
        self.log_odds_occ   = self._prob_to_log_odds(p_occ)
        self.log_odds_prior = self._prob_to_log_odds(p_prior)
    
    def _log_odds_to_prob(self, log_odds: np.ndarray) -> np.ndarray:
        return 1 - 1 / (1e-6 + 1 + np.exp(log_odds))

    def _prob_to_log_odds(self, prob: np.ndarray) -> np.ndarray:
        return np.log(prob / (1e-6 + 1 - prob))

    def _coords_to_grid_indicies(self, x, y, w, sign=1):
        grid_x = int((x + sign * self.grid_origin_x) / self.grid_resolution)
        grid_y = int((y + sign * self.grid_origin_y) / self.grid_resolution)
        return np.array([grid_x, grid_y, int(w)])

    def _grid_indices_to_coords(self, grid_x, grid_y, w, sign=1):
        x = grid_x * self.grid_resolution - sign * self.grid_origin_x
        y = grid_y * self.grid_resolution - sign * self.grid_origin_y
        return x, y, w