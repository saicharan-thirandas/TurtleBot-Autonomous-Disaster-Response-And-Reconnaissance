import numpy as np
from dataclasses import dataclass
import rospy

@dataclass
class Grid:
    grid_resolution = 1/20 # meters per grid cell
    grid_size_x = 20  # number of grid cells in the x direction
    grid_size_y = 20  # number of grid cells in the y direction
    grid_origin_x = 10  # origin of the grid in the x direction
    grid_origin_y = 10  # origin of the grid in the y direction


class Mapping(Grid):
    def __init__(self, 
                 p_free: float=0.1,
                 p_occ: float=1.0, 
                 p_prior: float=0.5):
        
        occupancy_grid_dims = (
            int(self.grid_size_x / self.grid_resolution),
            int(self.grid_size_y / self.grid_resolution),
        )

        # Initialize occupancy grid with p_prior
        self.occupancy_grid_logodds = np.zeros(
            occupancy_grid_dims
        )

        self.occupancy_grid_logodds_cam = np.zeros(
            occupancy_grid_dims
        )
        self.occupancy_grid_logodds_cam_filter = np.zeros(
            occupancy_grid_dims
        )

        self.log_odds_free  = self._prob_to_log_odds(p_free)
        self.log_odds_occ   = self._prob_to_log_odds(p_occ)
        self.log_odds_prior = self._prob_to_log_odds(p_prior)

        self.T = (1 / self.grid_resolution) * np.array(
            [
                [1, 0, self.grid_origin_x],
                [0, 1, self.grid_origin_y],
                [0, 0, 1],
            ]
        )
        self.T_inv = np.linalg.inv(self.T)

    def _log_odds_to_prob(self, log_odds: np.ndarray) -> np.ndarray:
        return 1 - 1 / (1e-6 + 1 + np.exp(log_odds))

    def _prob_to_log_odds(self, prob: np.ndarray) -> np.ndarray:
        return np.log(prob / (1e-6 + 1 - prob))
    
    def _world_coordinates_to_map_indices(self, position_in_world):

        position_in_world = np.array(position_in_world)
        position_in_world = position_in_world.reshape((-1, 2))
        column_ones = np.ones((position_in_world.shape[0], 1))
        position_homo = np.hstack([position_in_world, column_ones])
        position_in_grid = np.dot(self.T, position_homo.T).T
        return np.floor(position_in_grid).astype(int).squeeze()
    
    def _in_map(self, grid_coords):

        in_map = np.logical_and.reduce(
            (
                grid_coords[:, 0] >= 0,
                grid_coords[:, 1] >= 0,
                grid_coords[:, 0] < self.static_map.shape[0],
                grid_coords[:, 1] < self.static_map.shape[1],
            )
        )
        not_in_map_inds = np.where(in_map == False)
        grid_coords[not_in_map_inds[0], :] = -1
        return in_map, grid_coords

    def _grid_indices_to_coords(self, grid_x, grid_y, grid_w):
        grid_xyw = np.array([grid_x, grid_y, grid_w]).reshape((1, -1))
        return np.dot(self.T_inv, grid_xyw.T).T.squeeze()