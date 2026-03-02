import math

import numpy as np


class State:
    """Robot observation.

    Observation vector layout:
        [sin(th1), cos(th1), ..., sin(thN), cos(thN), ee_x, ee_y, dist_x, dist_y, ray_0, ..., ray_M]
    Total length: 2*n_dof + 4 + n_rays*n_lidars
    """

    def __init__(self, thetas: np.ndarray, ee_x: float, ee_y: float, rays: np.ndarray) -> None:
        self.thetas = np.asarray(thetas, dtype=float)  # joint angles, shape (n_dof,)
        self.ee_x: float = float(ee_x)
        self.ee_y: float = float(ee_y)
        self.dist_x: float = 0.0  # set by env after construction
        self.dist_y: float = 0.0  # set by env after construction
        self.lidar_rays: np.ndarray = rays

    def __array__(self, dtype=np.float32) -> np.ndarray:
        trig = np.array([[math.sin(t), math.cos(t)] for t in self.thetas], dtype=float).ravel()
        tail = np.array([self.ee_x, self.ee_y, self.dist_x, self.dist_y], dtype=float)
        rays = self.lidar_rays.ravel()
        return np.concatenate([trig, tail, rays]).astype(dtype)



if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
