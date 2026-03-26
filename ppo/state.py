import math

import numpy as np


class State:
    def __init__(
        self,
        thetas: np.ndarray,
        vels: np.ndarray,
        ee_x: float,
        ee_y: float,
        dist_x: float,
        dist_y: float,
        rays: np.ndarray,
    ) -> None:
        self.thetas = np.asarray(thetas, dtype=float)
        self.vels = np.asarray(vels, dtype=float)
        self.ee_x: float = float(ee_x)
        self.ee_y: float = float(ee_y)
        self.dist_x: float = float(dist_x)
        self.dist_y: float = float(dist_y)
        self.lidar_rays: np.ndarray = np.asarray(rays, dtype=float)

    def __array__(self, dtype=np.float32) -> np.ndarray:
        trig = np.array([[math.sin(t), math.cos(t)] for t in self.thetas], dtype=float).ravel()
        vel = self.vels.ravel()
        tail = np.array([self.ee_x, self.ee_y, self.dist_x, self.dist_y], dtype=float)
        rays = self.lidar_rays.ravel()
        return np.concatenate([trig, tail, rays]).astype(dtype)


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
