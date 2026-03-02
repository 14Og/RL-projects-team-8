from .state import State
from .config import RobotConfig, LidarConfig
from .obstacle import Obstacle
from .lidar import Lidar

import numpy as np

import math
from typing import Optional, Tuple, List

class Robot:
    def __init__(self, robot_cfg: RobotConfig, lidar_cfg: LidarConfig, obstacles: List[Obstacle], seed: int = 42, theta: Optional[np.ndarray] = None):
        self.cfg = robot_cfg
        self.lidar_cfg = lidar_cfg
        self.rng = np.random.default_rng(seed)

        self.base = np.asarray(self.cfg.base_xy, dtype=float)
        self.links = np.asarray(self.cfg.link_lengths, dtype=float)  # shape (n_dof,)
        self.n_dof: int = len(self.links)

        self._theta = np.zeros(self.n_dof, dtype=float)
        if theta is not None:
            self.set_theta(theta)

    @property
    def theta(self) -> np.ndarray:
        """Joint angles (copy), shape (n_dof,), radians."""
        return self._theta.copy()

    def joints_xy(self) -> np.ndarray:
        """Returns joint positions including base, shape (n_dof+1, 2)."""
        points = [self.base]
        cumulative_angle = 0.0
        p = self.base
        for i, L in enumerate(self.links):
            cumulative_angle += self._theta[i]
            p = p + np.array([L * math.cos(cumulative_angle), L * math.sin(cumulative_angle)], dtype=float)
            points.append(p)
        return np.stack(points, axis=0)

    def end_effector_xy(self) -> np.ndarray:
        return self.joints_xy()[-1]

    def obs(self) -> State:
        """RL observation: [sin(th1), cos(th1), ..., sin(thN), cos(thN), ee_x, ee_y, dist_x, dist_y]"""
        ee = self.end_effector_xy()
        return State(thetas=self._theta.copy(), ee_x=ee[0], ee_y=ee[1])

    def reset(self, randomize: bool = True) -> State:
        """Reset robot angles. Returns obs()."""
        if randomize:
            self._theta = self.rng.uniform(-math.pi, math.pi, size=self.n_dof).astype(float)
        else:
            self._theta[:] = 0.0

        if self.cfg.wrap_angles:
            self._theta = np.array([self.wrap_angle(t) for t in self._theta], dtype=float)

        return self.obs()

    def set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float).reshape(self.n_dof)
        if self.cfg.wrap_angles:
            theta = np.array([self.wrap_angle(float(t)) for t in theta], dtype=float)
        self._theta = theta

    def step(self, dtheta: np.ndarray) -> Tuple[State, np.ndarray]:
        dtheta = np.asarray(dtheta, dtype=float).reshape(self.n_dof)
        if self.cfg.dtheta_max is not None:
            dtheta = self.clip_dtheta(dtheta, self.cfg.dtheta_max)

        self._theta = self._theta + dtheta

        if self.cfg.wrap_angles:
            self._theta = np.array([self.wrap_angle(t) for t in self._theta], dtype=float)

        return self.obs(), dtheta

    @staticmethod
    def wrap_angle(theta: float) -> float:
        return (theta + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def clip_dtheta(dtheta: np.ndarray, max_dtheta: float) -> np.ndarray:
        return np.clip(dtheta, -max_dtheta, max_dtheta)


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
