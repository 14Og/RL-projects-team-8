from .state import State
from .config import RobotConfig, LidarConfig
from .obstacle import Obstacle
from .lidar import LidarManager
from .physics_robot import Robot_Dynamic_3DOF


import numpy as np

import math
from typing import List, Optional, Tuple


class Robot:
    def __init__(
        self,
        robot_cfg: RobotConfig,
        lidar_cfg: LidarConfig,
        obstacles: List[Obstacle],
        seed: int = 42,
        theta: Optional[np.ndarray] = None,
    ):
        self.cfg = robot_cfg
        self.rng = np.random.default_rng(seed)

        self.base = np.asarray(self.cfg.base_xy, dtype=float)
        self.links = np.asarray(self.cfg.link_lengths, dtype=float)
        self.n_dof: int = len(self.links)
        self._reach_max: float = float(np.sum(self.links))

        self._theta = np.zeros(self.n_dof, dtype=float)
        self._target: np.ndarray = self.base.copy()

        self._lidar_manager = LidarManager(lidar_cfg, self.n_dof)
        self._lidar_manager.set_obstacles(obstacles)

        self.physics = Robot_Dynamic_3DOF(
            masses=np.array(self.cfg.masses),
            lengthes=np.array(self.cfg.link_lengths) / 100.0
        )

        if theta is not None:
            self.set_theta(theta)

        self._dq = np.zeros(self.n_dof, dtype=float)
        self._q_target = self._theta.copy()

        self.Kp = np.array(self.cfg.Kp)
        self.Kd = np.array(self.cfg.Kd)

    @property
    def theta(self) -> np.ndarray:
        return self._theta.copy()

    @property
    def lidar_manager(self) -> LidarManager:
        return self._lidar_manager

    def set_target(self, target: np.ndarray) -> None:
        self._target = np.asarray(target, dtype=float)

    def set_obstacles(self, obstacles: List[Obstacle]) -> None:
        self._lidar_manager.set_obstacles(obstacles)

    def joints_xy(self) -> np.ndarray:
        points = [self.base]
        cumulative_angle = 0.0
        p = self.base
        for i, L in enumerate(self.links):
            cumulative_angle += self._theta[i]
            p = p + np.array(
                [L * math.cos(cumulative_angle), L * math.sin(cumulative_angle)], dtype=float
            )
            points.append(p)
        return np.stack(points, axis=0)

    def end_effector_xy(self) -> np.ndarray:
        return self.joints_xy()[-1]

    def obs(self) -> State:
        joints = self.joints_xy()
        ee = joints[-1]

        ee_x = (ee[0] - self.base[0]) / self._reach_max
        ee_y = (ee[1] - self.base[1]) / self._reach_max
        dist_x = (self._target[0] - ee[0]) / self._reach_max
        dist_y = (self._target[1] - ee[1]) / self._reach_max

        self._lidar_manager.update_positions(joints)
        rays = self._lidar_manager.scan()

        return State(
            thetas=self._theta.copy(),
            vels=self._dq.copy(),
            ee_x=ee_x,
            ee_y=ee_y,
            dist_x=dist_x,
            dist_y=dist_y,
            rays=rays,
        )

    def reset(
        self,
        randomize: bool = True,
        n_angle_candidates: int = 36,
        greedy_attempts: int = 5,
        collision_threshold: float = 0.05,
    ) -> State:
        if not randomize:
            if self.cfg.initial_thetas is not None:
                self._theta[:] = np.array(self.cfg.initial_thetas, dtype=float)[: self.n_dof]
            else:
                self._theta[:] = 0.0
            if self.cfg.wrap_angles:
                self._theta = np.array([self.wrap_angle(t) for t in self._theta], dtype=float)
            return self.obs()

        obstacles = self._lidar_manager._obstacles
        candidates = np.linspace(-math.pi, math.pi, n_angle_candidates, endpoint=False)

        for _ in range(greedy_attempts):
            theta = np.zeros(self.n_dof, dtype=float)
            p = self.base.copy()
            cumulative = 0.0
            success = True

            for i, L in enumerate(self.links):
                shuffled = self.rng.permutation(candidates)
                valid = []
                clearances = []

                for delta in shuffled:
                    abs_angle = cumulative + delta
                    p_next = p + np.array([L * math.cos(abs_angle), L * math.sin(abs_angle)])
                    clearance = self._segment_min_clearance(
                        p, p_next, obstacles, collision_threshold
                    )
                    if clearance > 0.0:
                        valid.append(delta)
                    clearances.append((clearance, delta))

                if valid:
                    theta[i] = self.rng.choice(valid)
                else:
                    theta[i] = max(clearances, key=lambda x: x[0])[1]
                    success = False

                cumulative += theta[i]
                p = p + np.array([L * math.cos(cumulative), L * math.sin(cumulative)])

            self._theta = (
                np.array([self.wrap_angle(t) for t in theta], dtype=float)
                if self.cfg.wrap_angles
                else theta
            )
            if success:
                break

  
        self._dq = np.zeros(self.n_dof, dtype=float)
        self._q_target = self._theta.copy()
        return self.obs()

    @staticmethod
    def _segment_min_clearance(
        p1: np.ndarray, p2: np.ndarray, obstacles: list, threshold: float
    ) -> float:
        min_clearance = math.inf
        vec = p2 - p1
        denom = float(np.dot(vec, vec)) + 1e-6
        for obs in obstacles:
            t = float(np.clip(np.dot(obs.center - p1, vec) / denom, 0.0, 1.0))
            dist = float(np.linalg.norm(obs.center - (p1 + t * vec)))
            clearance = dist - obs.radius - threshold
            if clearance < min_clearance:
                min_clearance = clearance
        return min_clearance if obstacles else math.inf

    def set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float).reshape(self.n_dof)
        if self.cfg.wrap_angles:
            theta = np.array([self.wrap_angle(float(t)) for t in theta], dtype=float)
        self._theta = theta

    def physics_step(self, dt: float):
        # 1. Compute the error between current angles and target angles, with proper wrapping
        error = (self._q_target - self._theta + np.pi) % (2 * np.pi) - np.pi
        
        # 2.get gravity vector from physics model
        _, G = self.physics.get_matrices(self._theta)
        
        # 3. PD-control to compute torques, with gravity compensation
        tau = self.Kp * error - self.Kd * self._dq + G
        tau = np.clip(tau, -50, 50) # torque limits for numerical stability
        #print(f"tau: {tau}, error: {error}")
        
        # 4. RK4 update of the robot state using the physics model
        self._theta, self._dq = self.physics.update_rk4(self._theta, self._dq, tau, dt)
        
        # 5. Обязательная нормализация углов
        self._theta = np.array([self.wrap_angle(t) for t in self._theta], dtype=float)

    def step(self, dtheta: np.ndarray) -> Tuple[State, np.ndarray]:
        dtheta = np.asarray(dtheta, dtype=float).reshape(self.n_dof)
        if self.cfg.dtheta_max is not None:
            dtheta = self.clip_dtheta(dtheta, self.cfg.dtheta_max)

        self._q_target += dtheta
        self._q_target = np.array([self.wrap_angle(t) for t in self._q_target], dtype=float)
        
        dt_phys = 0.001
        n_substeps = 10 
        
        for _ in range(n_substeps):
            self.physics_step(dt_phys)

        return self.obs(), dtheta

    @staticmethod
    def wrap_angle(theta: float) -> float:
        return (theta + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def clip_dtheta(dtheta: np.ndarray, max_dtheta: float) -> np.ndarray:
        return np.clip(dtheta, -max_dtheta, max_dtheta)


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
