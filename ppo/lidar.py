import math
from typing import List
import numpy as np

from ppo.fast_math import fast_multi_lidar_scan
from .config import LidarConfig
from .obstacle import Obstacle

class LidarManager:
    def __init__(self, cfg: LidarConfig, n_dof: int) -> None:
        self.cfg = cfg
        self.n_dof = n_dof
        self._obstacles: List[Obstacle] = []

        # Считаем общее количество лидаров
        self.n_lidars_count = n_dof * (int(cfg.lidar_joints) + int(cfg.lidar_midlinks))
        
        # Храним позиции всех лидаров в одном массиве [N_lidars, 2]
        self.positions = np.zeros((self.n_lidars_count, 2), dtype=float)

        # Предрассчитываем направления лучей ОДИН раз для всех лидаров
        angles = np.linspace(0, 2 * math.pi, cfg.num_rays, endpoint=False)
        self.ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(float)

    @property
    def n_lidars(self) -> int:
        return self.n_lidars_count

    @property
    def n_rays_total(self) -> int:
        return self.n_lidars_count * self.cfg.num_rays

    def set_obstacles(self, obstacles: List[Obstacle]) -> None:
        self._obstacles = obstacles

    def update_positions(self, joints: np.ndarray) -> None:
        # joints: [4, 2] (база + 3 сустава)
        idx = 0
        if self.cfg.lidar_joints:
            # Устанавливаем позиции лидаров на суставы (1, 2, 3)
            for i in range(1, self.n_dof + 1):
                self.positions[idx] = joints[i]
                idx += 1
        if self.cfg.lidar_midlinks:
            # Устанавливаем позиции лидаров на середины звеньев
            for i in range(self.n_dof):
                self.positions[idx] = (joints[i] + joints[i + 1]) / 2.0
                idx += 1

    def scan(self) -> np.ndarray:
        # Если препятствий нет — возвращаем массив единиц (чистый путь)
        if not self._obstacles:
            return np.ones(self.n_rays_total, dtype=float)

        # Подготавливаем данные препятствий в виде плоских массивов для Numba
        obs_centers = np.array([o.center for o in self._obstacles], dtype=float)
        obs_radii = np.array([o.radius for o in self._obstacles], dtype=float)

        # ВЫЗЫВАЕМ ОПТИМИЗИРОВАННУЮ ФУНКЦИЮ СРАЗУ ДЛЯ ВСЕХ ЛИДАРОВ
        # Она вернет один плоский массив (например, 24 значения)
        return fast_multi_lidar_scan(
            self.positions,      # [3, 2]
            self.ray_dirs,       # [8, 2]
            obs_centers,         # [4, 2]
            obs_radii,           # [4]
            float(self.cfg.ray_maxlen_px)
        )

# Класс Lidar больше не нужен, так как LidarManager делает всё сам через массивы