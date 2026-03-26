from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import ObstacleConfig


@dataclass
class Obstacle:
    center: np.ndarray
    radius: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    phase: float = 0.0
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.origin = np.asarray(self.origin, dtype=float)

    def describe(self) -> Dict:
        return {"center": self.center.copy(), "radius": self.radius, "velocity": self.velocity.copy()}


class ObstacleManager:
    def __init__(self, cfg: ObstacleConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.cfg = cfg
        self._rng = rng or np.random.default_rng()
        self._base_positions = [np.asarray(pos, dtype=float) for pos in cfg.positions]
        self._t: float = 0.0
        self.obstacles: List[Obstacle] = [
            Obstacle(center=bp.copy(), radius=cfg.radius, origin=bp.copy())
            for bp in self._base_positions
        ]

    def randomize(self) -> None:
        self._t = 0.0
        jr = self.cfg.jitter_radius
        for obs, bp in zip(self.obstacles, self._base_positions):
            if self.cfg.random and jr > 0:
                angle = self._rng.uniform(0, 2 * np.pi)
                r = jr * np.sqrt(self._rng.uniform())
                obs.origin = bp + np.array([r * np.cos(angle), r * np.sin(angle)])
            else:
                obs.origin = bp.copy()

            # Random phase so obstacles don't all move in sync
            obs.phase = self._rng.uniform(0, 2 * np.pi) if self.cfg.dynamic else 0.0
            obs.center = obs.origin.copy()
            obs.velocity = np.zeros(2)

    def update(self, dt: float) -> None:
        if not self.cfg.dynamic:
            return
        self._t += dt
        a = self.cfg.ellipse_a
        b = self.cfg.ellipse_b
        w = self.cfg.omega
        for obs in self.obstacles:
            phi = obs.phase
            obs.center[0] = obs.origin[0] + a * np.cos(w * self._t + phi)
            obs.center[1] = obs.origin[1] + b * np.sin(w * self._t + phi)
            # velocity = time derivative of position
            obs.velocity[0] = -a * w * np.sin(w * self._t + phi)
            obs.velocity[1] =  b * w * np.cos(w * self._t + phi)

    def get_render_data(self) -> List[Dict]:
        return [obs.describe() for obs in self.obstacles]


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
