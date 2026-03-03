from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import ObstacleConfig


@dataclass
class Obstacle:
    center: np.ndarray  # shape (2,), [x, y] in pixels
    radius: float       # radius in pixels

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)

    def describe(self) -> Dict:
        return {"center": self.center.copy(), "radius": self.radius}


class ObstacleManager:
    def __init__(self, cfg: ObstacleConfig, rng: np.random.Generator | None = None) -> None:
        if cfg.dynamic:
            raise NotImplementedError("Dynamic obstacles TBD")

        self.cfg = cfg
        self._rng = rng or np.random.default_rng()
        self._base_positions = [np.asarray(pos, dtype=float) for pos in cfg.positions]
        self.obstacles: List[Obstacle] = [
            Obstacle(center=bp.copy(), radius=cfg.radius)
            for bp in self._base_positions
        ]

    def randomize(self) -> None:
        """Jitter each obstacle around its base position within jitter_radius (only if cfg.random)."""
        if not self.cfg.random:
            return
        jr = self.cfg.jitter_radius
        if jr <= 0:
            return
        for obs, bp in zip(self.obstacles, self._base_positions):
            angle = self._rng.uniform(0, 2 * np.pi)
            r = jr * np.sqrt(self._rng.uniform())
            obs.center = bp + np.array([r * np.cos(angle), r * np.sin(angle)])

    def get_render_data(self) -> List[Dict]:
        return [obs.describe() for obs in self.obstacles]


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
