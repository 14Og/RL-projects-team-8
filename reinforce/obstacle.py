from .config import ObstacleConfig
import numpy as np
from dataclasses import dataclass
from typing import Dict
from __future__ import annotations

@dataclass
class Obstacle:
    center: np.ndarray  # shape (2,), [x, y] in pixels
    radius: float  # radius in pixels

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)

    def describe(self) -> Dict:
        return {"center": self.center.copy(), "radius": self.radius}

class ObstacleManager:
    def __init__(self, cfg: ObstacleConfig):
        if cfg.random:
            raise NotImplementedError("Random obstacles TBD")
        if cfg.dynamic:
            raise NotImplementedError("Dynamic obstacles TBD")
        
        self.obstacles = [Obstacle(center=pos, radius=cfg.radius) for pos in cfg.positions]

    def get_render_data(self) -> list:
        return [obs.describe() for obs in self.obstacles]


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
