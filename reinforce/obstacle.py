from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class ObstacleConfig:
    """Конфигурация препятствия"""
    position: Tuple[float, float]
    radius: float
    obstacle_type: str = "circle"  # можно расширить для других форм


class Obstacle:
    """Класс для представления препятствия"""
    
    def __init__(self, config: ObstacleConfig):
        self.position = np.asarray(config.position, dtype=np.float32)
        self.radius = float(config.radius)
        self.type = config.obstacle_type
    
    def get_render_data(self) -> dict:
        """Данные для визуализации"""
        return {
            "position": self.position.copy(),
            "radius": self.radius,
            "type": self.type
        }


class ObstacleManager:
    """Менеджер для работы с несколькими препятствиями"""
    
    def __init__(self, obstacles_config: List[ObstacleConfig]):
        self.obstacles = [Obstacle(cfg) for cfg in obstacles_config]
    
    def get_render_data(self) -> list:
        """Данные для визуализации всех препятствий"""
        return [obs.get_render_data() for obs in self.obstacles]