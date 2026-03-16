from enum import Enum, auto
from dataclasses import dataclass


class Action(Enum):
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5
    REFUEL = 6


@dataclass
class EnvConfig:
    rows: int = 7
    cols: int = 7
    max_fuel: int = 16
    max_steps: int = 50
    gas_station: tuple = (3, 3)
    use_fuel: bool = True


@dataclass
class TrainConfig:
    num_episodes: int = 300_000
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.99997
    epsilon_min: float = 0.01
    save_every: int = 50_000
    progress_every: int = 1000


@dataclass
class GUIConfig:
    fps: int = 5
    model_path: str = "policy/q_table.npy"
