from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import numpy as np


@dataclass(frozen=True)
class RobotConfig:

    @staticmethod
    def theta_default() -> Tuple:
        return (np.pi, np.pi, np.pi)

    base_xy: Tuple[float, float] = (400, 600)
    link_lengths: Tuple[float, ...] = (100, 70, 40)
    masses: Tuple[float, ...] = (1.0, 0.7, 0.4)
    Kp: Tuple[float, ...] = (40.0, 30.0, 20.0)
    Kd: Tuple[float, ...] = (10.0, 5, 3)
    wrap_angles: bool = True
    dtheta_max: Optional[float] = 0.2
    initial_thetas: Optional[Tuple[float, ...]] = field(default_factory=theta_default)
    randomize_theta: bool = True


@dataclass(frozen=True)
class LidarConfig:
    lidar_joints: bool = True
    lidar_midlinks: bool = False
    num_rays: int = 8
    ray_maxlen_px: float = 50.0


@dataclass
class ObstacleConfig:

    @staticmethod
    def _default_obs() -> List:
        return [[200, 600]]#, [600, 600], [400, 800], [400, 400]]

    positions: List[List[float]] = field(default_factory=_default_obs)
    radius: float = 50.0
    jitter_radius: float = 40.0
    random: bool = True
    dynamic: bool = False


@dataclass
class ModelConfig:
    gamma: float = 0.97
    lr_start: float = 3e-4
    lr_min: float = 1e-6
    grad_clip_norm: float = 1.0
    hidden_sizes: Tuple[int, ...] = (256, 128)
    log_std_min: float = -3.0
    log_std_max: float = -1.0
    entropy_coef: float = 0.01
    target_kl: float = 0.015
    clip_epsilon: float = 0.15
    ppo_epochs: int = 10
    mini_batch_size: int = 256
    batch_size_limit: int = 2048


@dataclass
class RewardConfig:
    progress_scale: float = 0.5
    progress_near_boost: float = 3.0
    progress_boost_radius: float = 80.0
    step_penalty: float = 0.005
    goal_reward: float = 100.0
    fail_penalty: float = 15.0
    obstacle_danger_threshold: float = 0.4
    obstacle_danger_penalty: float = 0.2
    collision_penalty: float = 30.0
    stagnation_window: int = 15
    stagnation_thresh: float = 1.0


@dataclass
class EnvConfig:
    target_xy: Tuple[float, float] = (250, 300.0)
    randomize_target: bool = True
    target_thresh: float = 30.0
    max_steps: int = 400
    forbid_link_target_intersection: bool = True
    target_point_radius: float = 1.0
    min_target_distance_from_ee: float = 0.0
    target_line_of_sight: bool = False


@dataclass
class GUIConfig:
    window_size: Tuple[int, int] = (2000, 1000)
    sim_width: int = 800
    plot_update_every: int = 2
    pause_on_done_frames: int = 0
    steps_per_frame: int = 1
    steps_per_frame_no_sim: int = 1000
    model_path: str = "policy/best_policy.pt"
    train_episodes: int = 3000
    test_episodes: int = 500


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
