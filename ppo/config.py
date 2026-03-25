from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import numpy as np


@dataclass(frozen=True)
class RobotConfig:

    @staticmethod
    def theta_default() -> Tuple:
        return (np.pi, np.pi, np.pi)

    base_xy: Tuple[float, float] = (400, 600)
    link_lengths: Tuple[float, ...] = (90, 70, 40)
    masses: Tuple[float, ...] = (1.0, 0.7, 0.6)
    wrap_angles: bool = True
    tau_limits: Tuple[float, ...] = (12.0, 8.0, 8.0)
    initial_thetas: Optional[Tuple[float, ...]] = field(default_factory=theta_default)
    randomize_theta: bool = False
    theta_jitter: float = 0.15   # rad, applied when randomize_theta=False


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
        return [[200, 600], [600, 600]]#, [400, 800], [400, 400]]

    positions: List[List[float]] = field(default_factory=_default_obs)
    radius: float = 30.0
    jitter_radius: float = 30.0
    random: bool = True
    dynamic: bool = True
    ellipse_a: float = 80.0   # semi-axis x (px)
    ellipse_b: float = 60.0   # semi-axis y (px)
    omega: float = 0.1        # angular frequency (rad/s)


@dataclass
class ModelConfig:
    gamma: float = 0.97
    gae_lambda: float = 0.95
    lr_start: float = 3e-4
    lr_min: float = 1e-6
    grad_clip_norm: float = 1.0
    hidden_sizes: Tuple[int, ...] = (256, 128)
    log_std_min: float = -3
    log_std_max: float = -0.7
    damping: float = 1
    entropy_coef: float = 0.002
    value_loss_coef: float = 0.1   # coefficient on value loss; kept at 0.5 because TD targets are normalized
    target_kl: float = 0.15
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    mini_batch_size: int = 512
    batch_size_limit: int = 2048


@dataclass
class RewardConfig:
    progress_scale: float = 0.9
    progress_near_boost: float = 0.2
    progress_boost_radius: float = 60.0
    step_penalty: float = 0.08
    goal_reward: float = 80.0
    fail_penalty: float = 40.0
    obstacle_danger_threshold: float = 0.7
    obstacle_danger_penalty: float = 2
    collision_penalty: float = 60.0
    stagnation_window: int = 100
    stagnation_thresh: float = 0
    vel_penalty: float = 0.08
    torque_penalty: float = 0.05


@dataclass
class EnvConfig:
    target_xy: Tuple[float, float] = (700, 600.0)
    randomize_target: bool = False
    target_thresh: float = 30.0
    max_steps: int = 600
    forbid_link_target_intersection: bool = False  # too many spurious failures early in training
    target_point_radius: float = 1.0
    min_target_distance_from_ee: float = 0.0
    target_line_of_sight: bool = False
    # Curriculum observation modes:
    #   "base"    — 13 dims: sin/cos/ee/dist/vels (no lidar, no obstacles)
    #   "static"  — 41 dims: base + lidar(24) + obstacle rel positions (2 per obs)
    #   "dynamic" — 45 dims: base + lidar(24) + obstacle rel pos + velocity (4 per obs)
    obs_mode: str = "dynamic"


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
