from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class RobotConfig:
    base_xy: Tuple[float, float] = (400, 400)
    #! num of DoFs is determined by this field
    link_lengths: Tuple[float, ...] = (100, 80, 60) 
    wrap_angles: bool = True
    dtheta_max: Optional[float] = 0.1

@dataclass(frozen=True)
class LidarConfig:
    lidar_joints: bool = True    # attach a lidar at each joint (excluding base)
    lidar_midlinks: bool = False  # attach a lidar at each link midpoint
    num_rays: int = 8             # rays per lidar, uniformly spread over full 360°
    ray_maxlen_px: float = 100.0     # max cast distance; normalized hit distance in [0, 1]

@dataclass
class ModelConfig:
    gamma: float = 0.99
    lr_start: float = 1e-4
    lr_min: float = 1e-5
    baseline_buf_len: int = 200
    grad_clip_norm: float = 1.0
    hidden_sizes: Tuple[int, ...] = (128, 128)
    log_std_min: float = -3.0
    log_std_max: float = -0.5

@dataclass
class RewardConfig:
    progress_scale: float = 0.03
    step_penalty: float = 0.01
    goal_reward: float = 15.0
    fail_penalty: float = 5.0
    action_l2_scale: float = 0.0
    action_delta_scale: float = 0.0

@dataclass
class EnvConfig:
    target_xy: Tuple[float, float] = (250, 300.0)
    randomize_target: bool = True
    target_thresh: float = 30.0
    max_steps: int = 200
    forbid_link_target_intersection: bool = True
    target_point_radius: float = 1.0
    use_abs_dist: bool = False
    normalize_dist: bool = True
    dist_scale: float = 300.0
    
@dataclass
class GUIConfig:
    window_size: Tuple[int, int] = (1600, 800)
    sim_width: int = 800
    plot_update_every: int = 10
    pause_on_done_frames: int = 0
    steps_per_frame: int = 2
    steps_per_frame_no_sim: int = 500
    model_path: str = "policy/best_policy.pt"
    train_episodes: int = 5000
    test_episodes: int = 200


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
