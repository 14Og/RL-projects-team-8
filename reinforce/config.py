from dataclasses import dataclass, field
from typing import Tuple, Optional, List

@dataclass(frozen=True)
class RobotConfig:
    base_xy: Tuple[float, float] = (400, 400)
    #! num of DoFs is determined by this field
    link_lengths: Tuple[float, ...] = (100, 80, 60) 
    wrap_angles: bool = True
    dtheta_max: Optional[float] = 0.3

@dataclass(frozen=True)
class LidarConfig:
    lidar_joints: bool = True    # attach a lidar at each joint (excluding base)
    lidar_midlinks: bool = False  # attach a lidar at each link midpoint
    num_rays: int = 8             # rays per lidar, uniformly spread over full 360°
    ray_maxlen_px: float = 100.0     # max cast distance; normalized hit distance in [0, 1]

@dataclass
class ObstacleConfig:
        
    @staticmethod
    def _default_obs() -> List:
        return [[200, 400], [600, 400]]
    
    positions: List[List[float]] = field(default_factory=_default_obs)
    radius: float = 60.0
    random: bool = False   # TBD
    dynamic: bool = False  # TBD
    
        
@dataclass
class ModelConfig:
    gamma: float = 0.99
    lr_start: float = 1e-3
    lr_min: float = 1e-5
    baseline_buf_len: int = 200
    grad_clip_norm: float = 1.0
    hidden_sizes: Tuple[int, ...] = (256, 128)
    log_std_min: float = -3.0
    log_std_max: float = -0.5

@dataclass
class RewardConfig:
    progress_scale: float = 0.03
    progress_near_boost: float = 4.0     # extra multiplier when ee is within boost_radius of target
    progress_boost_radius: float = 80.0  # px – distance at which boost starts ramping up
    step_penalty: float = 0.02
    goal_reward: float = 50.0
    fail_penalty: float = 15.0
    joint_velocity_scale: float = 0.1     # penalty on squared joint velocity
    action_delta_scale: float = 0.1       # penalty on squared change in action
    # Lidar-based obstacle avoidance rewards
    obstacle_safety_scale: float = 0.02      # Reward for maintaining distance from obstacles
    obstacle_danger_threshold: float = 0.2    # Lidar reading < 0.3 = danger zone
    obstacle_danger_penalty: float = 0.5      # Penalty scale for being in danger zone
    collision_penalty: float = 25.0            # Heavy penalty for actual collision

@dataclass
class EnvConfig:
    target_xy: Tuple[float, float] = (250, 300.0)
    randomize_target: bool = True
    target_thresh: float = 30.0
    max_steps: int = 300
    forbid_link_target_intersection: bool = True
    target_point_radius: float = 1.0
    min_target_distance_from_ee: float = 0.0  # min dist from initial ee to sampled target
    target_line_of_sight: bool = False         # reject targets with no line-of-sight from base
    
@dataclass
class GUIConfig:
    window_size: Tuple[int, int] = (1600, 800)
    sim_width: int = 800
    plot_update_every: int = 10
    pause_on_done_frames: int = 0
    steps_per_frame: int = 1
    steps_per_frame_no_sim: int = 500
    model_path: str = "policy/best_policy.pt"
    train_episodes: int = 5000
    test_episodes: int = 200


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
