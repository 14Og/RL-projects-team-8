from dataclasses import dataclass, field
from typing import Tuple, Optional, List

@dataclass(frozen=True)
class RobotConfig:
    base_xy: Tuple[float, float] = (400, 600)
    #! num of DoFs is determined by this field
    link_lengths: Tuple[float, ...] = (100, 70, 40) 
    wrap_angles: bool = True
    dtheta_max: Optional[float] = 0.3

@dataclass(frozen=True)
class LidarConfig:
    lidar_joints: bool = True    # attach a lidar at each joint (excluding base)
    lidar_midlinks: bool = False  # attach a lidar at each link midpoint
    num_rays: int = 8             # rays per lidar, uniformly spread over full 360°
    ray_maxlen_px: float = 50.0     # max cast distance; normalized hit distance in [0, 1]

@dataclass
class ObstacleConfig:
        
    @staticmethod
    def _default_obs() -> List:
        return [[200, 600], [600, 600], [400, 800], [400, 400]]
    
    positions: List[List[float]] = field(default_factory=_default_obs)
    radius: float = 50.0
    random: bool = False   # TBD
    dynamic: bool = False  # TBD
    
        
@dataclass
class ModelConfig:
    gamma: float = 0.97
    lr_start: float = 3e-4
    lr_min: float = 1e-5
    baseline_buf_len: int = 200
    grad_clip_norm: float = 1.0
    hidden_sizes: Tuple[int, ...] = (256, 128)
    log_std_min: float = -2.0
    log_std_max: float = 0.5
    entropy_coef: float = 0.01            # entropy bonus coefficient
    target_kl: float = 0.015              # KL early stopping threshold
    clip_epsilon: float = 0.15            # PPO clip range
    ppo_epochs: int = 10                  # max PPO epochs per update
    mini_batch_size: int = 256            # mini-batch size within each PPO epoch
    n_ppo_updates: int = 500              # expected total PPO updates (for LR scheduler)

@dataclass
class RewardConfig:
    progress_scale: float = 0.15
    progress_near_boost: float = 3.0      # extra multiplier when ee is within boost_radius of target
    progress_boost_radius: float = 80.0   # px – distance at which boost starts ramping up
    step_penalty: float = 0.005
    goal_reward: float = 15.0
    fail_penalty: float = 5.0
    joint_velocity_scale: float = 0.02     # penalty on squared joint velocity
    action_delta_scale: float = 0.02       # penalty on squared change in action
    # Lidar-based obstacle avoidance penalty (per-lidar smoothed)
    obstacle_danger_threshold: float = 0.15    # Lidar reading below this = danger zone
    obstacle_danger_penalty: float = 0.15      # Penalty scale per lidar in danger zone
    collision_penalty: float = 10.0            # Heavy penalty for actual collision
    # Stagnation penalty — punish the robot for not making progress
    stagnation_window: int = 15               # number of steps to check for progress
    stagnation_thresh: float = 2.0            # min distance change over window to not be "stuck"
    stagnation_penalty: float = 0.05           # penalty per step while stagnating (ramps up)

@dataclass
class EnvConfig:
    target_xy: Tuple[float, float] = (250, 300.0)
    randomize_target: bool = True
    target_thresh: float = 30.0
    max_steps: int = 200
    stagnation_max: int = 40                  # terminate episode after this many stagnant steps
    forbid_link_target_intersection: bool = True
    target_point_radius: float = 1.0
    min_target_distance_from_ee: float = 0.0  # min dist from initial ee to sampled target
    target_line_of_sight: bool = False         # reject targets with no line-of-sight from base
    
@dataclass
class GUIConfig:
    window_size: Tuple[int, int] = (2000, 1200)
    sim_width: int = 800
    plot_update_every: int = 10
    pause_on_done_frames: int = 0
    steps_per_frame: int = 10
    steps_per_frame_no_sim: int = 1000
    model_path: str = "policy/best_policy.pt"
    train_episodes: int = 5000
    test_episodes: int = 300


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
