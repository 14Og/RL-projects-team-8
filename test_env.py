#!/usr/bin/env python3
import numpy as np
from reinforce.env import Environment
from reinforce.config import EnvConfig, RewardConfig, RobotConfig, ModelConfig, GUIConfig
from reinforce.model import Model

if __name__ == "__main__":
    # Configs
    env_cfg = EnvConfig(target_xy=[1.5, 1.5])
    reward_cfg = RewardConfig()
    robot_cfg = RobotConfig()
    model_cfg = ModelConfig()
    gui_cfg = GUIConfig()
    
    # Model parameters
    obs_dim = 58  # 10 base + 48 lidar (3 joints * 16 rays)
    act_dim = 3   # 3 joints
    action_limit = 0.1
    train_episodes = gui_cfg.train_episodes
    
    # Create model
    model = Model(
        obs_dim=obs_dim,
        act_dim=act_dim,
        cfg=model_cfg,
        action_limit=action_limit,
        train_episodes=train_episodes,
    )
    
    # Obstacles
    obstacles = np.array([
        [1.2, 1.0, 0.3], 
        [-0.8, 1.2, 0.3],
        [0.5, -1.2, 0.4]
    ])
    
    # Environment
    env = Environment(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        model=model,
        obstacles=obstacles,
    )
    
    # Reset and test
    env.reset_episode(train=False)
    print("✓ Environment created successfully")
    print(f"✓ State size: {env.reset_episode(train=False).shape}")
    print(f"✓ Obstacles: {len(obstacles)} detected")
    print("✓ Ready for testing")