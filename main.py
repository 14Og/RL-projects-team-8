import argparse

from taxi.config import EnvConfig, TrainConfig, GUIConfig


def parse_args():
    p = argparse.ArgumentParser(description="Taxi 7x7 Q-Learning")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--train", action="store_true", help="Train policy, save, then run test.")
    g.add_argument(
        "--test", action="store_true", help="Run test only (loads model from --model-path)."
    )

    p.add_argument(
        "--no-sim", action="store_true",
        help="Headless mode: no pygame window, live matplotlib plots only."
    )
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--train-episodes", type=int, default=None)
    p.add_argument("--test-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    # Fine-tuning: load pre-trained weights before starting training
    p.add_argument(
        "--finetune", type=str, default=None, metavar="WEIGHTS_PATH",
        help="Load policy weights from this file before training (e.g. policy/best_policy_const.pt)."
    )

    # Curriculum flags: override config defaults from the command line
    p.add_argument("--randomize-target", action="store_true",
                   help="Randomise the target position each episode.")
    p.add_argument("--randomize-theta", action="store_true",
                   help="Randomise the initial joint angles each episode.")

    return p.parse_args()


def _build_model(args, robot_cfg, lidar_cfg, model_cfg, gui_cfg, env_cfg, obstacle_cfg):
    """Construct a PPO Model."""
    from ppo.runner import compute_obs_dim
    from ppo.model_actor_critic_ppo import Model

    obs_dim = compute_obs_dim(
        robot_cfg, lidar_cfg,
        obs_mode=env_cfg.obs_mode,
        n_obstacles=len(obstacle_cfg.positions),
    )
    return Model(
        obs_dim=obs_dim,
        act_dim=len(robot_cfg.link_lengths),
        cfg=model_cfg,
        max_steps=env_cfg.max_steps,
        action_limit=robot_cfg.tau_limits,
        train_episodes=gui_cfg.train_episodes,
    )


def main() -> None:

    args = parse_args()

    robot_cfg    = RobotConfig(randomize_theta=args.randomize_theta)
    env_cfg      = EnvConfig(randomize_target=args.randomize_target)
    rew_cfg      = RewardConfig()
    gui_cfg      = GUIConfig()
    model_cfg    = ModelConfig()
    lidar_cfg    = LidarConfig()
    obstacle_cfg = ObstacleConfig()

    gui_cfg.train_episodes = args.train_episodes or gui_cfg.train_episodes
    gui_cfg.test_episodes  = args.test_episodes  or gui_cfg.test_episodes
    gui_cfg.model_path     = args.model_path     or gui_cfg.model_path

    model = _build_model(args, robot_cfg, lidar_cfg, model_cfg, gui_cfg, env_cfg, obstacle_cfg)

    if args.finetune:
        print(f"[finetune] loading weights from {args.finetune}", flush=True)
        model.load(args.finetune, load_optimizer=False)
        print("[finetune] weights loaded — starting training with randomized curriculum", flush=True)

    from ppo.runner import Runner

    if args.no_sim:
        # ----------------------------------------------------------------
        # Headless path — no pygame imported
        # ----------------------------------------------------------------
        runner = Runner(
            env_cfg=env_cfg,
            reward_cfg=rew_cfg,
            robot_cfg=robot_cfg,
            lidar_cfg=lidar_cfg,
            obstacle_cfg=obstacle_cfg,
            gui_cfg=gui_cfg,
            model=model,
            seed=int(args.seed),
            # pygame_renderer=None  (default → headless)
        )

    if args.no_sim or args.train:
        runner = Runner(env_cfg=env_cfg, train_cfg=train_cfg, gui_cfg=gui_cfg)
        if args.train:
            runner.train()
        else:
            runner.test()
    else:
        from taxi.gui import PygameRenderer
        renderer = PygameRenderer(gui_cfg=gui_cfg)
        runner = Runner(
            env_cfg=env_cfg,
            train_cfg=train_cfg,
            gui_cfg=gui_cfg,
            renderer=renderer,
        )
        runner.test(args.model_path[0] if args.model_path else None)


if __name__ == "__main__":
    main()
