import argparse as ap

from ppo.config import RobotConfig, EnvConfig, RewardConfig, GUIConfig, ModelConfig, LidarConfig, ObstacleConfig

def parse_args() -> ap.Namespace:
    p = ap.ArgumentParser()
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
    return p.parse_args()


def _build_model(args, robot_cfg, lidar_cfg, model_cfg, gui_cfg, env_cfg):
    """Construct a PPO Model."""
    from ppo.runner import compute_obs_dim
    from ppo.model_actor_critic import Model

    obs_dim = compute_obs_dim(robot_cfg, lidar_cfg)
    return Model(
        obs_dim=obs_dim,
        act_dim=len(robot_cfg.link_lengths),
        cfg=model_cfg,
        max_steps=env_cfg.max_steps,
        action_limit=robot_cfg.dtheta_max,
        train_episodes=gui_cfg.train_episodes,
    )


def main() -> None:

    args = parse_args()

    robot_cfg    = RobotConfig()
    env_cfg      = EnvConfig()
    rew_cfg      = RewardConfig()
    gui_cfg      = GUIConfig()
    model_cfg    = ModelConfig()
    lidar_cfg    = LidarConfig()
    obstacle_cfg = ObstacleConfig()

    gui_cfg.train_episodes = args.train_episodes or gui_cfg.train_episodes
    gui_cfg.test_episodes  = args.test_episodes  or gui_cfg.test_episodes
    gui_cfg.model_path     = args.model_path     or gui_cfg.model_path

    model = _build_model(args, robot_cfg, lidar_cfg, model_cfg, gui_cfg, env_cfg)

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

        if args.train:
            runner.train()
            runner.save_model()
        else:
            runner.test(model_path=gui_cfg.model_path)

        import matplotlib.pyplot as plt
        plt.show(block=True)   # keep the final figure open until user closes it

    else:
        # ----------------------------------------------------------------
        # Pygame path — PygameRenderer injected into the same Runner
        # ----------------------------------------------------------------
        from ppo.gui import PygameRenderer

        if args.test:
            model.load(gui_cfg.model_path, load_optimizer=False)
            model.set_eval_mode()

        renderer = PygameRenderer(gui_cfg=gui_cfg, env_cfg=env_cfg)
        runner = Runner(
            env_cfg=env_cfg,
            reward_cfg=rew_cfg,
            robot_cfg=robot_cfg,
            lidar_cfg=lidar_cfg,
            obstacle_cfg=obstacle_cfg,
            gui_cfg=gui_cfg,
            model=model,
            seed=int(args.seed),
            pygame_renderer=renderer,
        )

        if args.train:
            runner.train()
        else:
            runner.test()  # test only
            renderer.close()


if __name__ == "__main__":
    main()
