import argparse

from taxi.config import EnvConfig, TrainConfig, GUIConfig


def parse_args():
    p = argparse.ArgumentParser(description="Taxi 7x7 Q-Learning")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--train",     action="store_true", help="Train a new Q-table.")
    g.add_argument("--test",      action="store_true", help="Run a trained agent.")
    p.add_argument("--no-sim",     action="store_true", help="Headless mode (no pygame window).")
    p.add_argument("--no-fuel",    action="store_true", help="Disable fuel mechanic (smaller state space, 6 actions).")
    p.add_argument("--model-path", type=str, nargs="+", default=None,
                   help="Q-table .npy file(s). Multiple paths for checkpoint comparison.")
    p.add_argument("--episodes",   type=int, default=None, help="Override number of training episodes.")
    return p.parse_args()


def main():
    args = parse_args()

    env_cfg   = EnvConfig()
    train_cfg = TrainConfig()
    gui_cfg   = GUIConfig()

    if args.no_fuel:
        env_cfg.use_fuel = False
    if args.model_path:
        gui_cfg.model_path = args.model_path[0]
    if args.episodes:
        train_cfg.num_episodes = args.episodes
        
    from taxi.runner import Runner

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
        runner.test()


if __name__ == "__main__":
    main()
