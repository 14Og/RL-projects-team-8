from __future__ import annotations
from os import path, makedirs
from typing import Optional, cast
import numpy as np
import matplotlib.pyplot as plt
from .config import EnvConfig, TrainConfig, GUIConfig
from .env import TaxiEnv
from .agent import QAgent


class Runner:
    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        train_cfg: TrainConfig,
        gui_cfg: GUIConfig,
        renderer=None,
    ):
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.gui_cfg = gui_cfg
        self.renderer = renderer
        from gymnasium.spaces import Discrete

        self.env = TaxiEnv(cfg=env_cfg)
        self.agent = QAgent(
            num_states=int(cast(Discrete, self.env.observation_space).n),
            num_actions=int(cast(Discrete, self.env.action_space).n),
            cfg=train_cfg,
        )

    def train(self):
        cfg = self.train_cfg
        all_rewards = []
        all_success = []
        fig, (ax_reward, ax_success) = self._init_plot()
        print("Starting training...")
        for episode in range(1, cfg.num_episodes + 1):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.learn(state, action, reward, next_state, terminated)
                state = next_state
                episode_reward += reward
            all_rewards.append(episode_reward)
            all_success.append(1 if reward == 60 else 0)
            self.agent.decay_epsilon()
            if episode % cfg.progress_every == 0 or episode == 1:
                recent_sr = np.mean(all_success[-2000:]) * 100
                avg_r = np.mean(all_rewards[-1000:])
                print(
                    f"Episode {episode:>7} | "
                    f"Avg reward: {avg_r:>6.1f} | "
                    f"Success: {recent_sr:>5.1f}% | "
                    f": {self.agent.epsilon:.4f}"
                )
                self._update_plot(fig, ax_reward, ax_success, all_rewards, all_success)

            if episode % cfg.save_every == 0:
                self.agent.save(self._save_path(f"_step_{episode}"))

            if not plt.fignum_exists(fig.number):
                print("Plotbreak window closed. Interrupting training.")
                return

        self.agent.save(self._save_path("_final"))
        self._update_plot(fig, ax_reward, ax_success, all_rewards, all_success)
        plt.ioff()
        plt.show()

    def test(self, model_path: Optional[str] = None):
        load_path = model_path or self._save_path("_final")
        self.agent.load(load_path)

        if self.renderer is None:
            # headless test: print stats over 100 episodes
            successes, rewards = [], []
            for _ in range(100):
                state, _ = self.env.reset()
                ep_r, done = 0, False
                while not done:
                    action = self.agent.choose_action(state)
                    state, r, terminated, truncated, _ = self.env.step(action)
                    ep_r += r
                    done = terminated or truncated
                rewards.append(ep_r)
                successes.append(1 if r == 60 else 0)
            print(
                f"Test (100 episodes): "
                f"avg reward={np.mean(rewards):.1f}, "
                f"success={np.mean(successes)*100:.1f}%"
            )
        else:
            self.renderer.run(self.env, self.agent)

    def _save_path(self, suffix: str) -> str:
        """Derive a checkpoint path from gui_cfg.model_path.

        model_path = "policy/q_table.npy"
        suffix = "_step_10000"  → "policy/q_table_step_10000.npy"
        suffix = "_final"       → "policy/q_table_final.npy"
        """
        base, ext = path.splitext(self.gui_cfg.model_path)
        save_dir = path.dirname(base)
        if not self.env_cfg.use_fuel:
            suffix = f"_nofuel{suffix}"
        if save_dir:
            makedirs(save_dir, exist_ok=True)
        return f"{base}{suffix}{ext}"

    # ------------------------------------------------------------------
    # Live plotting
    # ------------------------------------------------------------------

    @staticmethod
    def _init_plot():
        plt.ion()
        fig, (ax_r, ax_s) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax_r.set_ylabel("Total reward")
        ax_r.set_title("Taxi 7×7 – Q-Learning")
        ax_r.grid(True)

        ax_s.set_xlabel("Episode")
        ax_s.set_ylabel("Success rate (%)")
        ax_s.set_ylim(-5, 105)
        ax_s.grid(True)

        fig.tight_layout()
        fig.show()
        return fig, (ax_r, ax_s)

    @staticmethod
    def _update_plot(fig, ax_reward, ax_success, rewards, successes, window=1000):
        # --- reward ---
        ax_reward.clear()
        ax_reward.set_ylabel(f"Total reward (window={window})")
        ax_reward.set_title("Taxi 7×7 – Q-Learning")
        ax_reward.grid(True)

        ax_reward.plot(rewards, alpha=0.15, color="blue")
        if len(rewards) >= window:
            avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax_reward.plot(range(window - 1, len(rewards)), avg, color="red", linewidth=2)

        # --- success rate ---
        ax_success.clear()
        ax_success.set_xlabel("Episode")
        ax_success.set_ylabel(f"Success rate % (window={window})")
        ax_success.set_ylim(-5, 105)
        ax_success.grid(True)

        if len(successes) >= window:
            sr = np.convolve(successes, np.ones(window) / window, mode="valid") * 100
            ax_success.plot(
                range(window - 1, len(successes)),
                sr,
                color="green",
                linewidth=2,
            )

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
