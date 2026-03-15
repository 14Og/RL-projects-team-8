from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

from .config import (
    EnvConfig,
    GUIConfig,
    LidarConfig,
    ObstacleConfig,
    RewardConfig,
    RobotConfig,
)
from .env import Environment
from .model_actor_critic import Model


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def compute_obs_dim(robot_cfg: RobotConfig, lidar_cfg: LidarConfig) -> int:
    """Compute flat observation dimension from config (no robot instantiation needed).

    Layout mirrors ``State.__array__``:
        [sin(θi), cos(θi)] × n_dof   →  2 * n_dof
        [ee_x, ee_y, dist_x, dist_y] →  4
        lidar rays                    →  n_lidars * num_rays
    """
    n_dof = len(robot_cfg.link_lengths)
    n_lidars = n_dof * (int(lidar_cfg.lidar_joints) + int(lidar_cfg.lidar_midlinks))
    return 2 * n_dof + 4 + n_lidars * lidar_cfg.num_rays + n_dof


class Runner:
    """Unified training/evaluation loop.

    Parameters
    ----------
    model :
        Pre-built Model instance.  The runner never constructs the
        model so the caller controls obs_dim, architecture, checkpoints, etc.
    pygame_renderer :
        ``None`` (default) → headless mode.
        Pass a ``PygameRenderer`` instance (from ``reinforce.gui``) to get the
        full interactive pygame window.
    plot_update_every :
        Redraw the live matplotlib figure every N completed episodes.
    progress_every :
        Print a one-line console summary every N episodes.  0 = silent.
    """

    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        robot_cfg: RobotConfig,
        lidar_cfg: LidarConfig,
        obstacle_cfg: ObstacleConfig,
        gui_cfg: GUIConfig,
        model: Model,
        seed: int = 42,
        pygame_renderer: Optional[Any] = None,
        progress_every: int = 100,
    ) -> None:
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.robot_cfg = robot_cfg
        self.lidar_cfg = lidar_cfg
        self.obstacle_cfg = obstacle_cfg
        self.gui_cfg = gui_cfg
        self.model = model
        self.seed = seed
        self.pygame_renderer = pygame_renderer
        self.progress_every = int(progress_every)

        self._fig: Optional[Any] = None
        self._axes: Optional[Tuple[Any, Any, Any]] = None
        self._win = 50  # windowed success-rate width

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Full train → save → test cycle.  Returns ``(train_metrics, test_metrics)``."""
        ok = self._run_phase("train", self.gui_cfg.train_episodes)
        if ok:
            self._save_model()
        self._run_phase("test", self.gui_cfg.test_episodes)
        if self.pygame_renderer is not None:
            self.pygame_renderer.close()
        return self.model.get_train_metrics(), self.model.get_test_metrics()

    def train(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """Run training phase and return train metrics."""
        self._run_phase("train", int(episodes or self.gui_cfg.train_episodes))
        return self.model.get_train_metrics()

    def test(
        self,
        episodes: Optional[int] = None,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run test phase and return test metrics.

        If *model_path* is provided the weights are reloaded first — useful when
        calling ``test()`` standalone after a separate training run.
        """
        if model_path is not None:
            self.model.load(model_path, load_optimizer=False)
        self._run_phase("test", int(episodes or self.gui_cfg.test_episodes))
        return self.model.get_test_metrics()

    def save_model(self, path: Optional[str] = None) -> None:
        self._save_model(path)

    # ------------------------------------------------------------------
    # Unified episode loop
    # ------------------------------------------------------------------

    def _run_phase(self, mode: str, n_episodes: int) -> bool:
        """Run one phase (train or test).  Returns False if the user quit pygame."""
        train = mode == "train"
        env = self._make_env(train=train)
        env.reset_episode(train=train)
        self._setup_live_figure(mode)

        episode_count = 0
        paused = False
        pause_frames_left = 0

        # Pygame batches multiple env steps per rendered frame for speed.
        # Headless mode also batches to avoid per-step Python loop overhead.
        spf = (
            self.gui_cfg.steps_per_frame
            if self.pygame_renderer is not None
            else self.gui_cfg.steps_per_frame_no_sim
        )

        while episode_count < n_episodes:

            # ---- pygame outer event poll ----
            if self.pygame_renderer is not None:
                quit_req, paused = self.pygame_renderer.handle_events(paused)
                if quit_req:
                    return False

            # ---- pause handling ----
            if paused or pause_frames_left > 0:
                if pause_frames_left > 0:
                    pause_frames_left -= 1
                if self.pygame_renderer is not None:
                    self.pygame_renderer.render(
                        env.get_render_data(), mode, episode_count, n_episodes
                    )
                continue

            # ---- inner step loop ----
            for _ in range(spf):
                # Poll events inside the inner loop too so the window stays
                # responsive at high step-rates (large spf in pygame mode).
                if self.pygame_renderer is not None:
                    quit_req, paused = self.pygame_renderer.handle_events(paused)
                    if quit_req:
                        return False
                    if paused:
                        break

                _, _, done, info = env.step()

                if done and not info.get("needs_reset", False):
                    episode_count += 1

                    if self.progress_every > 0 and episode_count % self.progress_every == 0:
                        self._print_progress(mode, episode_count, n_episodes)

                    if (
                        episode_count % self.gui_cfg.plot_update_every == 0
                        or episode_count == n_episodes
                    ):
                        self._update_live_plot(mode)

                    if self.pygame_renderer is not None and self.gui_cfg.pause_on_done_frames > 0:
                        pause_frames_left = self.gui_cfg.pause_on_done_frames
                        break

                    if episode_count >= n_episodes:
                        break

            # ---- render frame (pygame only) ----
            if self.pygame_renderer is not None:
                self.pygame_renderer.render(env.get_render_data(), mode, episode_count, n_episodes)

        # Ensure the final episode batch is reflected in the plot.
        self._update_live_plot(mode)
        return True

    # ------------------------------------------------------------------
    # Live plot — shared drawing logic for both headless and pygame modes
    # ------------------------------------------------------------------

    def _setup_live_figure(self, mode: str) -> None:
        import matplotlib.pyplot as plt

        if self._fig is not None:
            try:
                plt.close(self._fig)
            except Exception:
                pass

        if self.pygame_renderer is not None:
            # Non-interactive Agg figure.  PygameRenderer will convert it to a
            # pygame surface on demand.
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            plot_w = self.gui_cfg.window_size[0] - self.gui_cfg.sim_width
            plot_h = self.gui_cfg.window_size[1]
            self._fig = Figure(figsize=(plot_w / 100.0, plot_h / 100.0), dpi=100)
            FigureCanvasAgg(self._fig)  # attaches Agg canvas
        else:
            # Interactive figure: shows a live window in scripts, or renders
            # inline in Jupyter notebooks.
            if not _is_notebook():
                plt.ion()
            self._fig = plt.figure(figsize=(12, 12))

        if mode == "train":
            # 4 rows × 2 cols; last row spans both columns for success rate
            from matplotlib.gridspec import GridSpec

            gs = GridSpec(4, 2, figure=self._fig, hspace=0.55, wspace=0.35)
            self._axes = (
                self._fig.add_subplot(gs[0, 0]),  # ax1: total reward
                self._fig.add_subplot(gs[0, 1]),  # ax2: steps/episode
                self._fig.add_subplot(gs[1, 0]),  # ax3: KL divergence
                self._fig.add_subplot(gs[1, 1]),  # ax4: sigma
                self._fig.add_subplot(gs[2, 0]),  # ax5: collision rate
                self._fig.add_subplot(gs[2, 1]),  # ax6: entropy
                self._fig.add_subplot(gs[3, :]),  # ax7: success rate (full width)
            )
        else:
            self._axes = (
                self._fig.add_subplot(311),
                self._fig.add_subplot(312),
                self._fig.add_subplot(313),
            )

        if self.pygame_renderer is not None:
            # Produce an initial (empty) surface so the window isn't blank.
            self.pygame_renderer.notify_figure_updated(self._fig)
        else:
            if not _is_notebook():
                plt.show(block=False)

        self._fig.suptitle(f"{mode} metrics", fontsize=18, y=0.995)
        self._fig.tight_layout(pad=0.4, rect=[0.01, 0.01, 0.99, 0.97])

    def _update_live_plot(self, mode: str) -> None:
        """Redraw axes content, then push the update to the correct display backend."""
        self._draw_metrics(mode)
        self._fig.tight_layout(pad=0.4, rect=[0.01, 0.01, 0.99, 0.97])

        if self.pygame_renderer is not None:
            # Renderer caches a converted surface; it stays valid until the next
            # notify_figure_updated call.
            self.pygame_renderer.notify_figure_updated(self._fig)
        else:
            self._headless_display()

    def _headless_display(self) -> None:
        import matplotlib.pyplot as plt

        if _is_notebook():
            from IPython.display import display, clear_output

            clear_output(wait=True)
            display(self._fig)
        else:
            # In a script plt.pause() flushes the event loop and keeps the
            # window responsive without blocking.
            self._fig.canvas.draw_idle()
            plt.pause(0.001)

    def _draw_metrics(self, mode: str) -> None:
        win = 50
        """Clear and redraw the axes from current model metrics."""
        assert self._axes is not None

        if mode == "train":
            ax1, ax2, ax3, ax4, ax5, ax6, ax7 = self._axes
            for ax in self._axes:
                ax.clear()

            m = self.model.get_train_metrics()
            r = np.asarray(m.get("total_reward", []), dtype=np.float32)
            s = np.asarray(m.get("success", []), dtype=np.float32)
            c = np.asarray(m.get("collision", []), dtype=np.float32)
            steps = np.asarray(m.get("steps", []), dtype=np.float32)
            kl = np.asarray(m.get("kl_div", []), dtype=np.float32)
            entropy = np.asarray(m.get("entropy", []), dtype=np.float32)
            sigma0 = np.asarray(m.get("sigma_joint_0", []), dtype=np.float32)
            sigma1 = np.asarray(m.get("sigma_joint_1", []), dtype=np.float32)
            sigma2 = np.asarray(m.get("sigma_joint_2", []), dtype=np.float32)

            # Row 0, Col 0: Total reward
            if r.size:
                ax1.plot(*_downsample(r), alpha=0.3, lw=0.5, color="#2196F3")
                ax1.plot(*_downsample(_running_mean(r, win)), lw=1.8, color="#1565C0")
            ax1.set_title(f"total reward (ma={win})", fontsize=14)
            ax1.set_xlabel("episode", fontsize=12)
            ax1.tick_params(labelsize=11)
            ax1.grid(True, alpha=0.3)

            # Row 0, Col 1: Steps/episode
            if steps.size:
                ax2.plot(*_downsample(steps), alpha=0.3, lw=0.5, color="#FF9800")
                ax2.plot(*_downsample(_running_mean(steps, win)), lw=1.8, color="#E65100")
            ax2.set_title(f"steps / episode (ma={win})", fontsize=14)
            ax2.set_xlabel("episode", fontsize=12)
            ax2.tick_params(labelsize=11)
            ax2.grid(True, alpha=0.3)

            # Row 1, Col 0: KL divergence
            if kl.size:
                valid_kl = kl[~np.isnan(kl)]
                if valid_kl.size > 0:
                    ax3.plot(*_downsample(valid_kl), alpha=0.3, lw=0.5, color="#9C27B0")
                    ax3.plot(*_downsample(_running_mean(valid_kl, win)), lw=1.8, color="#6A1B9A")
            ax3.set_title(f"KL divergence (ma={win})", fontsize=14)
            ax3.set_xlabel("update", fontsize=12)
            ax3.tick_params(labelsize=11)
            ax3.grid(True, alpha=0.3)

            # Row 1, Col 1: Sigma (all joints)
            if sigma0.size:
                valid_sigma0 = sigma0[~np.isnan(sigma0)]
                if valid_sigma0.size > 0:
                    ax4.plot(
                        *_downsample(valid_sigma0),
                        lw=1.2,
                        color="#FF5252",
                        label="joint_0",
                        alpha=0.8,
                    )
            if sigma1.size:
                valid_sigma1 = sigma1[~np.isnan(sigma1)]
                if valid_sigma1.size > 0:
                    ax4.plot(
                        *_downsample(valid_sigma1),
                        lw=1.2,
                        color="#2196F3",
                        label="joint_1",
                        alpha=0.8,
                    )
            if sigma2.size:
                valid_sigma2 = sigma2[~np.isnan(sigma2)]
                if valid_sigma2.size > 0:
                    ax4.plot(
                        *_downsample(valid_sigma2),
                        lw=1.2,
                        color="#4CAF50",
                        label="joint_2",
                        alpha=0.8,
                    )
            ax4.set_title("sigma (policy std)", fontsize=14)
            ax4.set_xlabel("update", fontsize=12)
            ax4.legend(fontsize=11, loc="best")
            ax4.tick_params(labelsize=11)
            ax4.grid(True, alpha=0.3)

            # Row 2, Col 0: Collision rate
            if c.size:
                ax5.plot(*_downsample(_windowed_rate(c, self._win)), lw=1.8, color="#F44336")
            ax5.set_ylim(-0.05, 1.05)
            ax5.set_title(f"collision rate (window={self._win})", fontsize=14)
            ax5.set_xlabel("episode", fontsize=12)
            ax5.tick_params(labelsize=11)
            ax5.grid(True, alpha=0.3)

            # Row 2, Col 1: Entropy
            if entropy.size:
                valid_entropy = entropy[~np.isnan(entropy)]
                if valid_entropy.size > 0:
                    ax6.plot(*_downsample(valid_entropy), alpha=0.3, lw=0.5, color="#607D8B")
                    ax6.plot(
                        *_downsample(_running_mean(valid_entropy, win)), lw=1.8, color="#455A64"
                    )
            ax6.set_title(f"entropy (ma={win})", fontsize=14)
            ax6.set_xlabel("update", fontsize=12)
            ax6.tick_params(labelsize=11)
            ax6.grid(True, alpha=0.3)

            # Row 3, full width: Success rate
            if s.size:
                ax7.plot(*_downsample(_windowed_rate(s, self._win)), lw=2.0, color="#4CAF50")
            ax7.set_ylim(-0.05, 1.05)
            ax7.set_title(f"success rate (window={self._win})", fontsize=15, fontweight="bold")
            ax7.set_xlabel("episode", fontsize=12)
            ax7.tick_params(labelsize=11)
            ax7.grid(True, alpha=0.3)

        else:  # test
            ax1, ax2, ax3 = self._axes
            ax1.clear()
            ax2.clear()
            ax3.clear()

            m = self.model.get_test_metrics()
            s = np.asarray(m.get("success", []), dtype=np.float32)
            coll = np.asarray(m.get("collision", []), dtype=np.float32)
            steps = np.asarray(m.get("steps", []), dtype=np.float32)

            if s.size:
                ax1.plot(np.cumsum(s) / np.arange(1, s.size + 1), lw=1.8, color="#4CAF50")
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_title("cumulative success rate", fontsize=14)
            ax1.set_xlabel("episode", fontsize=12)
            ax1.tick_params(labelsize=11)
            ax1.grid(True, alpha=0.3)

            if coll.size:
                ax2.plot(np.cumsum(coll) / np.arange(1, coll.size + 1), lw=1.8, color="#F44336")
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_title("cumulative collision rate", fontsize=14)
            ax2.set_xlabel("episode", fontsize=12)
            ax2.tick_params(labelsize=11)
            ax2.grid(True, alpha=0.3)

            if steps.size:
                ax3.plot(steps, color="#FF9800", lw=0.8)
            ax3.set_title("steps / episode", fontsize=14)
            ax3.set_xlabel("episode", fontsize=12)
            ax3.tick_params(labelsize=11)
            ax3.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_env(self, *, train: bool) -> Environment:
        if train:
            self.model.set_train_mode()
        else:
            self.model.set_eval_mode()
        return Environment(
            env_cfg=self.env_cfg,
            reward_cfg=self.reward_cfg,
            robot_cfg=self.robot_cfg,
            lidar_cfg=self.lidar_cfg,
            model=self.model,  # type: ignore[arg-type]
            obstacle_cfg=self.obstacle_cfg,
            seed=self.seed,
        )

    def _save_model(self, path: Optional[str] = None) -> None:
        p = path or self.gui_cfg.model_path
        self.model.save(p, include_optimizer=False, include_metrics=False)
        if self.progress_every > 0:
            print(f"[runner] model saved → {p}", flush=True)

    def _print_progress(self, mode: str, episode: int, n: int) -> None:
        if mode == "train":
            m = self.model.get_train_metrics()
            rewards = m.get("total_reward", [])
            sr = m.get("success", [])
            win = min(self._win, len(sr))
            recent_sr = float(np.mean(sr[-win:])) if win else 0.0
            last_r = rewards[-1] if rewards else float("nan")
            ent = m.get("entropy", [])
            last_ent = ent[-1] if ent else float("nan")
            print(
                f"[train] ep {episode:>5}/{n}  "
                f"reward={last_r:+.2f}  "
                f"sr({win})={recent_sr:.3f}",
                flush=True,
            )
        else:
            sr = self.model.get_test_metrics().get("success", [])
            csr = float(np.mean(sr)) if sr else 0.0
            print(f"[test]  ep {episode:>5}/{n}  cumulative_sr={csr:.3f}", flush=True)


# ---------------------------------------------------------------------------
# Private numeric helpers
# ---------------------------------------------------------------------------

_MAX_PLOT_POINTS = 500


def _downsample(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    if n <= _MAX_PLOT_POINTS:
        return np.arange(n), arr
    idx = np.linspace(0, n - 1, _MAX_PLOT_POINTS, dtype=int)
    return idx, arr[idx]


def _running_mean(arr: np.ndarray, win: int) -> np.ndarray:
    cs = np.cumsum(arr)
    out = np.empty_like(cs)
    out[:win] = cs[:win] / np.arange(1, min(win, arr.size) + 1, dtype=np.float32)
    if arr.size > win:
        out[win:] = (cs[win:] - cs[:-win]) / win
    return out


def _windowed_rate(s: np.ndarray, win: int) -> np.ndarray:
    cs = np.cumsum(s)
    out = np.empty_like(cs)
    out[:win] = cs[:win] / np.arange(1, min(win, s.size) + 1, dtype=np.float32)
    if s.size > win:
        out[win:] = (cs[win:] - cs[:-win]) / win
    return out
