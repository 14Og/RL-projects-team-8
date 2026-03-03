"""
Runner — unified training/evaluation loop.

Works in two modes:
  - Headless (``pygame_renderer=None``): live-updating matplotlib figure with no
    pygame dependency. Works from scripts (interactive window via plt.ion) and
    Jupyter notebooks (inline via IPython.display).
  - Pygame: passes the same matplotlib Figure to ``PygameRenderer`` which
    converts it to a surface and blits it into the pygame window. Zero
    duplicated loop logic between the two modes.

Typical notebook usage
----------------------
::

    from reinforce.runner import Runner, compute_obs_dim
    from reinforce.model_ppo import Model
    from reinforce.config import (
        RobotConfig, LidarConfig, EnvConfig, RewardConfig, ObstacleConfig, GUIConfig, ModelConfig,
    )

    robot_cfg  = RobotConfig()
    lidar_cfg  = LidarConfig()
    model_cfg  = ModelConfig()
    gui_cfg    = GUIConfig()
    obs_dim    = compute_obs_dim(robot_cfg, lidar_cfg)
    model      = Model(
        obs_dim=obs_dim,
        act_dim=len(robot_cfg.link_lengths),
        cfg=model_cfg,
        action_limit=robot_cfg.dtheta_max,
        train_episodes=gui_cfg.train_episodes,
    )
    runner = Runner(
        env_cfg=EnvConfig(), reward_cfg=RewardConfig(),
        robot_cfg=robot_cfg, lidar_cfg=lidar_cfg,
        obstacle_cfg=ObstacleConfig(), gui_cfg=gui_cfg, model=model,
    )
    runner.train(episodes=500)   # live plot updates inline every plot_update_every ep
    runner.test(episodes=100)

Typical pygame usage (called from main.py, not directly)
---------------------------------------------------------
::

    from reinforce.gui import PygameRenderer
    renderer = PygameRenderer(gui_cfg, env_cfg)
    runner   = Runner(..., pygame_renderer=renderer)
    runner.run()   # train → save → test, full interactive window
"""

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


# ---------------------------------------------------------------------------
# Notebook detection
# ---------------------------------------------------------------------------

def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Protocol — anything that looks like a Model
# ---------------------------------------------------------------------------

class _ModelLike(Protocol):
    def start_episode(self) -> None: ...
    def select_action(self, state: np.ndarray, *, train: bool) -> np.ndarray: ...
    def observe(self, reward: float) -> None: ...
    def finish_episode(self, *, success: bool, collision: bool = False, final_distance: Optional[float] = None) -> Dict: ...
    def record_test_episode(self, *, success: bool, final_distance: float, steps: int) -> None: ...
    def get_train_metrics(self) -> Dict[str, list]: ...
    def get_test_metrics(self) -> Dict[str, list]: ...
    def set_train_mode(self) -> None: ...
    def set_eval_mode(self) -> None: ...
    def save(self, path: str, **kwargs: Any) -> None: ...
    def load(self, path: str, **kwargs: Any) -> None: ...


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def compute_obs_dim(robot_cfg: RobotConfig, lidar_cfg: LidarConfig) -> int:
    """Compute flat observation dimension from config (no robot instantiation needed).

    Layout mirrors ``State.__array__``:
        [sin(θi), cos(θi)] × n_dof   →  2 * n_dof
        [ee_x, ee_y, dist_x, dist_y] →  4
        lidar rays                    →  n_lidars * num_rays
    """
    n_dof = len(robot_cfg.link_lengths)
    n_lidars = n_dof * (int(lidar_cfg.lidar_joints) + int(lidar_cfg.lidar_midlinks))
    return 2 * n_dof + 4 + n_lidars * lidar_cfg.num_rays


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

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
        model: _ModelLike,
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
        # Headless runs one step per outer iteration — no render overhead anyway.
        spf = self.gui_cfg.steps_per_frame if self.pygame_renderer is not None else 1

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

                    if (
                        self.pygame_renderer is not None
                        and self.gui_cfg.pause_on_done_frames > 0
                    ):
                        pause_frames_left = self.gui_cfg.pause_on_done_frames
                        break

                    if episode_count >= n_episodes:
                        break

            # ---- render frame (pygame only) ----
            if self.pygame_renderer is not None:
                self.pygame_renderer.render(
                    env.get_render_data(), mode, episode_count, n_episodes
                )

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

        n_subplots = 7 if mode == "train" else 3
        if self.pygame_renderer is not None:
            # Non-interactive Agg figure.  PygameRenderer will convert it to a
            # pygame surface on demand.
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            plot_w = self.gui_cfg.window_size[0] - self.gui_cfg.sim_width
            plot_h = self.gui_cfg.window_size[1]
            self._fig = Figure(figsize=(plot_w / 100.0, plot_h / 100.0), dpi=100)
            FigureCanvasAgg(self._fig)  # attaches Agg canvas
            if mode == "train":
                self._axes = (
                    self._fig.add_subplot(711),
                    self._fig.add_subplot(712),
                    self._fig.add_subplot(713),
                    self._fig.add_subplot(714),
                    self._fig.add_subplot(715),
                    self._fig.add_subplot(716),
                    self._fig.add_subplot(717),
                )
            else:
                self._axes = (
                    self._fig.add_subplot(311),
                    self._fig.add_subplot(312),
                    self._fig.add_subplot(313),
                )
            # Produce an initial (empty) surface so the window isn't blank.
            self.pygame_renderer.notify_figure_updated(self._fig)
        else:
            # Interactive figure: shows a live window in scripts, or renders
            # inline in Jupyter notebooks.
            if not _is_notebook():
                plt.ion()
            self._fig, _axes = plt.subplots(n_subplots, 1, figsize=(10, 7 + 2 * (n_subplots - 3)))
            if n_subplots == 1:
                self._axes = (_axes,)  # type: ignore[assignment]
            else:
                self._axes = tuple(_axes)  # type: ignore[assignment]
            if not _is_notebook():
                plt.show(block=False)

        self._fig.suptitle(f"{mode} metrics", fontsize=12)
        self._fig.tight_layout(pad=1.5)

    def _update_live_plot(self, mode: str) -> None:
        """Redraw axes content, then push the update to the correct display backend."""
        self._draw_metrics(mode)
        self._fig.tight_layout(pad=1.5)

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
        """Clear and redraw the axes from current model metrics."""
        assert self._axes is not None
        
        if mode == "train":
            ax1, ax2, ax3, ax4, ax5, ax6, ax7 = self._axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax5.clear()
            ax6.clear()
            ax7.clear()
            
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

            if r.size:
                ax1.plot(*_downsample(r), alpha=0.3, lw=0.5, color="#2196F3")
                ax1.plot(*_downsample(_running_mean(r, 10)), lw=1.8, color="#1565C0")
            ax1.set_title("total reward (ma=10)")
            ax1.grid(True, alpha=0.3)

            if s.size:
                ax2.plot(*_downsample(_windowed_rate(s, self._win)), lw=1.8, color="#4CAF50")
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_title(f"success rate (window={self._win})")
            ax2.grid(True, alpha=0.3)

            if steps.size:
                ax3.plot(*_downsample(steps), alpha=0.3, lw=0.5, color="#FF9800")
                ax3.plot(*_downsample(_running_mean(steps, 10)), lw=1.8, color="#E65100")
            ax3.set_title("steps / episode (ma=10)")
            ax3.grid(True, alpha=0.3)
            
            # KL divergence
            if kl.size:
                valid_kl = kl[~np.isnan(kl)]
                if valid_kl.size > 0:
                    ax4.plot(*_downsample(valid_kl), alpha=0.3, lw=0.5, color="#9C27B0")
                    ax4.plot(*_downsample(_running_mean(valid_kl, 10)), lw=1.8, color="#6A1B9A")
            ax4.set_title("KL divergence (ma=10)")
            ax4.grid(True, alpha=0.3)
            
            # Sigma values for all joints on one plot
            if sigma0.size:
                valid_sigma0 = sigma0[~np.isnan(sigma0)]
                if valid_sigma0.size > 0:
                    ax5.plot(*_downsample(valid_sigma0), lw=1.2, color="#FF5252", label="joint_0", alpha=0.8)
            if sigma1.size:
                valid_sigma1 = sigma1[~np.isnan(sigma1)]
                if valid_sigma1.size > 0:
                    ax5.plot(*_downsample(valid_sigma1), lw=1.2, color="#2196F3", label="joint_1", alpha=0.8)
            if sigma2.size:
                valid_sigma2 = sigma2[~np.isnan(sigma2)]
                if valid_sigma2.size > 0:
                    ax5.plot(*_downsample(valid_sigma2), lw=1.2, color="#4CAF50", label="joint_2", alpha=0.8)
            ax5.set_title("sigma (policy std)")
            ax5.legend(fontsize=8, loc="best")
            ax5.grid(True, alpha=0.3)
            
            # Collision rate
            if c.size:
                ax6.plot(*_downsample(_windowed_rate(c, self._win)), lw=1.8, color="#F44336")
            ax6.set_ylim(-0.05, 1.05)
            ax6.set_title(f"collision rate (window={self._win})")
            ax6.grid(True, alpha=0.3)
            
            # Entropy (policy exploration)
            if entropy.size:
                ax7.plot(*_downsample(entropy), alpha=0.3, lw=0.5, color="#607D8B")
                ax7.plot(*_downsample(_running_mean(entropy, 10)), lw=1.8, color="#455A64")
            ax7.set_title("entropy (ma=10)")
            ax7.grid(True, alpha=0.3)

        else:  # test
            ax1, ax2, ax3 = self._axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            m = self.model.get_test_metrics()
            s = np.asarray(m.get("success", []), dtype=np.float32)
            dist = np.asarray(m.get("final_distance", []), dtype=np.float32)
            steps = np.asarray(m.get("steps", []), dtype=np.float32)

            if s.size:
                ax1.plot(np.cumsum(s) / np.arange(1, s.size + 1), lw=1.8, color="#4CAF50")
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_title("cumulative success rate")
            ax1.grid(True, alpha=0.3)

            if dist.size:
                ax2.plot(dist, color="#F44336", lw=0.8)
                ax2.axhline(
                    float(dist.mean()), color="#B71C1C", ls="--", lw=1.2,
                    label=f"mean={dist.mean():.2f}",
                )
                ax2.legend(fontsize=7)
            ax2.set_title("final distance")
            ax2.grid(True, alpha=0.3)

            if steps.size:
                ax3.plot(steps, color="#FF9800", lw=0.8)
            ax3.set_title("steps / episode")
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
                f"sr({win})={recent_sr:.3f}  "
                f"entropy={last_ent:.3f}",
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
