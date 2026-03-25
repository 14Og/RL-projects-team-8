from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .config import EnvConfig, GUIConfig


class PygameRenderer:
    """Pygame window + drawing, decoupled from the training loop.

    Parameters
    ----------
    gui_cfg :
        Window dimensions, FPS caps, pause-on-done duration, etc.
    env_cfg :
        Used for ``target_thresh`` (circle radius drawn around the target).
    """

    def __init__(self, gui_cfg: GUIConfig, env_cfg: EnvConfig) -> None:
        pygame.init()
        self.cfg = gui_cfg
        self.env_cfg = env_cfg
        self.screen = pygame.display.set_mode(self.cfg.window_size)
        pygame.display.set_caption("RL Robot")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)

        self._plot_surface: Optional[pygame.Surface] = None

    def handle_events(self, paused: bool) -> Tuple[bool, bool]:
        """Drain the event queue.  Returns ``(quit_requested, paused)``."""
        quit_req = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_req = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_req = True
                elif event.key == pygame.K_SPACE:
                    paused = not paused
        return quit_req, paused

    def notify_figure_updated(self, fig: Any) -> None:
        """Convert a freshly-drawn matplotlib Figure to a cached pygame surface.

        Runner calls this every ``plot_update_every`` episodes.  The conversion
        is the expensive step (Agg rasterisation); render() just blits the cache.
        """
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[..., :3]
        surf = pygame.surfarray.make_surface(np.transpose(np.ascontiguousarray(rgb), (1, 0, 2)))
        plot_w = self.cfg.window_size[0] - self.cfg.sim_width
        plot_h = self.cfg.window_size[1]
        self._plot_surface = pygame.transform.smoothscale(surf, (plot_w, plot_h))

    def render(
        self,
        render_data: Dict[str, Any],
        mode: str,
        episode: int,
        n_episodes: int,
    ) -> None:
        """Draw one display frame: sim panel + plot panel + HUD text."""
        sim_w = self.cfg.sim_width
        w, h = self.cfg.window_size

        self.screen.fill((245, 245, 245))

        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, 0, sim_w, h))
        self._draw_scene(render_data)
        pygame.draw.line(self.screen, (200, 200, 200), (sim_w, 0), (sim_w, h), 2)

        if self._plot_surface is not None:
            self.screen.blit(self._plot_surface, (sim_w, 0))

        dist = render_data.get("distance", 0.0)
        step = render_data.get("step", 0)
        hud = f"mode={mode}  ep={episode}/{n_episodes}  " f"step={step}  dist={dist:.1f}"
        self.screen.blit(self.font.render(hud, True, (20, 20, 20)), (10, 10))

        lidar_data = render_data.get("lidar", [])
        y_offset = 35
        for i, lidar_info in enumerate(lidar_data):
            pos = lidar_info.get("position", np.zeros(2))
            readings = lidar_info.get("readings", np.array([]))

            if len(readings) > 0:
                min_reading = float(np.min(readings))
                max_reading = float(np.max(readings))
                avg_reading = float(np.mean(readings))
                lidar_text = (
                    f"Lidar {i}: pos=({pos[0]:.1f},{pos[1]:.1f})  "
                    f"min={min_reading:.3f} max={max_reading:.3f} avg={avg_reading:.3f}"
                )
            else:
                lidar_text = f"Lidar {i}: pos=({pos[0]:.1f},{pos[1]:.1f})  (no readings)"

            self.screen.blit(self.font.render(lidar_text, True, (60, 60, 60)), (10, y_offset))
            y_offset += 25

        pygame.display.flip()

        if mode == "test":
            self.clock.tick(10)

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close("all")
        pygame.quit()

    def _draw_scene(self, render: Dict[str, Any]) -> None:
        joints: np.ndarray = render["joints"]  # (n+1, 2)
        target: np.ndarray = render["target"]  # (2,)
        obstacles: List[Dict] = render.get("obstacles", [])
        H = self.cfg.window_size[1]

        def fy(y: float) -> int:
            return H - int(y)

        # Obstacles + danger zones
        danger_zone_px = int(render.get("danger_zone_px", 0))
        for obs in obstacles:
            cx = int(obs["center"][0])
            cy = fy(obs["center"][1])
            r = int(obs["radius"])

            # Danger zone: semi-transparent orange ring
            if danger_zone_px > 0:
                dz_r = r + danger_zone_px
                dz_surf = pygame.Surface((dz_r * 2, dz_r * 2), pygame.SRCALPHA)
                pygame.draw.circle(dz_surf, (255, 160, 0, 50), (dz_r, dz_r), dz_r)
                pygame.draw.circle(dz_surf, (255, 160, 0, 120), (dz_r, dz_r), dz_r, 2)
                self.screen.blit(dz_surf, (cx - dz_r, cy - dz_r))

            pygame.draw.circle(self.screen, (180, 100, 100), (cx, cy), r)
            pygame.draw.circle(self.screen, (140, 55, 55), (cx, cy), r, 2)

        pts = [(int(joints[i, 0]), fy(joints[i, 1])) for i in range(joints.shape[0])]
        pygame.draw.lines(self.screen, (50, 50, 50), False, pts, 6)

        for p in pts:
            pygame.draw.circle(self.screen, (80, 80, 80), p, 8)
        tx, ty = int(target[0]), fy(target[1])
        pygame.draw.circle(self.screen, (220, 50, 50), (tx, ty), 6)
        pygame.draw.circle(
            self.screen,
            (220, 50, 50),
            (tx, ty),
            max(1, int(self.env_cfg.target_thresh)),
            1,
        )


if __name__ == "__main__":
    raise RuntimeError("Import PygameRenderer from reinforce.gui.")
