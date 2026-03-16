from __future__ import annotations
import pygame
from .config import GUIConfig
from .env import TaxiEnv
from .agent import QAgent


class PygameRenderer:
    def __init__(self, gui_cfg: GUIConfig):
        self.cfg = gui_cfg

    def run(self, env: TaxiEnv, agent: QAgent):
        env.render_mode = "human"
        state, _ = env.reset()
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if not running:
                break
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print(f"Episode done | reward={reward}")
                pygame.time.delay(1000)
                state, _ = env.reset()
            clock.tick(self.cfg.fps)
        env.close()
