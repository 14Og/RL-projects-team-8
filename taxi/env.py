from contextlib import closing
from io import StringIO
from os import path
import numpy as np
import gymnasium as gym
from gymnasium import spaces, utils
from .config import EnvConfig, Action

MAP = [
    "+-------------+",
    "|R: | : : | :G|",
    "| : | : : | : |",
    "| : | : : | : |",
    "| : : :F| : : |",
    "| | : : | : : |",
    "| | : : | : : |",
    "|Y| : : : : :B|",
    "+-------------+",
]
GRID_W, GRID_H = 750 * 2, 450 * 2
INFO_H = 80
WINDOW_SIZE = (GRID_W, GRID_H + INFO_H)
LOC_NAMES = ["R", "G", "Y", "B"]


class TaxiEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, cfg: EnvConfig = None, render_mode: str = None):
        self.cfg = cfg or EnvConfig()
        self.desc = np.asarray(MAP, dtype="c")
        self.locs = [(0, 0), (0, 6), (6, 0), (6, 6)]
        self.locs_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (255, 165, 0),
            (0, 0, 255),
        ]
        self.gas_station = self.cfg.gas_station
        fuel_states = (self.cfg.max_fuel + 1) if self.cfg.use_fuel else 1
        num_states = self.cfg.rows * self.cfg.cols * 5 * 4 * fuel_states
        self.observation_space = spaces.Discrete(num_states)
        self.action_space = spaces.Discrete(len(Action) if self.cfg.use_fuel else len(Action) - 1)
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = (GRID_W / self.desc.shape[1], GRID_H / self.desc.shape[0])
        self._sprites: dict = {}
        self.taxi_pos = [0, 0]
        self.pass_idx = 0
        self.dest_idx = 1
        self.fuel = self.cfg.max_fuel
        self._count_steps = 0
        self.lastaction = None

    def encode(self, row, col, pass_idx, dest_idx, fuel=0):
        s = row
        s = s * self.cfg.cols + col
        s = s * 5 + pass_idx
        s = s * 4 + dest_idx
        if self.cfg.use_fuel:
            s = s * (self.cfg.max_fuel + 1) + fuel
        return s

    def decode(self, s):
        if self.cfg.use_fuel:
            fuel = s % (self.cfg.max_fuel + 1)
            s //= self.cfg.max_fuel + 1
        else:
            fuel = 0
        dest_idx = s % 4
        s //= 4
        pass_idx = s % 5
        s //= 5
        col = s % self.cfg.cols
        row = s // self.cfg.cols
        return row, col, pass_idx, dest_idx, fuel

    # ------------------------------------------------------------------
    # Core gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._count_steps = 0
        self.lastaction = None

        self.taxi_pos = [
            self.np_random.integers(0, self.cfg.rows),
            self.np_random.integers(0, self.cfg.cols),
        ]
        self.pass_idx = int(self.np_random.integers(0, 4))
        self.dest_idx = int(self.np_random.integers(0, 4))
        while self.dest_idx == self.pass_idx:
            self.dest_idx = int(self.np_random.integers(0, 4))

        self.fuel = self.cfg.max_fuel

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action):
        row, col = self.taxi_pos
        reward = -1
        terminated = False
        truncated = False

        # Convert integer action to Enum if needed
        if isinstance(action, int):
            action = Action(action)

        # Fuel consumed on every action except refuel
        if self.cfg.use_fuel and action != Action.REFUEL:
            self.fuel -= 1


        if action in (Action.SOUTH, Action.NORTH, Action.EAST, Action.WEST):
            new_row, new_col = row, col
            blocked = False
            if action == Action.SOUTH:
                new_row = min(row + 1, self.cfg.rows - 1)
                if new_row == row:
                    blocked = True
            elif action == Action.NORTH:
                new_row = max(row - 1, 0)
                if new_row == row:
                    blocked = True
            elif action == Action.EAST:
                if self.desc[1 + row, 2 * col + 2] != b":":
                    blocked = True
                else:
                    new_col = col + 1
            elif action == Action.WEST:
                if self.desc[1 + row, 2 * col] != b":":
                    blocked = True
                else:
                    new_col = col - 1
            if blocked:
                reward = -3
            else:
                self.taxi_pos = [new_row, new_col]
        elif action == Action.PICKUP:
            if self.pass_idx < 4 and self.taxi_pos == list(self.locs[self.pass_idx]):
                self.pass_idx = 4
                reward = 15
            else:
                reward = -10
        elif action == Action.DROPOFF:
            if self.pass_idx == 4 and self.taxi_pos == list(self.locs[self.dest_idx]):
                self.pass_idx = self.dest_idx
                reward = 60
                terminated = True
            else:
                reward = -10
        elif action == Action.REFUEL and self.cfg.use_fuel:
            if tuple(self.taxi_pos) == self.gas_station:
                self.fuel = self.cfg.max_fuel
                reward = -1
            else:
                reward = -10

        # Out-of-fuel death
        if self.cfg.use_fuel and self.fuel < 0:
            reward = -50
            terminated = True
            self.fuel = 0

        # Step limit
        self._count_steps += 1
        if not terminated and self._count_steps >= self.cfg.max_steps:
            reward = -50
            truncated = True

        self.lastaction = action
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return self.encode(
            self.taxi_pos[0],
            self.taxi_pos[1],
            self.pass_idx,
            self.dest_idx,
            self.fuel if self.cfg.use_fuel else 0,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        return self._render_gui(self.render_mode)

    def _render_text(self):
        desc = self.desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in desc]
        row, col = self.taxi_pos

        def ul(x):
            return "_" if x == " " else x

        if self.pass_idx < 4:
            out[1 + row][2 * col + 1] = utils.colorize(
                out[1 + row][2 * col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[self.pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], "blue", bold=True)
        else:
            out[1 + row][2 * col + 1] = utils.colorize(
                ul(out[1 + row][2 * col + 1]), "green", highlight=True
            )

        di, dj = self.locs[self.dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")

        # Gas station marker
        gi, gj = self.gas_station
        out[1 + gi][2 * gj + 1] = utils.colorize(out[1 + gi][2 * gj + 1], "cyan", bold=True)

        outfile = StringIO()
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        outfile.write(f"  Fuel: {self.fuel}/{self.cfg.max_fuel}")
        if self.lastaction is not None:
            try:
                action_name = Action(self.lastaction).name.capitalize()
            except Exception:
                action_name = str(self.lastaction)
            outfile.write(f"  ({action_name})\n")
        else:
            outfile.write("\n")
        with closing(outfile):
            return outfile.getvalue()

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            ) from e

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Taxi 7x7")
            if mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            else:
                self.window = pygame.Surface(WINDOW_SIZE)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._load_sprites(pygame)

        desc = self.desc
        cell_w, cell_h = self.cell_size

        # --- Background and wall tiles ---
        for y in range(desc.shape[0]):
            for x in range(desc.shape[1]):
                cell = (x * cell_w, y * cell_h)
                self.window.blit(self._sprites["bg"], cell)
                ch = desc[y][x]
                if ch == b"|":
                    if y == 0 or desc[y - 1][x] != b"|":
                        self.window.blit(self._sprites["vert"][0], cell)
                    elif y == desc.shape[0] - 1 or desc[y + 1][x] != b"|":
                        self.window.blit(self._sprites["vert"][2], cell)
                    else:
                        self.window.blit(self._sprites["vert"][1], cell)
                elif ch == b"-":
                    if x == 0 or desc[y][x - 1] != b"-":
                        self.window.blit(self._sprites["horiz"][0], cell)
                    elif x == desc.shape[1] - 1 or desc[y][x + 1] != b"-":
                        self.window.blit(self._sprites["horiz"][2], cell)
                    else:
                        self.window.blit(self._sprites["horiz"][1], cell)

        # --- Location color overlays ---
        for loc, color in zip(self.locs, self.locs_colors):
            surf = pygame.Surface(self.cell_size)
            surf.set_alpha(128)
            surf.fill(color)
            px, py = self._surf_loc(loc)
            self.window.blit(surf, (px, py + 10))

        # --- Gas station overlay ---
        gi, gj = self.gas_station
        gas_surf = pygame.Surface(self.cell_size)
        gas_surf.set_alpha(180)
        gas_surf.fill((80, 180, 80))
        px, py = self._surf_loc((gi, gj))
        self.window.blit(gas_surf, (px, py))
        font = pygame.font.SysFont("Arial", int(cell_h * 0.35), bold=True)
        label = font.render("GAS", True, (255, 255, 255))
        lw, lh = label.get_size()
        self.window.blit(label, (px + (cell_w - lw) // 2, py + (cell_h - lh) // 2))

        # --- Passenger sprite ---
        taxi_r, taxi_c = self.taxi_pos
        if self.pass_idx < 4:
            self.window.blit(
                self._sprites["passenger"],
                self._surf_loc(self.locs[self.pass_idx]),
            )

        # --- Destination hotel ---
        dest_loc = self._surf_loc(self.locs[self.dest_idx])
        taxi_loc = self._surf_loc((taxi_r, taxi_c))
        hotel = self._sprites["hotel"]
        cab = self._sprites["cabs"][self.lastaction if self.lastaction in (0, 1, 2, 3) else 0]

        if dest_loc[1] <= taxi_loc[1]:
            self.window.blit(hotel, (dest_loc[0], dest_loc[1] - cell_h // 2))
            self.window.blit(cab, taxi_loc)
        else:
            self.window.blit(cab, taxi_loc)
            self.window.blit(hotel, (dest_loc[0], dest_loc[1] - cell_h // 2))

        # --- Info panel (fuel bar + target) ---
        panel_y = GRID_H
        self.window.fill((30, 30, 30), (0, panel_y, GRID_W, INFO_H))

        info_font = pygame.font.SysFont("Arial", 20)
        # Fuel text
        fuel_label = info_font.render(
            f"Fuel: {self.fuel}/{self.cfg.max_fuel}", True, (220, 220, 220)
        )
        self.window.blit(fuel_label, (10, panel_y + 10))
        # Fuel bar
        bar_x, bar_y, bar_w, bar_h = 10, panel_y + 40, 220, 18
        pygame.draw.rect(self.window, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))
        ratio = self.fuel / self.cfg.max_fuel
        bar_color = (50, 200, 50) if ratio > 0.3 else (200, 50, 50)
        pygame.draw.rect(self.window, bar_color, (bar_x, bar_y, int(bar_w * ratio), bar_h))

        # Target
        target_label = info_font.render(
            f"Deliver to: {LOC_NAMES[self.dest_idx]}", True, (220, 220, 220)
        )
        self.window.blit(target_label, (260, panel_y + 25))

        # Passenger status
        pass_text = "In taxi" if self.pass_idx == 4 else f"At {LOC_NAMES[self.pass_idx]}"
        pass_label = info_font.render(f"Passenger: {pass_text}", True, (180, 180, 180))
        self.window.blit(pass_label, (480, panel_y + 25))

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def _surf_loc(self, map_loc):
        """Grid (row, col) → pixel top-left of that cell."""
        cw, ch = self.cell_size
        return (map_loc[1] * 2 + 1) * cw, (map_loc[0] + 1) * ch

    def _load_sprites(self, pygame):
        if self._sprites:
            return
        img_dir = path.join(path.dirname(__file__), "img")
        cs = self.cell_size

        def load(name):
            return pygame.transform.scale(pygame.image.load(path.join(img_dir, name)), cs)

        self._sprites = {
            "bg": load("taxi_background.png"),
            "passenger": load("passenger.png"),
            "hotel": load("hotel.png"),
            "cabs": [
                load("cab_rear.png"),  # 0 south → rear view
                load("cab_front.png"),  # 1 north → front view
                load("cab_right.png"),  # 2 east
                load("cab_left.png"),  # 3 west
            ],
            "horiz": [
                load("gridworld_median_left.png"),
                load("gridworld_median_horiz.png"),
                load("gridworld_median_right.png"),
            ],
            "vert": [
                load("gridworld_median_top.png"),
                load("gridworld_median_vert.png"),
                load("gridworld_median_bottom.png"),
            ],
        }
        self._sprites["hotel"].set_alpha(170)

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
