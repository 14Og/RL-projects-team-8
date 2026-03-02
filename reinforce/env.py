from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from .config import EnvConfig, LidarConfig, RewardConfig, RobotConfig
from .obstacle import Obstacle
from .robot import Robot
from .model import Model
from .state import State
from .obstacle import ObstacleManager, ObstacleConfig, Obstacle, random_obstacle_config


class Environment:
    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        robot_cfg: RobotConfig,
        lidar_cfg: LidarConfig,
        model: Model,
        obstacles_config: Optional[List[ObstacleConfig]] = None,
        obstacles_gen_params: Optional[dict] = None,
        seed: int = 42,
        obstacles: Optional[List[Obstacle]] = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.rew_cfg = reward_cfg
        self.target = np.asarray(env_cfg.target_xy, dtype=np.float32)

        self._obstacles: List[Obstacle] = obstacles if obstacles is not None else []
        self.robot = Robot(robot_cfg, lidar_cfg, self._obstacles, seed=seed)
        self.model = model
        self._rng = np.random.default_rng(seed)

        base = np.asarray(robot_cfg.base_xy, dtype=np.float32)
        links = np.asarray(robot_cfg.link_lengths, dtype=float)
        self._base = base
        self._reach_max = float(np.sum(links))
        self._reach_min = float(max(0.0, links[0] - np.sum(links[1:])))

        self._collision_threshold: float = 0.05

        self._train_mode: bool = True
        self._needs_reset: bool = True

        self.steps: int = 0
        self.done: bool = False
        self.success: bool = False
        self.reason: str = "not_started"
        self._prev_dist: float = float("nan")

        self._last_state: Optional[np.ndarray] = None
        self._prev_action = None
        self._curr_action = None

    def check_collision(self, joints: np.ndarray) -> bool:
        """Check if any robot link segment collides with any obstacle."""
        for obs in self._obstacles:
            r = obs.radius + self._collision_threshold
            for i in range(len(joints) - 1):
                p1, p2 = joints[i], joints[i + 1]
                vec = p2 - p1
                t = np.clip(np.dot(obs.center - p1, vec) / (np.dot(vec, vec) + 1e-6), 0, 1)
                if np.linalg.norm(obs.center - (p1 + t * vec)) <= r:
                    return True
        return False


        # Инициализация препятствий
        self.obstacle_manager = None

        # 1) Явно переданные конфиги
        if obstacles_config is not None:
            self.obstacle_manager = ObstacleManager(obstacles_config)

        # 2) Параметры для генерации (если переданы)
        elif obstacles_gen_params is not None:
            self.obstacle_manager = ObstacleManager([])
            self.add_random_obstacles(**obstacles_gen_params)

        # 3) Генерация из EnvConfig (если включена)
        elif env_cfg.obstacles_enabled:
            gen_params = {
                "num": env_cfg.obstacles_num,
                "area": env_cfg.obstacles_area,
                "radius_range": (env_cfg.obstacles_radius_min, env_cfg.obstacles_radius_max),
                "check_robot": env_cfg.obstacles_check_robot,
                "check_overlap": env_cfg.obstacles_check_overlap,
            }
            self.obstacle_manager = ObstacleManager([])
            self.add_random_obstacles(**gen_params)

        # Если ни одно условие не сработало – manager остаётся None

    def reset_episode(self, *, train: bool = True, randomize_theta: bool = True) -> np.ndarray:
        self._train_mode = bool(train)

        self.robot.reset(randomize=randomize_theta)
        ee_start = self.robot.end_effector_xy()

        # Генерация цели с учётом текущего положения ee и препятствий
        if self.env_cfg.randomize_target:
            self.target = self._sample_valid_target(ee_start)

        self.robot.set_target(self.target)

        self.steps = 0
        self.done = False
        self.success = False
        self.reason = "running"

        if self._train_mode:
            self.model.start_episode()

        ee = self.robot.end_effector_xy()
        self._prev_dist = float(np.linalg.norm(ee - self.target))
        
        self._prev_action = None
        self._curr_action = None

        self._needs_reset = False

        st = self._get_state()
        self._last_state = st
        return np.asarray(st, dtype=np.float32)

    def step(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._needs_reset:
            obs0 = self.reset_episode(train=self._train_mode, randomize_theta=True)
            return obs0, 0.0, False, {"reset": True}

        if self.done:
            self._needs_reset = True
            st = self._get_state()
            return np.asarray(st, dtype=np.float32), 0.0, True, {"needs_reset": True}

        st = self._get_state()
        self._last_state = st

        if self._train_mode:
            u = self.model.select_action(st, train=True)
        else:
            u = self.model.select_action(st, train=False)
        _, self._curr_action = self.robot.step(u)
        self.steps += 1

        reward, done, info = self._compute_reward_and_done()

        if self._train_mode:
            self.model.observe(reward)

        self.done = bool(done)

        if self.done:
            final_dist = float(info.get("final_distance", float("nan")))
            self.success = bool(info.get("success", False))
            self.reason = str(info.get("reason", "done"))

            if self._train_mode:
                self.model.finish_episode(success=self.success, final_distance=final_dist)
            else:
                self.model.record_test_episode(
                    success=self.success, final_distance=final_dist, steps=int(self.steps)
                )

            self._needs_reset = True

        st_next = self._get_state()
        self._last_state = st_next
        self._prev_action = self._curr_action
        return np.asarray(st_next, dtype=np.float32), float(reward), bool(self.done), info

    def get_render_data(self) -> Dict[str, Any]:
        joints = self.robot.joints_xy().astype(np.float32)
        ee = joints[-1]
        dist = float(np.linalg.norm(ee - self.target))

        mgr = self.robot.lidar_manager
        lidar_data = [
            {
                "position": lidar.position.copy(),
                "readings": lidar.scan(mgr._obstacles),
                "ray_dirs": lidar.ray_dirs,
                "ray_maxlen": lidar.ray_maxlen,
            }
            for lidar in mgr.lidars
        ]

        obs_arr = (
            np.array([[o.center[0], o.center[1], o.radius] for o in self._obstacles], dtype=np.float32)
            if self._obstacles
            else np.empty((0, 3), dtype=np.float32)
        )

        data = {
            "joints": joints,
            "end_effector": ee,
            "target": self.target.copy(),
            "distance": dist,
            "theta": self.robot.theta,
            "obstacles": obs_arr,
            "lidar": lidar_data,
            "step": int(self.steps),
            "done": bool(self.done),
            "success": bool(self.success),
            "reason": self.reason,
            "train_mode": bool(self._train_mode),
        }

        if self.obstacle_manager:
            data["obstacles"] = self.obstacle_manager.get_render_data()

        return data

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "train": self.model.get_train_metrics(),
            "test": self.model.get_test_metrics(),
        }

    def _sample_reachable_target(self) -> np.ndarray:
        """(Legacy) старый метод без проверок, оставлен для обратной совместимости"""
        angle = self._rng.uniform(0, 2 * np.pi)
        r_sq = self._rng.uniform(self._reach_min ** 2, self._reach_max ** 2)
        r = np.sqrt(r_sq)
        return self._base + np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)

    def _is_target_valid(self, target: np.ndarray, ee_start: np.ndarray) -> bool:
        """Проверяет, удовлетворяет ли цель всем ограничениям."""
        # 1. Кинематическая достижимость (расстояние от базы)
        dist_from_base = np.linalg.norm(target - self._base)
        if dist_from_base < self._reach_min - 1e-6 or dist_from_base > self._reach_max + 1e-6:
            return False

        # 2. Не внутри препятствий
        if self.obstacle_manager is not None:
            for obs in self.obstacle_manager.obstacles:
                dist_to_obs = np.linalg.norm(target - obs.position)
                if dist_to_obs < obs.radius - 1e-6:
                    return False

        # 3. Минимальное расстояние от начального положения схвата
        if self.env_cfg.min_target_distance_from_ee > 0:
            dist_to_ee = np.linalg.norm(target - ee_start)
            if dist_to_ee < self.env_cfg.min_target_distance_from_ee:
                return False

        # 4. Прямая видимость от базы до цели (опционально)
        if self.env_cfg.target_line_of_sight and self.obstacle_manager is not None:
            for obs in self.obstacle_manager.obstacles:
                d = _point_to_segment_distance(obs.position, self._base, target)
                if d < obs.radius - 1e-6:
                    return False

        return True

    def _sample_valid_target(self, ee_start: np.ndarray, max_attempts: int = 1000) -> np.ndarray:
        """Генерирует случайную цель, удовлетворяющую всем условиям."""
        for _ in range(max_attempts):
            angle = self._rng.uniform(0, 2 * np.pi)
            r_sq = self._rng.uniform(self._reach_min ** 2, self._reach_max ** 2)
            r = np.sqrt(r_sq)
            candidate = self._base + np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
            if self._is_target_valid(candidate, ee_start):
                return candidate

        # Если не удалось, возвращаем запасной вариант (ближайшую к базе точку в направлении 0°)
        print("Warning: could not generate valid target after many attempts, using fallback")
        fallback = self._base + np.array([self._reach_min * 0.5, 0.0], dtype=np.float32)
        return fallback

    def _get_state(self) -> State:
        st = self.robot.obs()
        st.ee_x = (st.ee_x - self._base[0]) / self._reach_max
        st.ee_y = (st.ee_y - self._base[1]) / self._reach_max

        dx = float(self.target[0] - self.robot.end_effector_xy()[0])
        dy = float(self.target[1] - self.robot.end_effector_xy()[1])

        if self.env_cfg.use_abs_dist:
            dx, dy = abs(dx), abs(dy)

        if self.env_cfg.normalize_dist:
            s = float(self.env_cfg.dist_scale)
            dx, dy = dx / s, dy / s

        st.dist_x = dx
        st.dist_y = dy 
        return st

    def _compute_reward_and_done(self) -> Tuple[float, bool, Dict[str, Any]]:
        joints = self.robot.joints_xy()
        ee = joints[-1]
        dist = float(np.linalg.norm(ee - self.target))

        progress = float(self._prev_dist - dist)
        reward = float(self.rew_cfg.progress_scale) * progress - float(self.rew_cfg.step_penalty)
        
        a_t = self._curr_action
        if a_t is not None and self.rew_cfg.action_l2_scale != 0.0:
            reward -= self.rew_cfg.action_l2_scale * np.linalg.norm(a_t, a_t)

        if a_t is not None and self._prev_action is not None and self.rew_cfg.action_delta_scale != 0.0:
            da = a_t - self._prev_action
            reward -= self.rew_cfg.action_delta_scale * np.linalg.norm(da, da)
        
        # Check collision
        collision = self.check_collision(joints)
        fail_reason = ""
        if collision:
            fail = True
            fail_reason = "collision"
        else:
            fail = False
            
        goal_reached = dist < float(self.env_cfg.target_thresh)

        if self.env_cfg.forbid_link_target_intersection and (not goal_reached):
            r = float(self.env_cfg.target_point_radius)
            if any(
                _point_to_segment_distance(self.target, joints[i], joints[i + 1]) < r
                for i in range(len(joints) - 1)
            ):
                fail = True
                fail_reason = "link_target_intersection"

        timeout = self.steps >= int(self.env_cfg.max_steps)
        done = goal_reached or fail or timeout

        info: Dict[str, Any] = {
            "success": bool(goal_reached),
            "final_distance": dist,
            "progress": progress,
            "timeout": bool(timeout),
            "fail": bool(fail),
            "reason": (
                "goal" if goal_reached else ("timeout" if timeout else (fail_reason or "fail"))
            ),
        }

        if goal_reached:
            reward += float(self.rew_cfg.goal_reward)
        elif fail:
            reward -= float(self.rew_cfg.fail_penalty)
        elif timeout:
            reward -= float(self.rew_cfg.fail_penalty)

        self._prev_dist = dist
        return float(reward), bool(done), info

    def _obstacle_collides_with_robot(self, obs_pos: np.ndarray, obs_radius: float) -> bool:
        """Проверяет, пересекается ли круг препятствия с любым звеном робота в текущей конфигурации."""
        joints = self.robot.joints_xy()  # (3, 2) – точки сочленений
        for i in range(len(joints) - 1):
            dist = _point_to_segment_distance(obs_pos, joints[i], joints[i+1])
            if dist < obs_radius:
                return True
        return False

    def _obstacle_collides_with_others(self, obs_pos: np.ndarray, obs_radius: float, existing: List[Obstacle]) -> bool:
        """Проверяет пересечение с уже существующими препятствиями."""
        for other in existing:
            dist = np.linalg.norm(obs_pos - other.position)
            if dist < (obs_radius + other.radius):
                return True
        return False

    def add_random_obstacles(
        self,
        num: int,
        area: Tuple[float, float, float, float],
        radius_range: Tuple[float, float],
        check_robot: bool = True,
        check_overlap: bool = True,
        max_attempts_per_obstacle: int = 1000
    ) -> int:
        """
        Генерирует и добавляет заданное количество препятствий со случайными параметрами.

        Параметры:
            num – количество препятствий для добавления
            area – область появления (xmin, xmax, ymin, ymax)
            radius_range – диапазон радиусов (rmin, rmax)
            check_robot – проверять, чтобы препятствие не пересекалось с роботом
            check_overlap – проверять пересечения между препятствиями
            max_attempts_per_obstacle – максимальное число попыток для генерации одного препятствия

        Возвращает:
            количество успешно добавленных препятствий
        """
        if self.obstacle_manager is None:
            self.obstacle_manager = ObstacleManager([])

        added = 0
        attempts = 0
        existing_obstacles = self.obstacle_manager.obstacles.copy()

        while added < num and attempts < num * max_attempts_per_obstacle:
            attempts += 1
            cfg = random_obstacle_config(area, radius_range, rng=self._rng)
            pos = np.asarray(cfg.position)
            r = cfg.radius

            if check_robot and self._obstacle_collides_with_robot(pos, r):
                continue

            if check_overlap and self._obstacle_collides_with_others(pos, r, existing_obstacles):
                continue

            new_obs = Obstacle(cfg)
            self.obstacle_manager.obstacles.append(new_obs)
            existing_obstacles.append(new_obs)
            added += 1

        return added


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).reshape(2)
    a = np.asarray(a, dtype=float).reshape(2)
    b = np.asarray(b, dtype=float).reshape(2)

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")