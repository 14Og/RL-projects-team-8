from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from .config import EnvConfig, LidarConfig, ObstacleConfig, RewardConfig, RobotConfig
from .obstacle import Obstacle, ObstacleManager
from .robot import Robot
from .state import State
from .model_ppo import Model


class Environment:
    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        robot_cfg: RobotConfig,
        lidar_cfg: LidarConfig,
        model: Model,
        obstacle_cfg: ObstacleConfig,
        seed: int = 42,
    ) -> None:
        self.env_cfg = env_cfg
        self.rew_cfg = reward_cfg
        self.target = np.asarray(env_cfg.target_xy, dtype=np.float32)

        self.obstacle_manager = ObstacleManager(obstacle_cfg)
        self.robot = Robot(robot_cfg, lidar_cfg, self.obstacle_manager.obstacles, seed=seed)
        self.model = model
        self._rng = np.random.default_rng(seed)

        base = np.asarray(robot_cfg.base_xy, dtype=np.float32)
        links = np.asarray(robot_cfg.link_lengths, dtype=float)
        self._base = base
        self._reach_max = float(np.sum(links))
        self._reach_min = float(max(links[0] * 0.8, links[0] - np.sum(links[1:])))

        self._collision_threshold: float = 0.05

        self._train_mode: bool = True
        self._needs_reset: bool = True

        self.steps: int = 0
        self.done: bool = False
        self.success: bool = False
        self.reason: str = "not_started"
        self._prev_dist: float = float("nan")

        self._last_state: State = None
        self._prev_action = None
        self._curr_action = None

        # Stagnation tracking
        self._dist_history: List[float] = []
        self._stagnation_steps: int = 0

    def check_collision(self, joints: np.ndarray) -> bool:
        """Check if any robot link segment collides with any obstacle."""
        for obs in self.obstacle_manager.obstacles:
            r = obs.radius + self._collision_threshold
            for i in range(len(joints) - 1):
                p1, p2 = joints[i], joints[i + 1]
                vec = p2 - p1
                t = np.clip(np.dot(obs.center - p1, vec) / (np.dot(vec, vec) + 1e-6), 0, 1)
                if np.linalg.norm(obs.center - (p1 + t * vec)) <= r:
                    return True
        return False

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

        self._dist_history.clear()
        self._stagnation_steps = 0

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
        st, self._curr_action = self.robot.step(u)
        self.steps += 1

        reward, done, info = self._compute_reward_and_done(st)

        if self._train_mode:
            self.model.observe(reward)

        self.done = bool(done)

        if self.done:
            final_dist = float(info.get("final_distance", float("nan")))
            self.success = bool(info.get("success", False))
            self.reason = str(info.get("reason", "done"))

            if self._train_mode:
                collision = info.get("reason") == "collision"
                self.model.finish_episode(success=self.success, collision=collision, final_distance=final_dist)
            else:
                self.model.record_test_episode(
                    success=self.success, final_distance=final_dist, steps=int(self.steps)
                )

            self._needs_reset = True

        self._last_state = st
        self._prev_action = self._curr_action
        return np.asarray(st, dtype=np.float32), float(reward), bool(self.done), info

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

        return {
            "joints": joints,
            "end_effector": ee,
            "target": self.target.copy(),
            "distance": dist,
            "theta": self.robot.theta,
            "obstacles": self.obstacle_manager.get_render_data(),
            "lidar": lidar_data,
            "step": int(self.steps),
            "done": bool(self.done),
            "success": bool(self.success),
            "reason": self.reason,
            "train_mode": bool(self._train_mode),
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "train": self.model.get_train_metrics(),
            "test": self.model.get_test_metrics(),
        }

    def _is_target_valid(self, target: np.ndarray, ee_start: np.ndarray) -> bool:
        dist_from_base = float(np.linalg.norm(target - self._base))
        if dist_from_base < self._reach_min - 1e-6 or dist_from_base > self._reach_max + 1e-6:
            return False
        for obs in self.obstacle_manager.obstacles:
            if float(np.linalg.norm(target - obs.center)) < obs.radius - 1e-6:
                return False
        if self.env_cfg.min_target_distance_from_ee > 0:
            if float(np.linalg.norm(target - ee_start)) < self.env_cfg.min_target_distance_from_ee:
                return False
        if self.env_cfg.target_line_of_sight:
            for obs in self.obstacle_manager.obstacles:
                if _point_to_segment_distance(obs.center, self._base, target) < obs.radius - 1e-6:
                    return False
        return True

    def _sample_valid_target(self, ee_start: np.ndarray, max_attempts: int = 1000) -> np.ndarray:
        for _ in range(max_attempts):
            angle = self._rng.uniform(0, 2 * np.pi)
            r = np.sqrt(self._rng.uniform(self._reach_min ** 2, self._reach_max ** 2))
            candidate = self._base + np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
            if self._is_target_valid(candidate, ee_start):
                return candidate
        import warnings
        warnings.warn("Could not sample a valid target after many attempts, using fallback.", RuntimeWarning)
        return self._base + np.array([self._reach_max * 0.5, 0.0], dtype=np.float32)

    def _get_state(self) -> State:
        return self.robot.obs()

    def _compute_reward_and_done(self, current_state: State) -> Tuple[float, bool, Dict[str, Any]]:
        joints = self.robot.joints_xy()
        ee = joints[-1]
        dist = float(np.linalg.norm(ee - self.target))

        # 1. Base reward for approaching the target and step penalty
        progress = float(self._prev_dist - dist)
        reward = float(self.rew_cfg.progress_scale) * progress - float(self.rew_cfg.step_penalty)

        # 1b. Distance-adaptive progress boost — amplify signal near the goal
        if dist < self.rew_cfg.progress_boost_radius:
            boost = 1.0 + self.rew_cfg.progress_near_boost * (
                1.0 - dist / self.rew_cfg.progress_boost_radius
            )
            reward += float(self.rew_cfg.progress_scale) * progress * (boost - 1.0)

        # 2. Joint velocity penalty (squared velocities)
        if self._curr_action is not None and self.rew_cfg.joint_velocity_scale != 0.0:
            vel_sq = float(np.dot(self._curr_action, self._curr_action))
            reward -= float(self.rew_cfg.joint_velocity_scale) * vel_sq

        # 3. Action delta / smoothness penalty
        if self._curr_action is not None and self._prev_action is not None and self.rew_cfg.action_delta_scale != 0.0:
            delta_a = self._curr_action - self._prev_action
            accel_sq = float(np.dot(delta_a, delta_a))
            reward -= float(self.rew_cfg.action_delta_scale) * accel_sq

        # 4. Lidar-based obstacle proximity penalty (per-lidar, smooth)
        #    For each lidar, take the closest reading; if below threshold
        #    apply a squared penalty — smooth, bounded, one value per lidar.
        all_readings = current_state.lidar_rays
        n_rays = self.robot.lidar_manager.cfg.num_rays
        n_lidars = self.robot.lidar_manager.n_lidars
        danger_threshold = float(self.rew_cfg.obstacle_danger_threshold)
        for i in range(n_lidars):
            lidar_min = float(np.min(all_readings[i * n_rays : (i + 1) * n_rays]))
            if lidar_min < danger_threshold:
                # Quadratic ramp: 0 at threshold, 1 at contact
                proximity = (1.0 - lidar_min / danger_threshold) ** 2
                reward -= float(self.rew_cfg.obstacle_danger_penalty) * proximity
        
        # 5. Stagnation penalty — punish freezing in place
        win = self.rew_cfg.stagnation_window
        self._dist_history.append(dist)
        if len(self._dist_history) >= win:
            dist_change = abs(self._dist_history[-win] - dist)
            if dist_change < self.rew_cfg.stagnation_thresh:
                self._stagnation_steps += 1
                # Ramp up: the longer it stalls, the worse it gets
                ramp = min(self._stagnation_steps / 10.0, 3.0)
                reward -= self.rew_cfg.stagnation_penalty * ramp
            else:
                self._stagnation_steps = 0

        # Check collision
        collision = self.check_collision(joints)
        fail_reason = ""
        if collision:
            fail = True
            fail_reason = "collision"
            # Heavy penalty for collision
            reward -= float(self.rew_cfg.collision_penalty)
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
        stagnation_timeout = self._stagnation_steps >= self.env_cfg.stagnation_max
        if stagnation_timeout:
            fail = True
            fail_reason = fail_reason or "stagnation"
        done = goal_reached or fail or timeout

        info: Dict[str, Any] = {
            "success": bool(goal_reached),
            "final_distance": dist,
            "progress": progress,
            "timeout": bool(timeout),
            "fail": bool(fail),
            "reason": (
                "goal" if goal_reached else ("timeout" if timeout else (fail_reason or None))
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