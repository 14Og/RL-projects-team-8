from __future__ import annotations

from .state import State
from .config import ModelConfig

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Union


class GaussianMLPPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: ModelConfig,
        action_limit: float,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in self.cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, act_dim)
        self.log_std_head = nn.Linear(in_dim, act_dim)
        self.action_limit = float(action_limit)

        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.net(x)
        mu = torch.tanh(self.mu_head(z)) * self.action_limit
        log_std = torch.clamp(self.log_std_head(z), self.cfg.log_std_min, self.cfg.log_std_max)
        sigma = torch.exp(log_std) * self.action_limit
        return mu, sigma


class Model:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: ModelConfig,
        action_limit: float,
        train_episodes: int,
        policy: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        if not (0.0 < self.cfg.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if policy is None:
            policy = GaussianMLPPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                cfg=self.cfg,
                action_limit=action_limit
            )
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(self.cfg.lr_start))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(train_episodes), eta_min=float(self.cfg.lr_min)
        )

        self.baseline_buffer: Deque[float] = deque(maxlen=int(self.cfg.baseline_buf_len))

        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []
        self._steps: int = 0

        self.train: Dict[str, List[float]] = {
            "total_reward": [],
            "success": [],
            "collision": [],
            "steps": [],
            "final_distance": [],
            "loss": [],
            "baseline": [],
            "grad_norm": [],
        }

        self.test: Dict[str, List[float]] = {
            "success": [],
            "final_distance": [],
            "steps": [],
        }

    def start_episode(self) -> None:
        self._log_probs.clear()
        self._rewards.clear()
        self._steps = 0

    def select_action(self, state: Union[np.ndarray, State], *, train: bool = True) -> np.ndarray:
        s = self._to_tensor(state)
        mu, sigma = self.policy(s)
        dist = Normal(mu, sigma)

        if train:
            u = dist.sample()
            logp = dist.log_prob(u).sum(dim=-1)
            self._log_probs.append(logp.squeeze(0))
            self._steps += 1
            return u.squeeze(0).detach().cpu().numpy()

        return mu.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def act(self, state: Union[np.ndarray, State], *, deterministic: bool = True) -> np.ndarray:
        s = self._to_tensor(state)
        mu, sigma = self.policy(s)
        u = mu if deterministic else Normal(mu, sigma).sample()
        return u.squeeze(0).cpu().numpy()

    def observe(self, reward: float) -> None:
        self._rewards.append(float(reward))

    def finish_episode(self, *, success: bool, collision: bool = False, final_distance: Optional[float] = None) -> Dict[str, float]:
        total_reward = float(sum(self._rewards))
        baseline = float(np.mean(self.baseline_buffer)) if self.baseline_buffer else 0.0

        if not self._rewards or not self._log_probs:
            metrics = {
                "total_reward": total_reward,
                "success": float(bool(success)),
                "collision": float(bool(collision)),
                "steps": float(self._steps),
                "final_distance": float(final_distance) if final_distance is not None else float("nan"),
                "loss": float("nan"),
                "baseline": baseline,
                "grad_norm": float("nan"),
            }
            self._append_train(metrics)
            return metrics

        returns = self._discounted_returns(self._rewards, self.cfg.gamma).to(self.device)
        episode_return = float(returns[0].item())

        advantages = returns - baseline
        logps = torch.stack(self._log_probs).to(self.device)

        loss = -(logps * advantages).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        self.baseline_buffer.append(episode_return)

        metrics = {
            "total_reward": total_reward,
            "success": float(bool(success)),
            "collision": float(bool(collision)),
            "steps": float(self._steps),
            "final_distance": float(final_distance) if final_distance is not None else float("nan"),
            "loss": float(loss.item()),
            "baseline": baseline,
            "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
        }
        self._append_train(metrics)
        return metrics

    def record_test_episode(self, *, success: bool, final_distance: float, steps: int) -> None:
        self.test["success"].append(float(bool(success)))
        self.test["final_distance"].append(float(final_distance))
        self.test["steps"].append(float(steps))

    def get_train_metrics(self) -> Dict[str, List[float]]:
        return self.train

    def get_test_metrics(self) -> Dict[str, List[float]]:
        return self.test

    def _append_train(self, m: Dict[str, float]) -> None:
        for k in self.train.keys():
            self.train[k].append(float(m[k]))

    def _to_tensor(self, state: Union[np.ndarray, State]) -> torch.Tensor:
        if not isinstance(state, np.ndarray):
            state = np.asarray(state, dtype=np.float32)
        return torch.from_numpy(state).float().unsqueeze(0).to(self.device)

    @staticmethod
    def _discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
        out: List[float] = []
        R = 0.0
        for r in reversed(rewards):
            R = float(r) + float(gamma) * R
            out.append(R)
        out.reverse()
        return torch.tensor(out, dtype=torch.float32)

    def save(self, path: str, *, include_optimizer: bool = False, include_metrics: bool = False) -> None:
        ckpt = {
            "policy_state_dict": self.policy.state_dict(),
        }
        if include_optimizer:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
        if include_metrics:
            ckpt["train_metrics"] = self.train
            ckpt["test_metrics"] = self.test
            ckpt["baseline_buffer"] = list(self.baseline_buffer)
        torch.save(ckpt, path)

    def load(self, path: str, *, load_optimizer: bool = False, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if "policy_state_dict" not in ckpt:
            raise ValueError("Checkpoint missing 'policy_state_dict'.")
        self.policy.load_state_dict(ckpt["policy_state_dict"], strict=strict)
        if load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def set_train_mode(self) -> None:
        self.policy.train()

    def set_eval_mode(self) -> None:
        self.policy.eval()


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")