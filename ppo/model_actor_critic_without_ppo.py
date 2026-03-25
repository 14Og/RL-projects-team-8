from __future__ import annotations

from .state import State
from .config import ModelConfig

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from typing import Dict, List, Optional, Tuple, Union


class GaussianMLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: ModelConfig, action_limit: float) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = obs_dim
        print(f"Building policy network with obs_dim={obs_dim}, act_dim={act_dim}, hidden_sizes={self.cfg.hidden_sizes}")
        print("model_actor_critic_without_ppo")
        for h in self.cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, act_dim)
        self.log_std_head = nn.Linear(in_dim, act_dim)
        self.value_head = nn.Linear(in_dim, 1) # critic 
        self.action_limit = float(action_limit)

        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.net(x)
        mu = torch.tanh(self.mu_head(z)) * self.action_limit
        log_std = torch.clamp(self.log_std_head(z), self.cfg.log_std_min, self.cfg.log_std_max)
        sigma = torch.exp(log_std) #* self.action_limit
        #print(f"DEBUG: log_std min/max: {log_std.min().item()}, {log_std.max().item()}")
        #critc
        value = self.value_head(z)
        return mu, sigma, value 


class Model:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: ModelConfig,
        action_limit: float,
        train_episodes: int,
        max_steps: int = 200,
        policy: Optional[nn.Module] = None,
        device: Optional[str] = None,
        
    ) -> None:
        self.cfg = cfg
        if not (0.0 < self.cfg.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if policy is None:
            policy = GaussianMLPPolicy(
                obs_dim=obs_dim, act_dim=act_dim, cfg=self.cfg, action_limit=action_limit
            )
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(self.cfg.lr_start))
        n_ppo_updates = max(1, (train_episodes * max_steps) // self.cfg.batch_size_limit)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_ppo_updates, eta_min=float(self.cfg.lr_min)
        )

        self._states: List[torch.Tensor] = []
        self._actions: List[torch.Tensor] = []
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []
        self._steps: int = 0

        self.buffer_states: List[torch.Tensor] = []
        self.buffer_actions: List[torch.Tensor] = []
        self.buffer_log_probs: List[torch.Tensor] = []
        self.buffer_rewards: List[float] = []
        self.buffer_terminals: List[bool] = []
        self.buffer_step_count = 0
        
        #save values for critic
        self._values: List[torch.Tensor] = []
        self.buffer_values: List[torch.Tensor] = []

        self.train: Dict[str, List[float]] = {
            "total_reward": [],
            "success": [],
            "collision": [],
            "steps": [],
            "final_distance": [],
            "loss": [],
            "grad_norm": [],
            "kl_div": [],
            "sigma_mean": [],
            "sigma_joint_0": [],
            "sigma_joint_1": [],
            "sigma_joint_2": [],
            "entropy": [],
        }

        self.test: Dict[str, List[float]] = {
            "success": [],
            "collision": [],
            "final_distance": [],
            "steps": [],
        }

    def start_episode(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear() 
        self._rewards.clear()
        self._steps = 0

    def select_action(self, state: Union[np.ndarray, State], *, train: bool = True) -> np.ndarray:
        s = self._to_tensor(state)
        mu, sigma, val = self.policy(s) # Get value from critic
        dist = Normal(mu, sigma)

        if train:
            u = dist.sample()
            logp = dist.log_prob(u).sum(dim=-1)
            self._states.append(s.squeeze(0))
            self._actions.append(u.squeeze(0))
            self._log_probs.append(logp.squeeze(0))
            self._values.append(val.squeeze(0).detach())
            self._steps += 1
            return u.squeeze(0).detach().cpu().numpy()

        return mu.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def act(self, state: Union[np.ndarray, State], *, deterministic: bool = True) -> np.ndarray:
        s = self._to_tensor(state)
        mu, sigma, values = self.policy(s)
        u = mu if deterministic else Normal(mu, sigma).sample()
        return u.squeeze(0).cpu().numpy()

    def observe(self, reward: float) -> None:
        self._rewards.append(float(reward))

    def should_update(self) -> bool:
        return self.buffer_step_count >= self.cfg.batch_size_limit

    def finish_episode(
        self, *, success: bool, collision: bool = False, final_distance: Optional[float] = None
    ) -> Dict[str, float]:
        total_reward = float(sum(self._rewards))

        if not self._rewards or not self._log_probs:
            metrics = {
                "total_reward": 0.0,
                "success": float(bool(success)),
                "collision": float(bool(collision)),
                "steps": 0.0,
                "final_distance": (
                    float(final_distance) if final_distance is not None else float("nan")
                ),
                "loss": float("nan"),
                "grad_norm": float("nan"),
                "kl_div": float("nan"),
                "sigma_mean": float("nan"),
                "sigma_joint_0": float("nan"),
                "sigma_joint_1": float("nan"),
                "sigma_joint_2": float("nan"),
                "entropy": float("nan"),
            }
            self._append_train(metrics)
            return metrics

        self.buffer_states.extend(self._states)
        self.buffer_actions.extend(self._actions)
        self.buffer_log_probs.extend(self._log_probs)
        self.buffer_rewards.extend(self._rewards)  # raw rewards for TD bootstrap
        self.buffer_terminals.extend([False] * (len(self._rewards) - 1) + [True])
        self.buffer_step_count += len(self._rewards)
        self.buffer_values.extend(self._values)

        metrics = {
            "total_reward": total_reward,
            "success": float(bool(success)),
            "collision": float(bool(collision)),
            "steps": float(len(self._rewards)),
            "final_distance": float(final_distance) if final_distance is not None else float("nan"),
            "loss": float("nan"),
            "grad_norm": float("nan"),
            "kl_div": float("nan"),
            "sigma_mean": float("nan"),
            "sigma_joint_0": float("nan"),
            "sigma_joint_1": float("nan"),
            "sigma_joint_2": float("nan"),
            "entropy": float("nan"),
        }

        if self.should_update():
            ppo_metrics = self._update_ppo()
            metrics.update(ppo_metrics)
            self._clear_batch_buffers()

        self._append_train(metrics)
        return metrics

    def _clear_batch_buffers(self) -> None:
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_log_probs.clear()
        self.buffer_rewards.clear()
        self.buffer_terminals.clear()
        self.buffer_values.clear()
        self.buffer_step_count = 0

    def _update_ppo(self) -> Dict[str, float]:
        b_states = torch.stack(self.buffer_states).detach().to(self.device)
        b_actions = torch.stack(self.buffer_actions).detach().to(self.device)
        b_rewards = torch.tensor(self.buffer_rewards, dtype=torch.float32).to(self.device)
        b_dones = torch.tensor(self.buffer_terminals, dtype=torch.float32).to(self.device)

        # TD(0) bootstrap: target = r + γ * V(s') * (1 - done)
        with torch.no_grad():
            next_states = torch.cat([b_states[1:], b_states[-1:]], dim=0)
            next_values = self.policy(next_states)[2].squeeze(-1)
            b_values = self.policy(b_states)[2].squeeze(-1)
            targets = b_rewards + self.cfg.gamma * next_values * (1.0 - b_dones)
            advantages = targets - b_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mu, sigma, values = self.policy(b_states)
        values = values.squeeze(-1)
        dist = Normal(mu, sigma)
        new_log_probs = dist.log_prob(b_actions).sum(dim=-1)

        policy_loss = -(new_log_probs * advantages).mean()
        value_loss = nn.functional.mse_loss(values, targets)
        entropy = dist.entropy().sum(dim=-1).mean()
        loss = policy_loss + 0.5 * value_loss - self.cfg.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        sigmas_per_joint = sigma.mean(dim=0).detach().cpu().numpy()
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "kl_div": float("nan"),
            "sigma_mean": float(sigmas_per_joint.mean()),
            "sigma_joint_0": float(sigmas_per_joint[0]),
            "sigma_joint_1": float(sigmas_per_joint[1]),
            "sigma_joint_2": float(sigmas_per_joint[2]),
            "entropy": entropy.item(),
        }

    def record_test_episode(
        self, *, success: bool, collision: bool, final_distance: float, steps: int
    ) -> None:
        self.test["success"].append(float(bool(success)))
        self.test["collision"].append(float(bool(collision)))
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

    def save(
        self, path: str, *, include_optimizer: bool = False, include_metrics: bool = False
    ) -> None:
        ckpt = {"policy_state_dict": self.policy.state_dict()}
        if include_optimizer:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
        if include_metrics:
            ckpt["train_metrics"] = self.train
            ckpt["test_metrics"] = self.test
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
