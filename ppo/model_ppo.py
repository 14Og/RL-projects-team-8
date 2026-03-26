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
        self._rewards.clear()
        self._steps = 0

    def select_action(self, state: Union[np.ndarray, State], *, train: bool = True) -> np.ndarray:
        s = self._to_tensor(state)
        mu, sigma = self.policy(s)
        dist = Normal(mu, sigma)

        if train:
            u = dist.sample()
            logp = dist.log_prob(u).sum(dim=-1)
            self._states.append(s.squeeze(0))
            self._actions.append(u.squeeze(0))
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

        returns = self._discounted_returns(self._rewards, self.cfg.gamma).to(self.device)
        episode_return = float(returns[0].item())

        self.buffer_states.extend(self._states)
        self.buffer_actions.extend(self._actions)
        self.buffer_log_probs.extend(self._log_probs)
        self.buffer_rewards.extend(returns.tolist())
        self.buffer_step_count += len(self._rewards)

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
        print("Clearing PPO buffers...")
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_log_probs.clear()
        self.buffer_rewards.clear()
        self.buffer_terminals.clear()
        self.buffer_step_count = 0

    def _update_ppo(self) -> Dict[str, float]:
        b_states = torch.stack(self.buffer_states).detach().to(self.device)
        b_actions = torch.stack(self.buffer_actions).detach().to(self.device)
        b_log_probs = torch.stack(self.buffer_log_probs).detach().to(self.device)
        b_returns = torch.tensor(self.buffer_rewards, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            advantages = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)

        batch_size = b_states.shape[0]
        mbs = min(self.cfg.mini_batch_size, batch_size)
        eps = self.cfg.clip_epsilon

        epoch_losses = []
        epoch_kls = []
        grad_norms = []
        epoch_entropies = []
        final_sigmas = None

        for _ in range(self.cfg.ppo_epochs):
            with torch.no_grad():
                mu_full, sigma_full = self.policy(b_states)
                dist_full = Normal(mu_full, sigma_full)
                new_lp_full = dist_full.log_prob(b_actions).sum(dim=-1)
                log_ratio_full = new_lp_full - b_log_probs
                ratio_full = torch.exp(log_ratio_full)
                approx_kl = ((ratio_full - 1) - log_ratio_full).mean().item()
                epoch_kls.append(approx_kl)
                final_sigmas = sigma_full.mean(dim=0).cpu().numpy()
                epoch_entropies.append(dist_full.entropy().sum(dim=-1).mean().item())

            if approx_kl > self.cfg.target_kl:
                break

            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mbs):
                mb_idx = indices[start : start + mbs]
                mb_states = b_states[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_log_probs = b_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]

                mu, sigma = self.policy(mb_states)
                dist = Normal(mu, sigma)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)

                log_ratio = new_log_probs - mb_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy = dist.entropy().sum(dim=-1).mean()
                loss = policy_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.grad_clip_norm
                )
                self.optimizer.step()

                epoch_losses.append(loss.item())
                grad_norms.append(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

        self.scheduler.step()

        sigmas_per_joint = final_sigmas if final_sigmas is not None else np.zeros(3)
        return {
            "loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
            "kl_div": float(np.mean(epoch_kls)) if epoch_kls else float("nan"),
            "sigma_mean": float(sigmas_per_joint.mean()),
            "sigma_joint_0": float(sigmas_per_joint[0]),
            "sigma_joint_1": float(sigmas_per_joint[1]),
            "sigma_joint_2": float(sigmas_per_joint[2]),
            "entropy": float(np.mean(epoch_entropies)) if epoch_entropies else float("nan"),
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
