from __future__ import annotations

from .state import State
from .config import ModelConfig

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from typing import Any, Dict, List, Optional, Tuple, Union


ADAPTER_DIM = 64


class GaussianMLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: ModelConfig, action_limits) -> None:
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        in_dim = obs_dim  # сразу используем obs_dim

        for h in self.cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        self.net = nn.Sequential(*layers)

        self.mu_head = nn.Linear(in_dim, act_dim)
        self.value_head = nn.Linear(in_dim, 1)

        nn.init.uniform_(self.mu_head.weight, -0.1, 0.1)
        nn.init.uniform_(self.mu_head.bias, -0.1, 0.1)

        self.register_buffer(
            "action_limits",
            torch.tensor(action_limits, dtype=torch.float32)
        )

        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.net(x)
        mu = torch.tanh(self.mu_head(z)) * self.action_limits
        log_std = torch.clamp(self.log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        #print(f"{torch.exp(log_std)}")
        sigma = torch.exp(log_std) * self.action_limits
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
        max_steps: int = 400,
        policy: Optional[nn.Module] = None,
        device: Optional[str] = None,
        
    ) -> None:
        self.cfg = cfg
        if not (0.0 < self.cfg.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1).")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if policy is None:
            policy = GaussianMLPPolicy(
                obs_dim=obs_dim, act_dim=act_dim, cfg=self.cfg, action_limits=action_limit
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
        self._values: List[torch.Tensor] = []
        self._steps: int = 0

        self.buffer_states: List[torch.Tensor] = []
        self.buffer_actions: List[torch.Tensor] = []
        self.buffer_log_probs: List[torch.Tensor] = []
        self.buffer_td_targets: List[float] = []   # Для критика
        self.buffer_advantages: List[float] = []   # Для актора (GAE)
        self.buffer_step_count = 0

        self.gae_lambda = getattr(self.cfg, "gae_lambda", 0.95)

        self.train: Dict[str, List[float]] = {
            "total_reward": [], "success": [], "collision": [], "steps": [],
            "final_distance": [], "loss": [], "value_loss": [], "grad_norm": [], "kl_div": [],
            "sigma_mean": [], "sigma_joint_0": [], "sigma_joint_1": [],
            "sigma_joint_2": [], "entropy": [], "angle_error": [],
        }

        # Per-step data (grows every env step, not per episode)
        self.step_torques: List[List[float]] = []   # each entry: [tau_j0, tau_j1, tau_j2]
        self.step_joint_vels: List[List[float]] = []  # each entry: [dq_j0, dq_j1, dq_j2]

        self.test: Dict[str, List[float]] = {
            "success": [],
            "collision": [],
            "final_distance": [],
            "steps": [],
        }

    def upgrade_obs_dim(self, new_obs_dim: int) -> None:
        """Replace adapter for a larger observation space, keeping trunk weights."""
        self.policy.adapter = nn.Sequential(
            nn.Linear(new_obs_dim, ADAPTER_DIM), nn.ReLU()
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(self.cfg.lr_start))

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

    def observe_step(self, torques: np.ndarray, joint_vels: np.ndarray) -> None:
        self.step_torques.append([float(torques[i]) for i in range(len(torques))])
        self.step_joint_vels.append([float(joint_vels[i]) for i in range(len(joint_vels))])

    def should_update(self) -> bool:
        return self.buffer_step_count >= self.cfg.batch_size_limit

    def finish_episode(
        self, *, success: bool, collision: bool = False, final_distance: Optional[float] = None,
        angle_error: Optional[float] = None,
    ) -> Dict[str, float]:
        total_reward = float(sum(self._rewards))

        if not self._rewards or not self._log_probs or len(self._values) < 5:
            metrics = {
                "total_reward": 0.0,
                "success": float(bool(success)),
                "collision": float(bool(collision)),
                "steps": 0.0,
                "final_distance": (
                    float(final_distance) if final_distance is not None else float("nan")
                ),
                "loss": float("nan"),
                "value_loss": float("nan"),
                "grad_norm": float("nan"),
                "kl_div": float("nan"),
                "sigma_mean": float("nan"),
                "sigma_joint_0": float("nan"),
                "sigma_joint_1": float("nan"),
                "sigma_joint_2": float("nan"),
                "entropy": float("nan"),
                "angle_error": float(angle_error) if angle_error is not None else float("nan"),
            }
            self._append_train(metrics)
            return metrics

        ep_rewards = torch.tensor(self._rewards, dtype=torch.float32)
        ep_values = torch.cat(self._values).squeeze(-1) # shape: (T,)

        next_values = torch.zeros_like(ep_values)
        next_values[:-1] = ep_values[1:]
        gamma = self.cfg.gamma
        lam = self.gae_lambda

        advantages = torch.zeros_like(ep_rewards)
        td_targets = torch.zeros_like(ep_rewards)
        gae = 0.0

        for t in reversed(range(len(ep_rewards))):
            # mask = 0 (done) for the last step, 1 otherwise
            mask = 0.0 if t == len(ep_rewards) - 1 else 1.0
            
            # target_t = r_t + γ·V(s_{t+1})·(1-done_t)
            td_targets[t] = ep_rewards[t] + gamma * next_values[t] * mask
            
            # A_t = δ_t + γλ·A_{t+1}
            delta = ep_rewards[t] + gamma * next_values[t] * mask - ep_values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        
        #print("adv mean:", advantages.mean().item())
        #print("adv std:", advantages.std().item())
        self.buffer_states.extend(self._states)
        self.buffer_actions.extend(self._actions)
        self.buffer_log_probs.extend(self._log_probs)
        self.buffer_td_targets.extend(td_targets.tolist())
        self.buffer_advantages.extend(advantages.tolist())
        self.buffer_step_count += len(self._rewards)

        metrics = {
            "total_reward": total_reward,
            "success": float(bool(success)),
            "collision": float(bool(collision)),
            "steps": float(len(self._rewards)),
            "final_distance": float(final_distance) if final_distance is not None else float("nan"),
            "loss": float("nan"),
            "value_loss": float("nan"),
            "grad_norm": float("nan"),
            "kl_div": float("nan"),
            "sigma_mean": float("nan"),
            "sigma_joint_0": float("nan"),
            "sigma_joint_1": float("nan"),
            "sigma_joint_2": float("nan"),
            "entropy": float("nan"),
            "angle_error": float(angle_error) if angle_error is not None else float("nan"),
        }

        if self.should_update():
            try:
                ppo_metrics = self._update_ppo()
            except Exception as e:
                print("PPO update failed:", e)
                self.save("ppo_failure.pt", include_optimizer=True, include_metrics=True)
                # Можно очистить буферы, чтобы продолжить тренироваться дальше или аккуратно остановиться
                self._clear_batch_buffers()
                ppo_metrics = {
                    "loss": float("nan"),
                    "value_loss": float("nan"),
                    "grad_norm": float("nan"),
                    "kl_div": float("nan"),
                    "sigma_mean": float("nan"),
                    "sigma_joint_0": float("nan"),
                    "sigma_joint_1": float("nan"),
                    "sigma_joint_2": float("nan"),
                    "entropy": float("nan"),
                }
            metrics.update(ppo_metrics)
            self._clear_batch_buffers()

        self._append_train(metrics)
        return metrics

    def _clear_batch_buffers(self) -> None:
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_log_probs.clear()
        self.buffer_td_targets.clear()   # Исправлено: очищаем правильные буферы
        self.buffer_advantages.clear()   # Исправлено
        self.buffer_step_count = 0

    def _update_ppo(self) -> Dict[str, float]:
        b_states = torch.stack(self.buffer_states).detach().to(self.device)
        b_actions = torch.stack(self.buffer_actions).detach().to(self.device)
        b_log_probs = torch.stack(self.buffer_log_probs).detach().to(self.device)
        b_td_targets = torch.tensor(self.buffer_td_targets, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(self.buffer_advantages, dtype=torch.float32).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # print(f"adv mean={advantages.mean():.3f}  std={advantages.std():.3f}  "
        # f"min={advantages.min():.3f}  max={advantages.max():.3f}")
        batch_size = b_states.shape[0]
        mbs = min(self.cfg.mini_batch_size, batch_size)
        eps = self.cfg.clip_epsilon

        epoch_losses = []
        epoch_value_losses = []
        epoch_kls = []
        grad_norms = []
        epoch_entropies = []
        final_sigmas = None

        for i in range(self.cfg.ppo_epochs):
            #print(f"{i} epoch")
            with torch.no_grad():
                mu_full, sigma_full, values_full = self.policy(b_states)
                var_y = torch.var(b_td_targets)
                explained_var = 1 - torch.var(b_td_targets - values_full.squeeze(-1)) / (var_y + 1e-8)
                #print(f"explained_variance: {explained_var.item():.3f}")
                dist_full = Normal(mu_full, sigma_full)
                new_lp_full = dist_full.log_prob(b_actions).sum(dim=-1)
                log_ratio_full = new_lp_full - b_log_probs
                ratio_full = torch.exp(log_ratio_full)
                approx_kl = ((ratio_full - 1) - log_ratio_full).mean().item()
                epoch_kls.append(approx_kl)
                #print(f"KL={approx_kl:.5f}")
                final_sigmas = sigma_full.detach().cpu().numpy().flatten()
                #print(final_sigmas)
                epoch_entropies.append(dist_full.entropy().sum(dim=-1).mean().item())
            if approx_kl > self.cfg.target_kl:
                print(f"break after {i} epoches")
                break
            if i == 9:
                print(f"All epoch")
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mbs):
                mb_idx = indices[start : start + mbs]
                mb_states = b_states[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_log_probs = b_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_td_targets = b_td_targets[mb_idx]

                mu, sigma, values = self.policy(mb_states)
                values = values.squeeze(-1) 

                dist = Normal(mu, sigma)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)

                log_ratio = new_log_probs - mb_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                # Normalize TD targets so value loss stays on the same scale as policy loss
                mb_td_targets_norm = (mb_td_targets - mb_td_targets.mean()) / (mb_td_targets.std() + 1e-8)
                values_norm = (values - mb_td_targets.mean()) / (mb_td_targets.std() + 1e-8)
                value_loss = nn.functional.mse_loss(values_norm, mb_td_targets_norm)

                entropy = dist.entropy().sum(dim=-1).mean()
                loss = policy_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy
                #print(f"policy: {policy_loss} value: {self.cfg.value_loss_coef * value_loss} entropy: {self.cfg.entropy_coef * entropy}")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                #print("grad log_std:", self.policy.log_std.grad)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.grad_clip_norm
                )
                self.optimizer.step()

                epoch_losses.append(loss.item())
                epoch_value_losses.append(value_loss.item())
                grad_norms.append(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
        self.scheduler.step()
        
        #print(final_sigmas)
        with torch.no_grad():
            sigmas_per_joint = (torch.exp(
                torch.clamp(self.policy.log_std, self.cfg.log_std_min, self.cfg.log_std_max)
            ) * self.policy.action_limits).cpu().numpy()

        return {
            "loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "value_loss": float(np.mean(epoch_value_losses)) if epoch_value_losses else float("nan"),
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

    def get_train_metrics(self) -> Dict[str, Any]:
        n_joints = len(self.step_torques[0]) if self.step_torques else 0
        torques_by_joint = [
            [self.step_torques[s][j] for s in range(len(self.step_torques))]
            for j in range(n_joints)
        ]
        vels_by_joint = [
            [self.step_joint_vels[s][j] for s in range(len(self.step_joint_vels))]
            for j in range(n_joints)
        ]
        return {
            **self.train,
            "step_torques": torques_by_joint,
            "step_joint_vels": vels_by_joint,
        }

    def get_test_metrics(self) -> Dict[str, List[float]]:
        return self.test

    def _append_train(self, m: Dict[str, float]) -> None:
        for k in self.train.keys():
            self.train[k].append(float(m.get(k, float("nan"))))

    def _to_tensor(self, state: Union[np.ndarray, State]) -> torch.Tensor:
        if not isinstance(state, np.ndarray):
            state = np.asarray(state, dtype=np.float32)
        return torch.from_numpy(state).float().unsqueeze(0).to(self.device)

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
