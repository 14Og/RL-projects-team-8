# Actor-Critic PPO - 3-DOF Planar Manipulator with Dynamic Obstacle Avoidance

Training a three-link planar manipulator with LIDAR-like perception using Actor-Critic PPO + TD + GAE, with curriculum from static tasks to dynamic targets/obstacles and full torque-level control.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

---

## Problem Definition

A three-link (3-DOF) planar manipulator is mounted on a fixed base.
The policy controls joint torques and must move the end-effector to a target while avoiding circular obstacles that can move over time.

Project workflow (per report): training started from simpler static setups and then moved to dynamic scenarios (moving targets/obstacles) to improve generalization.

| Property | Value |
|---|---|
| Link lengths | L1 = 90, L2 = 70, L3 = 40 (px) |
| Action constraint | tau in [-tau_max, +tau_max], tau_max = (12, 8, 8) |
| Goal condition | End-effector within 30 px of target |
| Max episode length | 600 steps |
| Obstacles (default) | 2 circles, radius 30 px |
| Obstacle dynamics | Elliptic motion + per-episode jitter |

The environment is deterministic given current state and action. Randomness is introduced by episode reset: obstacle jitter/phase, and optional randomization of target and initial joint angles.

---

## Environment

### State Space

The observation depends on `EnvConfig.obs_mode`:

- `base`: 13 dims
- `static`: 41 dims (with 2 obstacles)
- `dynamic`: 45 dims (default, with 2 obstacles)

Base features:

- `sin(theta_i), cos(theta_i)` for 3 joints -> 6
- End-effector position `(x_ee, y_ee)` normalized by reach -> 2
- Relative target vector `(dx, dy)` normalized by reach -> 2
- Joint velocities `dq / 15` -> 3

Additional features:

- LIDAR rays: `n_lidars * num_rays = 3 * 8 = 24`
- Obstacle relative positions (per obstacle): 2
- Obstacle velocities (only in `dynamic` mode, per obstacle): 2

### LIDAR Perception

LIDAR sensors are attached to all joints:

- 8 rays per sensor, uniformly over `[0, 2pi)`
- Maximum ray length: 50 px
- Reading is normalized to `[0, 1]`
- 1.0 means no hit, lower values indicate proximity to obstacle

### Action Space

Continuous 3D torque vector:

`a_t = (tau_1, tau_2, tau_3)`

Actions are sampled from a diagonal Gaussian policy in training mode and clipped by torque limits in the robot dynamics step.

### Obstacles

Obstacle manager supports randomization and motion:

- Per episode: each obstacle origin is jittered in a disk (`jitter_radius = 30`)
- During episode: each obstacle moves on an ellipse:
  - `x = x0 + a * cos(omega * t + phase)`
  - `y = y0 + b * sin(omega * t + phase)`

Default parameters:

| Parameter | Value |
|---|---|
| Number of obstacles | 2 |
| Radius | 30 px |
| Jitter radius | 30 px |
| Dynamic motion | enabled |
| Ellipse semi-axes | a = 80, b = 60 |
| Angular frequency | omega = 0.1 |

### Termination Conditions

| Condition | Type | Reward signal |
|---|---|---|
| End-effector within `target_thresh` | Success | `+goal_reward` |
| Collision with obstacle | Failure | `-collision_penalty`, `-fail_penalty` |
| Stagnation window criterion | Failure | `-fail_penalty` |
| Step count reaches `max_steps` | Timeout | `-fail_penalty` |

### Reward Function

Reward is composed of:

- progress term (distance reduction)
- step penalty
- torque penalty
- velocity penalty near danger zone
- obstacle danger penalty from LIDAR
- terminal bonuses/penalties

Config defaults (`RewardConfig`):

| Component | Parameter | Default |
|---|---|---|
| Progress scale | `progress_scale` | 0.9 |
| Near-goal boost | `progress_near_boost` | 0.2 |
| Boost radius | `progress_boost_radius` | 60.0 |
| Step penalty | `step_penalty` | 0.08 |
| Velocity penalty | `vel_penalty` | 0.08 |
| Torque penalty | `torque_penalty` | 0.05 |
| Obstacle danger threshold | `obstacle_danger_threshold` | 0.7 |
| Obstacle danger penalty | `obstacle_danger_penalty` | 2 |
| Collision penalty | `collision_penalty` | 60.0 |
| Goal reward | `goal_reward` | 80.0 |
| Fail/timeout penalty | `fail_penalty` | 40.0 |

---

## PPO Algorithm

Current training uses Actor-Critic PPO (`ppo/model_actor_critic_ppo.py`):

- Shared MLP backbone (`256 -> 128`, ReLU)
- Actor head: Gaussian mean (`tanh`-bounded by torque limits) + learnable `log_std`
- Critic head: scalar value estimate `V(s)`
- On-policy buffer with batch updates
- PPO clipped objective + value loss + entropy regularization
- GAE (`gae_lambda = 0.95`) for advantage estimation
- KL-based early stop for PPO epochs
- TD targets for critic: `r_t + gamma * V(s_{t+1}) * (1 - done_t)`

### Why Actor-Critic PPO

Compared to policy-gradient-only variants, current implementation:

1. Uses critic targets for lower-variance updates
2. Reuses trajectories via mini-batch PPO epochs
3. Constrains update size with clipping and KL monitoring
4. Optimizes exploration through entropy term

### Update Rule (implemented)

1. Collect episode trajectories (states, actions, log_probs, rewards, values)
2. Compute TD targets and GAE advantages at episode end
3. Accumulate episodes until `batch_size_limit` (default 2048 steps)
4. Normalize advantages over the batch
5. For up to `ppo_epochs`:
   - estimate KL on full batch
   - stop early if KL > `target_kl`
   - optimize shuffled mini-batches
6. Step cosine annealing scheduler
7. Clear batch buffer

Loss used in code:

`loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy`

---

## Development History

### Milestone 1 - Base Actor-Critic + PPO

- Implemented shared backbone with heads `mu(s)`, `sigma(s)`, `V(s)`
- Added Gaussian policy and log-likelihood computation
- Implemented trajectory buffer, PPO clipping, value loss, entropy bonus
- Added KL monitoring and early stopping by `target_kl`

### Milestone 2 - Transition to TD + GAE

- Switched critic learning to TD targets: `r + gamma * V(s')`
- Implemented GAE recursion for actor advantages
- Added batch normalization of advantages

### Milestone 3 - Realistic Physics and Pure Torque Control

- Removed PD-assist layer and switched to direct torque control
- Added manipulator dynamics terms: inertia, Coriolis, gravity
- Added gravity compensation in control signal

### Milestone 4 - Stabilizing Sigma and Losses

- Diagnosed KL spikes and overly aggressive PPO truncation
- Tracked `policy_loss`, `value_loss`, and `entropy` separately
- Normalized/rescaled value targets to prevent `value_loss` domination
- Stabilized KL and enabled meaningful `sigma` decay during learning

### Milestone 5 - From Static Configuration to Dynamic Tasks

- Static-stage training reached high success but overfit to one setup
- Extended state with obstacle positions and velocities
- Moved to training directly on moving targets/obstacles for better generalization

### Milestone 6 - Final Integration (planned)

- Formalize unified pipeline: `PPO + TD + GAE + manipulator physics`
- Consolidate report metrics: reward, success, episode length, sigma, KL, loss components
- Finalize structured write-up sections: Methods, Experiments, Results, Limitations

### Visual Milestones

<p align="center">
  <img src="assets/test_static_obs_determined_start.gif" alt="Test - static obstacles" width="400">
</p>

<p align="center">
  <img src="assets/test_random_obs_determined_start.gif" alt="Test - random obstacles" width="400">
</p>

<p align="center">
  <img src="assets/test_random_obs_random_start.gif" alt="Test - random obstacles, random start" width="400">
</p>

<p align="center">
  <img src="assets/test_metrics_static_obs_determined_start.gif" alt="Metrics - static/determined" width="33%">
  <img src="assets/test_metrics_random_obs_determined_start.gif" alt="Metrics - random/determined" width="33%">
  <img src="assets/test_metrics_random_obs_random_start.gif" alt="Metrics - random/random" width="33%">
</p>

---

## Project Structure

```
|-- main.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- assets/
|-- policy/
|   |-- best_policy.pt
|   |-- best_policy_const.pt
|   `-- policy_reinforce.pt
`-- ppo/
    |-- __init__.py
    |-- config.py
    |-- state.py
    |-- robot.py
    |-- physics_robot.py
    |-- lidar.py
    |-- obstacle.py
    |-- env.py
    |-- runner.py
    |-- gui.py
    |-- fast_math.py
    |-- model_actor_critic_ppo.py
    |-- model_ppo.py
    `-- model_reinforce.py
```

`main.py` currently builds and runs `model_actor_critic_ppo.py`.

---

## Hyperparameters

From `ModelConfig`, `EnvConfig`, and `GUIConfig` defaults:

| Parameter | Value |
|---|---|
| Hidden layers | 256 -> 128 (ReLU) |
| Discount gamma | 0.97 |
| GAE lambda | 0.95 |
| Learning rate | 3e-4 -> 1e-6 (cosine annealing) |
| Clip coefficient epsilon | 0.2 |
| PPO epochs per update | 10 |
| Mini-batch size | 512 |
| Batch size limit | 2048 steps |
| Target KL | 0.15 |
| Entropy coefficient | 0.002 |
| Value loss coefficient | 0.1 |
| Gradient clip norm | 1.0 |
| log_std range | [-3.0, -0.7] |
| Max episode steps | 600 |
| Training episodes | 3000 |
| Test episodes | 500 |

Values referenced in project report (`text.txt`) for algorithm discussion:

- `gamma = 0.99`
- `lambda = 0.95`
- `clip epsilon = 0.2`
- `entropy coef = 0.01`
- `ppo epochs = 10`
- `batch buffer = 2048`
- `mini-batch = 256`

---

## Reproducibility (implemented using python3.12)

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python main.py --train
```

Headless training (without pygame sim window):

```bash
python main.py --train --no-sim
```

Optional curriculum/randomization flags:

```bash
python main.py --train --randomize-target --randomize-theta
```

Optional finetuning from checkpoint:

```bash
python main.py --train --finetune policy/best_policy_const.pt
```

### Test

```bash
python main.py --test
```

### Additional Options

```
--model-path PATH
--train-episodes N
--test-episodes N
--seed N
--no-sim
--finetune WEIGHTS_PATH
--randomize-target
--randomize-theta
```

### Output

- Live simulation + metrics (pygame mode)
- Live matplotlib metrics (headless mode)
- Saved model checkpoint (default: `policy/best_policy.pt`)

---

## Collected Metrics

Training plots include:

- Total reward
- Applied torques per joint
- Value loss
- KL divergence
- Sigma per joint
- Collision rate
- Entropy
- Success rate

Test plots include:

- Cumulative success rate
- Cumulative collision rate
- Angle error trace (if available in metrics)
