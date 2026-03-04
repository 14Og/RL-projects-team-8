# PPO — 3-DOF Planar Manipulator with Obstacle Avoidance

Training a three-link planar robot arm to reach arbitrary target points while avoiding obstacles, using Proximal Policy Optimization (PPO) with LIDAR-like perception.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

---

## Problem Definition

A **three-link (3-DOF) planar manipulator** is mounted on a fixed base.
The agent controls three joint velocities $\Delta\theta_1, \Delta\theta_2, \Delta\theta_3$ to move the end-effector toward a randomly placed target point while avoiding four circular obstacles scattered around the workspace.

| Property | Value |
|---|---|
| Link lengths | $L_1 = 100,\; L_2 = 70,\; L_3 = 40$ (px) |
| Action constraint | $\|\Delta\theta_i\| \le 0.3$ rad per step |
| Goal condition | End-effector within 30 px of target |
| Max episode length | 200 steps |
| Obstacles | 4 circles, radius 50 px, jittered ±40 px |

The environment is **deterministic**: given the same state and action, the next state is uniquely determined by forward kinematics. Randomness enters through initial joint angles, target position, and obstacle jitter — all sampled at the start of each episode.

---

## Environment

### State Space

A 34-dimensional real-valued vector:

$$s_t = \bigl(\underbrace{\sin\theta_1, \cos\theta_1, \sin\theta_2, \cos\theta_2, \sin\theta_3, \cos\theta_3}_{6},\; \underbrace{x_{ee}, y_{ee}}_{2},\; \underbrace{\Delta x, \Delta y}_{2},\; \underbrace{r_1, \dots, r_{24}}_{\text{LIDAR}}\bigr)$$

| Feature | Dim | Description |
|---|---|---|
| $\sin\theta_i, \cos\theta_i$ | 6 | Trigonometric encoding of joint angles avoids the $-\pi/\pi$ discontinuity |
| $x_{ee},\; y_{ee}$ | 2 | End-effector position centered on the base and normalized by maximum reach |
| $\Delta x,\; \Delta y$ | 2 | Signed distance from end-effector to target, normalized by maximum reach |
| LIDAR rays | 24 | 8 rays per joint × 3 joints; each value in $[0, 1]$ (see below) |

### LIDAR Perception

Without explicit obstacle information in the state, the agent cannot learn to avoid collisions. We added a LIDAR-like sensor system:

- **8 rays** are cast from each of the 3 joints, uniformly distributed over $[0, 2\pi)$
- Each ray extends up to 50 px; the reading is the normalized distance to the first obstacle intersection
- If a ray does not hit any obstacle, the reading is $1.0$; contact yields $0.0$
- This produces $3 \times 8 = 24$ additional state features

### Action Space

Continuous, 3-dimensional: $a_t = (\Delta\theta_1, \Delta\theta_2, \Delta\theta_3) \in [-0.3, 0.3]^3$ rad.

The policy outputs a mean (bounded via $\tanh$) and a per-joint standard deviation; an action is sampled from the resulting diagonal Gaussian and then hard-clipped by the robot to enforce the physical constraint.

### Obstacles

Four circular obstacles are placed in the workspace. At the start of each episode, each obstacle is jittered by a random offset (up to 40 px) around its equilibrium position. The agent must navigate around them to reach the target.

| Parameter | Value |
|---|---|
| Number of obstacles | 4 |
| Radius | 50 px |
| Jitter radius | 40 px |
| Equilibrium positions | $(200, 600)$, $(600, 600)$, $(400, 800)$, $(400, 400)$ |

### Termination Conditions

| Condition | Type | Reward signal |
|---|---|---|
| End-effector within `target_thresh` of target | **Success** | $+50$ |
| A link collides with an obstacle | **Collision** | $-10$ |
| A link segment passes through the target point | **Failure** | $-15$ |
| Stagnation detected (no progress) | **Failure** | $-15$ |
| Step count reaches `max_steps` | **Truncation** | $-15$ |

### Reward Function

$$r_t = \underbrace{\alpha \cdot \Delta d \cdot b(d)}_{\text{progress + near-goal boost}} - \underbrace{\beta}_{\text{step penalty}} - \underbrace{\sum_i p_i}_{\text{obstacle proximity}} + \underbrace{R_{\text{terminal}}}_{\text{goal / collision / fail}}$$

where the boost factor $b(d)$ amplifies the progress signal near the goal:

$$b(d) = 1 + k \cdot \left(1 - \frac{d}{R}\right) \quad \text{if } d < R, \qquad b(d) = 1 \quad \text{otherwise}$$

| Component | Parameter | Default | Description |
|---|---|---|---|
| Progress scale | $\alpha$ | 0.15 | Reward proportional to distance reduction $\Delta d = d_{t-1} - d_t$ |
| Near-goal boost factor | $k$ | 3.0 | Maximum extra multiplier on progress when $d \to 0$ |
| Boost radius | $R$ | 80 px | Distance threshold below which the boost activates |
| Step penalty | $\beta$ | 0.005 | Small per-step cost to encourage efficiency |
| Obstacle danger | $p_i$ | 0.05 | Quadratic ramp when LIDAR reading < 0.15 threshold |
| Collision penalty | $R_{\text{collision}}$ | 10.0 | Large penalty on direct obstacle contact |
| Goal bonus | $R_{\text{goal}}$ | +50.0 | Terminal reward for reaching the target |
| Failure / timeout / stagnation | $R_{\text{fail}}$ | −15.0 | Terminal penalty for failure, timeout, or stagnation |

**Obstacle proximity penalty** — for each LIDAR sensor, if the minimum ray reading drops below the danger threshold:

$$p_i = c \cdot \left(1 - \frac{d_{\min}}{\text{threshold}}\right)^2$$

This provides a smooth, quadratic warning signal that grows stronger as the link approaches contact.

**Near-goal boost** — when the end-effector enters the boost radius ($d < R$), the progress signal is scaled up by $b(d)$, reaching a maximum factor of $1 + k = 4$ at contact. This ensures the agent prioritizes reaching the goal even when obstacles are nearby.

---

## PPO Algorithm

We use **Proximal Policy Optimization** (PPO), an on-policy actor-only algorithm that collects batches of experience and performs multiple epochs of clipped gradient updates. Compared to vanilla REINFORCE (used in the first iteration of this project), PPO provides more stable learning through its clipped objective and better sample efficiency through mini-batch reuse.

### Why PPO over REINFORCE?

REINFORCE updates the policy once per episode using the full trajectory. This leads to high variance and poor sample efficiency. PPO addresses this by:

1. **Batching trajectories** — accumulating multiple episodes into a buffer before updating
2. **Reusing samples** — performing multiple optimization epochs on the same batch
3. **Clipping the objective** — preventing destructively large policy updates
4. **Early stopping via KL divergence** — halting updates when the policy changes too much

### Trajectory Buffer

Instead of updating after each episode, PPO accumulates a buffer of at least 2048 steps. Episodes are never truncated mid-way — each episode runs to completion, so the actual buffer size slightly exceeds 2048 steps. The buffer stores:

- States $s_t$
- Actions $a_t$
- Log-probabilities $\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)$
- Discounted returns $G_t$

### Clipped Surrogate Objective

For the same batch of transitions, actions are re-evaluated under the **current** policy to compute the clipped surrogate loss. Let $\hat{A}_t$ denote the advantage estimate (batch-normalized returns) and $\epsilon = 0.15$ the clip coefficient:

$$\mathcal{L}^{\text{CLIP}} = -\mathbb{E}_{t\sim \mathrm{Uniform}[0, T_b - 1]} \left[ \min\left(\frac{\pi_\theta(A_t \mid S_t)}{\pi_{\theta_{\text{old}}}(A_t \mid S_t)} \hat{A}_t,\ \text{clip}\left(\frac{\pi_\theta(A_t \mid S_t)}{\pi_{\theta_{\text{old}}}(A_t \mid S_t)}, 1 - \epsilon, 1 + \epsilon\right) \hat{A}_t\right) \right]$$

The clipping prevents the probability ratio from deviating too far from $1$, limiting destructively large policy updates.

An **entropy bonus** is subtracted from the loss to encourage exploration:

$$\mathcal{L} = \mathcal{L}^{\text{CLIP}} - c_{\text{ent}} \cdot H[\pi_\theta]$$

### KL-Based Early Stopping

Before each epoch, the approximate KL divergence between old and new policies is computed:

$$D_{\text{KL}}^{\text{approx}} = \mathbb{E}_{t\sim \mathrm{Uniform}[0, T_b - 1]}  \left[ \left(\frac{\pi_\theta(A_t \mid S_t)}{\pi_{\theta_{\text{old}}}(A_t \mid S_t)} - 1\right) - \log \frac{\pi_\theta(A_t \mid S_t)}{\pi_{\theta_{\text{old}}}(A_t \mid S_t)} \right]$$

If this exceeds the target threshold ($0.015$), remaining epochs are skipped. This acts as a safety valve against policy collapse.

### Gaussian Policy

The policy network outputs parameters of a diagonal Gaussian distribution:

$$\mu = \tanh(\text{head}_\mu(z)) \cdot \Delta\theta_{\max}$$

$$\sigma = \exp\bigl(\text{clamp}(\text{head}_\sigma(z), \log\sigma_{\min}, \log\sigma_{\max})\bigr) \cdot \Delta\theta_{\max}$$

where $z$ is the output of a shared MLP trunk $(256 \to 128)$ with ReLU activations. The $\tanh$ on the mean ensures it stays within the physical action bounds. Standard deviation is clamped in log-space to $[-2.0,\; 0.5]$.

### Update Rule

1. Accumulate transitions into the batch buffer until $\ge 2048$ steps are collected.
2. Compute discounted returns $G_t$ for every step within each episode.
3. Normalize advantages over the entire batch: $\hat{A}_t = (G_t - \bar{G}) / \sigma_G$.
4. For up to 10 epochs:
   - a. Check approximate KL divergence; stop if $> 0.015$.
   - b. Shuffle the batch and split into mini-batches of 256.
   - c. For each mini-batch, compute the clipped loss + entropy bonus.
   - d. Backpropagate and update $\theta$ with Adam (gradient norm clipped to 1.0).
5. Step the cosine-annealing learning rate scheduler.
6. Clear the batch buffer.

---

## Development History

### Iteration 1 — LIDAR Implementation

To give the agent spatial awareness of obstacles, we implemented a **LIDAR-like perception system**: 8 rays per joint, uniformly distributed over $[0, 2\pi)$, each returning a normalized distance to the nearest obstacle intersection. This expanded the observation from 10 to 34 dimensions.

<p align="center">
  <img src="assets/lidars.gif" alt="LIDAR perception" width="400">
</p>

### Iteration 2 — Training with Two Obstacles

We began training with two circular obstacles in the workspace. The agent learned to reach targets while avoiding the obstacles, validating that the LIDAR signal and obstacle-proximity penalty provide a sufficient learning signal.

<p align="center">
  <img src="assets/2obs_train.gif" alt="Training with 2 obstacles" width="400">
</p>

### Iteration 3 — Training with Four Obstacles

We scaled up to four obstacles with jittered positions ($\pm 40$ px around equilibrium each episode). The increased clutter forced the policy to generalize rather than memorize specific layouts.

<p align="center">
  <img src="assets/early_train.gif" alt="Training with 4 obstacles" width="600">
</p>

### Iteration 4 — The "Stalled Robot" Problem

After adding obstacle penalties, an unintended strategy emerged: **the robot preferred not to move at all** when the target was near an obstacle. Avoiding penalties outweighed the goal reward, so the agent learned that staying still was the safest option.

<p align="center">
  <img src="assets/stalled_robot.gif" alt="Stalled robot" width="400">
</p>

### Iteration 5 — Reward Refactoring

We addressed the stalled-robot problem with several reward changes:

1. **Near-goal progress boost** — the progress signal is amplified when the end-effector is close to the target ($< 80$ px), forcing the agent to prioritize the goal even in cluttered areas
2. **Stagnation detection** — if the moving average of distance changes drops to near zero over a window of 15 steps, the episode terminates with a penalty
3. **Increased goal reward** ($+50$) to dominate the obstacle penalty in the cost landscape

These changes rebalanced the reward landscape so that reaching the goal always dominates obstacle avoidance.

### Final Result

The trained policy was evaluated across three progressively harder scenarios:

**Static obstacles, determined start position:**

<p align="center">
  <img src="assets/test_static_obs_determined_start.gif" alt="Test — static obstacles" width="400">
</p>

**Random obstacles, determined start position:**

<p align="center">
  <img src="assets/test_random_obs_determined_start.gif" alt="Test — random obstacles" width="400">
</p>

**Random obstacles, random start position:**

<p align="center">
  <img src="assets/test_random_obs_random_start.gif" alt="Test — random obstacles, random start" width="400">
</p>

**Test metrics** (static/determined | random/determined | random/random):

<p align="center">
  <img src="assets/test_metrics_static_obs_determined_start.gif" alt="Metrics — static obstacles, determined start" width="33%">
  <img src="assets/test_metrics_random_obs_determined_start.gif" alt="Metrics — random obstacles, determined start" width="33%">
  <img src="assets/test_metrics_random_obs_random_start.gif" alt="Metrics — random obstacles, random start" width="33%">
</p>

### PPO vs REINFORCE Comparison

We compared the trained PPO policy against a vanilla REINFORCE agent in the same obstacle environment. PPO showed:

- More stable training curves
- Lower sensitivity to reward scale
- Better behavior under complex reward shaping
- Higher resistance to policy collapse

<p align="center">
  <img src="assets/test_reinforce.gif" alt="REINFORCE test" width="400">
</p>

<p align="center">
  <img src="assets/test_metrics_reinforce.gif" alt="REINFORCE metrics" width="600">
</p>

---

## Project Structure

```
├── main.py                  # Entry point: argparse, config assembly, GUI launch
├── requirements.txt
├── README.md
├── LICENSE
├── assets/                  # Demo GIFs and source videos
├── policy/
│   └── best_policy.pt       # Saved trained policy weights
└── ppo/
    ├── __init__.py
    ├── config.py            # Dataclass configs (Robot, Lidar, Obstacle, Model, Reward, Env, GUI)
    ├── state.py             # State dataclass (34-dim observation vector)
    ├── robot.py             # 3-DOF planar arm: forward kinematics, step, obs
    ├── lidar.py             # LIDAR sensor: ray casting against obstacles
    ├── obstacle.py          # Obstacle / ObstacleManager: placement, jitter, collision
    ├── env.py               # RL environment: reset, step, reward, termination
    ├── model_ppo.py         # GaussianMLPPolicy + PPO Model (train/test)
    ├── runner.py            # Unified train/test loop (headless + pygame)
    └── gui.py               # Pygame GUI with live matplotlib plots
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Hidden layers | 256 → 128 (ReLU) |
| Discount $\gamma$ | 0.97 |
| Learning rate | $3 \times 10^{-4}$ → $10^{-6}$ (cosine annealing) |
| Clip coefficient $\epsilon$ | 0.15 |
| PPO epochs per update | 10 |
| Mini-batch size | 256 |
| Batch buffer size | 2048 steps |
| Target KL | 0.015 |
| Entropy coefficient | 0.01 |
| Gradient clip norm | 1.0 |
| $\log\sigma$ range | $[-2.0,\; 0.5]$ |
| Max episode steps | 200 |
| Training episodes | 10000 |
| Test episodes | 500 |

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

Training runs for 5000 episodes (configurable via `--train-episodes N`). After training completes, the model is saved and test evaluation begins automatically. A Pygame window shows the robot simulation on the left and live training plots on the right.

For faster training without the simulation panel:

```bash
python main.py --train --no-sim
```

### Test

```bash
python main.py --test
```

Loads the saved policy and runs 500 test episodes (configurable via `--test-episodes N`) with the simulation visible.

### Additional Options

```
--model-path PATH      Path to .pt model file (default: policy/best_policy.pt)
--train-episodes N     Number of training episodes
--test-episodes N      Number of test episodes
--seed N               Random seed (default: 42)
```

### Output

- **Live GUI**: robot arm + obstacles + LIDAR rays visualization + real-time plots
- **Saved model**: `policy/best_policy.pt`
- **Metrics displayed**: total reward, success rate, collision rate, steps/episode, loss, KL divergence, entropy, per-joint sigma

---

## Collected Metrics

| Metric | Description |
|---|---|
| Total reward | Sum of rewards for the episode |
| Success rate | Sliding-window average of goal-reached episodes |
| Collision rate | Sliding-window average of obstacle collision episodes |
| Steps / episode | Number of steps taken before termination |
| Final distance | End-effector to target distance at episode end |
| Policy loss | PPO clipped surrogate loss |
| Gradient norm | Norm of the policy gradient (post-clip) |
| KL divergence | Approximate KL between old and updated policy |
| Entropy | Policy entropy (higher = more exploration) |
| Sigma (per joint) | Mean standard deviation of the Gaussian policy per joint |
