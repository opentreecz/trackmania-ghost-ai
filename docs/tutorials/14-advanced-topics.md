# Advanced Topics: Custom Networks and Custom Rewards

How to go beyond the defaults -- designing new neural network architectures,
implementing fully custom reward functions, switching to image-based
observations, and using alternative RL algorithms.

## Table of Contents

1. [Custom Neural Network Architectures](#1-custom-neural-network-architectures)
2. [Writing a Custom Actor from Scratch](#2-writing-a-custom-actor-from-scratch)
3. [Adding Recurrence (LSTM/GRU)](#3-adding-recurrence-lstmgru)
4. [Switching to CNN (Image Observations)](#4-switching-to-cnn-image-observations)
5. [Custom Reward Functions](#5-custom-reward-functions)
6. [Recording a Custom Reward Trajectory](#6-recording-a-custom-reward-trajectory)
7. [tmrl Reward Internals](#7-tmrl-reward-internals)
8. [Alternative Algorithms (REDQ)](#8-alternative-algorithms-redq)
9. [Experiment Tracking](#9-experiment-tracking)

---

## 1. Custom Neural Network Architectures

The default architecture is a 2-layer MLP with 256 neurons per layer.
All architecture code lives in `src/trackmania_ghost_ai/models.py`.

### Deeper networks

Edit `configs/training.yaml`:

```yaml
algorithm:
  actor:
    hidden_sizes: [512, 256, 128]   # 3 layers, decreasing width
  critic:
    hidden_sizes: [512, 256, 128]
```

No code changes needed -- `_build_mlp()` in `models.py:40` dynamically
constructs any depth.

### Wider networks

```yaml
algorithm:
  actor:
    hidden_sizes: [1024, 1024]      # 2 layers, 1024 each
```

More parameters = more capacity, but also more data needed to train.

### Asymmetric actor-critic

The critic can be larger than the actor. The critic estimates Q-values
(a harder task), while the actor just needs to pick good actions:

```yaml
algorithm:
  actor:
    hidden_sizes: [256, 256]
    activation: "ReLU"
  critic:
    hidden_sizes: [512, 512, 256]   # bigger critic
    activation: "GELU"              # different activation
```

This is valid because `MLPActorCritic` in `models.py:184` constructs the
actor and each Q-function independently.

---

## 2. Writing a Custom Actor from Scratch

To create a fundamentally different policy network, subclass
`TorchActorModule` from tmrl.

### Example: Actor with skip connections

Create a new file or add to `models.py`:

```python
# src/trackmania_ghost_ai/models.py (add to existing file)

class ResidualBlock(nn.Module):
    """MLP block with a skip connection."""

    def __init__(self, dim: int, activation: type[nn.Module] = nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            activation(),
            nn.Linear(dim, dim),
        )
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ResidualMLPActor(TorchActorModule):
    """MLP actor with residual (skip) connections."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__(observation_space, action_space)

        self.dim_obs = _obs_dim(observation_space)
        self.dim_act = action_space.shape[0]
        self.act_limit = float(action_space.high[0])

        # Project observation to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.dim_obs, hidden_dim),
            activation(),
        )

        # Stack of residual blocks
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, activation) for _ in range(num_blocks)]
        )

        # Output heads
        self.mu_layer = nn.Linear(hidden_dim, self.dim_act)
        self.log_std_layer = nn.Linear(hidden_dim, self.dim_act)

    def forward(self, obs, test=False, with_logprob=True):
        x = _flatten_obs(obs)
        x = self.input_proj(x)
        x = self.blocks(x)

        mu = self.mu_layer(x)
        log_std = torch.clamp(
            self.log_std_layer(x), LOG_STD_MIN, LOG_STD_MAX
        )
        std = torch.exp(log_std)

        pi_dist = Normal(mu, std)
        pi_action = mu if test else pi_dist.rsample()

        logp = None
        if with_logprob:
            logp = pi_dist.log_prob(pi_action).sum(dim=-1)
            logp -= (
                2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))
            ).sum(dim=-1)

        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, logp

    def act(self, obs, test=False):
        with torch.no_grad():
            action, _ = self.forward(obs, test=test, with_logprob=False)
            return action.squeeze().cpu().numpy()
```

### Wire it into the ActorCritic

```python
class ResidualMLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__()
        self.actor = ResidualMLPActor(
            observation_space, action_space, **kwargs
        )
        self.q1 = MLPQFunction(observation_space, action_space)
        self.q2 = MLPQFunction(observation_space, action_space)
```

### Register in agent.py

Edit `src/trackmania_ghost_ai/agent.py` to use the new class:

```python
from trackmania_ghost_ai.models import ResidualMLPActorCritic

# In make_training_agent_cls():
model_cls = _partial(
    ResidualMLPActorCritic,
    hidden_dim=256,
    num_blocks=3,
)
```

---

## 3. Adding Recurrence (LSTM/GRU)

A recurrent actor can remember information across time steps, which may
help with long-term planning (e.g., remembering a curve from several steps
ago that LIDAR no longer sees).

### Example: GRU-based actor

```python
class GRUMLPActor(TorchActorModule):
    """Actor with a GRU layer for temporal memory."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        gru_layers: int = 1,
    ):
        super().__init__(observation_space, action_space)

        self.dim_obs = _obs_dim(observation_space)
        self.dim_act = action_space.shape[0]
        self.act_limit = float(action_space.high[0])
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=self.dim_obs,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.mu_layer = nn.Linear(hidden_dim, self.dim_act)
        self.log_std_layer = nn.Linear(hidden_dim, self.dim_act)

        self._hidden = None  # GRU hidden state

    def reset_hidden(self):
        """Call at the start of each episode."""
        self._hidden = None

    def forward(self, obs, test=False, with_logprob=True):
        x = _flatten_obs(obs).unsqueeze(1)  # add time dim: (B, 1, obs_dim)

        gru_out, self._hidden = self.gru(x, self._hidden)
        h = gru_out.squeeze(1)  # (B, hidden_dim)

        mu = self.mu_layer(h)
        log_std = torch.clamp(
            self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX
        )
        std = torch.exp(log_std)

        pi_dist = Normal(mu, std)
        pi_action = mu if test else pi_dist.rsample()

        logp = None
        if with_logprob:
            logp = pi_dist.log_prob(pi_action).sum(dim=-1)
            logp -= (
                2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))
            ).sum(dim=-1)

        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, logp

    def act(self, obs, test=False):
        with torch.no_grad():
            action, _ = self.forward(obs, test=test, with_logprob=False)
            return action.squeeze().cpu().numpy()
```

**Caveat**: Recurrent policies complicate the replay buffer. Standard SAC
samples random transitions -- this breaks the hidden state sequence. You
would need to sample trajectory segments instead of individual transitions.
This is a significant implementation effort and is not directly supported
by tmrl's `GenericTorchMemory`.

---

## 4. Switching to CNN (Image Observations)

Instead of LIDAR (19 beams), you can use raw screenshots. This requires
switching the observation interface and the network architecture.

### Change the interface

```yaml
environment:
  interface: "TM20FULL"          # full screenshot observations
  window_width: 256              # smaller for faster capture
  window_height: 128
  img_hist_len: 4                # 4 stacked grayscale frames
```

### Use tmrl's built-in CNN

tmrl provides `VanillaCNNActorCritic`:

```python
# src/trackmania_ghost_ai/agent.py
from tmrl.custom.custom_models import VanillaCNNActorCritic

model_cls = _partial(VanillaCNNActorCritic)
```

### Write your own CNN actor

```python
class CNNActor(TorchActorModule):
    """CNN policy for image observations."""

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # Assume obs is a (4, 64, 64) grayscale image stack
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 64, 64)
            conv_out = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(256, action_space.shape[0])
        self.log_std_layer = nn.Linear(256, action_space.shape[0])
        self.act_limit = float(action_space.high[0])

    # ... forward() and act() same pattern as MLP actor
```

### Trade-offs: LIDAR vs Images

| Aspect          | LIDAR (MLP)            | Images (CNN)                |
|-----------------|------------------------|-----------------------------|
| Training time   | 5 hours for turns      | 2+ days for turns           |
| Observation size| 83 floats              | 4 x 64 x 64 = 16,384 pixels|
| Track constraint| Black-bordered only    | Any track, any camera       |
| Network size    | ~100K parameters       | ~1M+ parameters             |
| GPU requirement | Low                    | High                        |

LIDAR is strongly recommended for initial experiments. Switch to images
only if you need to work on non-LIDAR-compatible tracks.

---

## 5. Custom Reward Functions

Beyond editing weights in YAML (covered in
[Tutorial 01](./01-defining-driving-capabilities.md)), you can write
entirely new reward logic.

### The shaped_reward hook

The `shaped_reward()` function in `src/trackmania_ghost_ai/environment.py:83`
is called after each environment step. It receives:

- `raw_reward` -- the base reward from tmrl (track progress)
- `info` -- a dict with game state information
- `cfg` -- your project configuration

### Example: Lap time reward

```python
def shaped_reward(raw_reward, info, cfg):
    r = raw_reward

    # Base progress reward
    r += cfg.reward.progress_weight * info.get("progress", 0.0)

    # If the car finished the track, reward inversely proportional to time
    if info.get("finished", False):
        lap_time = info.get("lap_time", 60.0)  # seconds
        # Faster lap = higher reward. 30s lap gets 100, 60s gets 50
        r += 3000.0 / max(lap_time, 10.0)

    # Per-step time penalty to encourage speed
    r -= 0.02

    return r
```

### Example: Smoothness reward

```python
def shaped_reward(raw_reward, info, cfg):
    r = raw_reward + cfg.reward.progress_weight * info.get("progress", 0.0)

    # Penalise steering oscillation
    steer_history = info.get("steer_history", [])
    if len(steer_history) >= 3:
        # Second derivative of steering = oscillation
        d2 = abs(steer_history[-1] - 2 * steer_history[-2] + steer_history[-3])
        r -= 2.0 * d2

    # Reward maintaining speed (penalise deceleration)
    speed = info.get("speed", 0.0)
    prev_speed = info.get("prev_speed", 0.0)
    if speed < prev_speed - 10:  # sudden deceleration
        r -= 0.5

    return r
```

### Example: Zone-based reward

Different reward weights for different track sections:

```python
def shaped_reward(raw_reward, info, cfg):
    r = raw_reward
    progress = info.get("progress_pct", 0.0)  # 0.0 to 1.0

    if progress < 0.3:
        # First third: focus on speed
        r += 2.0 * info.get("speed", 0.0) / 1000.0
    elif progress < 0.7:
        # Middle: technical section, focus on survival
        r += 8.0 * info.get("progress", 0.0)
        if info.get("collision", False):
            r -= 10.0
    else:
        # Final third: sprint to finish
        r += 3.0 * info.get("speed", 0.0) / 1000.0
        if info.get("finished", False):
            r += 200.0

    return r
```

---

## 6. Recording a Custom Reward Trajectory

tmrl's default reward is based on track progress measured against a
pre-recorded trajectory:

```bash
python -m tmrl --record-reward
```

This produces `~/TmrlData/reward/reward.pkl` -- an ordered list of 3D
positions along the track.

### Tips for the trajectory

- **Moderate speed**: Drive at 60-70% of your maximum, not flat out
- **Centre the road**: Stay in the middle -- the reward measures distance
  to the nearest trajectory point
- **Full lap**: Start at the beginning, drive to the finish
- **Re-record if needed**: Simply run the command again to overwrite

### Advanced: Editing the trajectory

```python
import pickle
import numpy as np

# Load
with open("~/TmrlData/reward/reward.pkl", "rb") as f:
    trajectory = pickle.load(f)

# trajectory is a list of [x, y, z] positions
print(f"Trajectory has {len(trajectory)} points")

# You can resample, smooth, or edit it
from scipy.ndimage import gaussian_filter1d
trajectory = np.array(trajectory)
smoothed = gaussian_filter1d(trajectory, sigma=3, axis=0)

# Save back
with open("~/TmrlData/reward/reward.pkl", "wb") as f:
    pickle.dump(smoothed.tolist(), f)
```

---

## 7. tmrl Reward Internals

Understanding how tmrl computes reward under the hood.

### The RewardFunction class

In tmrl's `compute_reward.py`, the reward function:

1. Takes the car's current 3D position
2. Finds the closest point on the recorded trajectory
3. Computes: `reward = (new_index - old_index) / 100.0`
4. This measures how many trajectory points the car passed since the last step

### Key config parameters

In `~/TmrlData/config/config.json`:

```json
"REWARD_CONFIG": {
    "END_OF_TRACK": 100.0,        // bonus for crossing finish line
    "CONSTANT_PENALTY": 0.0,      // per-step constant (negative = time penalty)
    "CHECK_FORWARD": 500,         // max search distance forward on trajectory
    "CHECK_BACKWARD": 10,         // search distance backward (for reversing)
    "FAILURE_COUNTDOWN": 10,      // zero-reward steps before episode ends
    "MIN_STEPS": 70,              // grace period before failure can trigger
    "MAX_STRAY": 100.0            // max metres from trajectory before ignoring
}
```

### Debugging the reward

```bash
# Check environment (shows reward values live)
python -m tmrl --check-environment
```

Drive manually and watch the reward values. They should:
- Be positive when driving forward
- Be zero when stopped
- Trigger episode end when driving backward for too long

---

## 8. Alternative Algorithms (REDQ)

tmrl includes REDQ (Randomised Ensembled Double Q-learning), which uses
N parallel Q-networks instead of 2. This can reduce overestimation bias
and improve sample efficiency.

### Enable REDQ

In `~/TmrlData/config/config.json`:

```json
{
    "ALG": {
        "ALGORITHM": "REDQSAC",
        "REDQ_N": 10,           // number of Q-networks
        "REDQ_M": 2,            // subset size for target computation
        "REDQ_Q_UPDATES_PER_POLICY": 20
    }
}
```

Or in Python:

```python
from tmrl.custom.custom_algorithms import REDQSACAgent
from tmrl.custom.custom_models import REDQMLPActorCritic

model_cls = _partial(REDQMLPActorCritic, q_num=10)
training_agent_cls = _partial(REDQSACAgent, model_cls=model_cls)
```

### When to use REDQ

| Scenario                         | Use SAC          | Use REDQ          |
|----------------------------------|------------------|-------------------|
| First experiments                | Yes (simpler)    | No                |
| Limited training data            | Depends          | Yes (more efficient)|
| Overestimation is a problem      | No               | Yes               |
| GPU memory is tight              | Yes (2 critics)  | No (10 critics)   |
| Maximum performance              | Try both         | Try both          |

### Trade-offs

| Aspect            | SAC (default)      | REDQ                        |
|-------------------|--------------------|------------------------------|
| Q-networks        | 2                  | 10 (configurable)            |
| GPU memory        | Low                | ~5x more for Q-networks     |
| Sample efficiency | Good               | Better (fewer samples needed)|
| Training speed    | Fast per step      | Slower per step              |
| Stability         | Good               | Often more stable            |

---

## 9. Experiment Tracking

### Weights & Biases

```yaml
run:
  wandb:
    enabled: true
    project: "trackmania-ghost-ai"
    entity: "your-team"
```

```bash
pip install wandb
wandb login
tm-train --config configs/training.yaml
```

Tracked metrics:
- `return_train` / `return_test` -- episode rewards
- `episode_length_train` / `episode_length_test`
- `loss_actor` / `loss_critic`
- `memory_len` -- replay buffer size
- `sampling_duration` / `training_step_duration`

### Manual logging

Add custom metrics by modifying the training loop. tmrl's training agent
returns a dict from `train()` -- add keys and they flow to W&B:

```python
# In a custom TrainingAgent.train():
return {
    "loss_actor": loss_a.item(),
    "loss_critic": loss_c.item(),
    "custom_metric": my_value,      # appears in W&B automatically
}
```

### Experiment naming convention

Use descriptive run names to keep track of experiments:

```yaml
run:
  name: "lidar-sac-256x2-g995-a037-v1"
  # format: interface-algo-arch-gamma-alpha-version
```

This makes it easy to compare runs in W&B or in the weights directory.
