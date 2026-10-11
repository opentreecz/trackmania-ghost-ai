# Writing New Driving Parameters

This tutorial covers how to modify the agent's parameters at every level:
the YAML config, the neural network architecture, the observation and action
spaces, and the SAC algorithm hyperparameters. This is the "knobs and dials"
guide for the project.

## Table of Contents

1. [Parameter Hierarchy](#1-parameter-hierarchy)
2. [YAML Configuration Reference](#2-yaml-configuration-reference)
3. [Network Architecture Parameters](#3-network-architecture-parameters)
4. [Observation Space Parameters](#4-observation-space-parameters)
5. [Action Space Parameters](#5-action-space-parameters)
6. [SAC Algorithm Parameters](#6-sac-algorithm-parameters)
7. [Training Loop Parameters](#7-training-loop-parameters)
8. [Replay Buffer Parameters](#8-replay-buffer-parameters)
9. [Creating a New Config Profile](#9-creating-a-new-config-profile)
10. [Parameter Tuning Strategies](#10-parameter-tuning-strategies)

---

## 1. Parameter Hierarchy

Parameters flow through three layers:

```
configs/training.yaml          <-- You edit this (no code changes)
        |
        v
src/.../config.py              <-- Dataclass definitions (add new params here)
        |
        v
src/.../agent.py, models.py    <-- Consumed by PyTorch models and SAC agent
```

For most changes, editing the YAML is sufficient. You only touch Python when
adding entirely new parameter types or changing the network architecture.

---

## 2. YAML Configuration Reference

The complete parameter tree with defaults and explanations:

```yaml
# ---- Run identity ----
run:
  name: "ghost-ai-default"     # Name for weights/checkpoints files
  seed: 42                      # Random seed for reproducibility
  device: null                  # null = auto (CUDA if available), "cpu", "cuda"
  wandb:
    enabled: false              # Enable Weights & Biases logging
    project: "trackmania-ghost-ai"
    entity: null                # W&B team name (null = personal)

# ---- Environment ----
environment:
  interface: "TM20LIDAR"        # "TM20LIDAR" | "TM20LIDARPROGRESS" | "TM20FULL"
  window_width: 958             # TM2020 window width in pixels
  window_height: 488            # TM2020 window height in pixels
  img_hist_len: 4               # Number of stacked observation frames
  rtgym:
    time_step_duration: 0.05    # Seconds per step (0.05 = 20 Hz)
    start_obs_capture: 0.04     # When to capture observation within the step
    act_buf_len: 2              # Previous actions included in observation
    ep_max_length: 2000         # Max steps per episode (2000 * 0.05 = 100s)
    benchmark: false            # Timing benchmark mode

# ---- Observation ----
observation:
  lidar_beams: 19               # Number of LIDAR rays (fixed by tmrl)
  speed_dim: 1                  # Speed dimensions (fixed: 1 scalar)

# ---- Action ----
action:
  dim: 3                        # [gas, brake, steer] (fixed by tmrl)
  low: -1.0                     # Action minimum
  high: 1.0                     # Action maximum

# ---- Reward ----
reward:
  velocity_weight: 1.0
  progress_weight: 5.0
  checkpoint_bonus: 10.0
  finish_bonus: 100.0
  collision_penalty: -5.0
  off_track_penalty: -2.0

# ---- Algorithm ----
algorithm:
  name: "SAC"
  gamma: 0.995                  # Discount factor
  polyak: 0.995                 # Target network EMA coefficient
  alpha: 0.37                   # Entropy coefficient
  learn_entropy_coef: true      # Auto-tune alpha
  target_entropy: -7.0          # Target entropy (null = auto: -dim(action))
  actor:
    lr: 0.0003                  # Actor learning rate
    hidden_sizes: [256, 256]    # Actor hidden layer widths
    activation: "ReLU"          # Activation function
  critic:
    lr: 0.00005                 # Critic learning rate
    hidden_sizes: [256, 256]    # Critic hidden layer widths
    activation: "ReLU"
  entropy:
    lr: 0.0003                  # Entropy coefficient learning rate

# ---- Memory ----
memory:
  capacity: 1000000             # Replay buffer size (transitions)
  batch_size: 256               # Training batch size
  min_samples: 1000             # Minimum samples before training starts

# ---- Training loop ----
training:
  epochs: 200                   # Total training epochs
  rounds: 50                    # Rounds per epoch
  steps: 100                    # Gradient steps per round
  start_training: 1000          # Min samples before first gradient step
  update_model_interval: 100    # Steps between broadcasting weights to worker
  update_buffer_interval: 100   # Steps between fetching new samples

# ---- Networking ----
networking:
  server_ip: "127.0.0.1"
  server_port: 6666
  password: "trackmania-ghost-ai"
  security: null
  local_mode: true

# ---- Checkpointing ----
checkpoint:
  dir: "checkpoints"
  save_interval: 10             # Save every N epochs
  keep_last: 5                  # Keep only the last N checkpoints
```

---

## 3. Network Architecture Parameters

The MLP architecture is defined in `src/trackmania_ghost_ai/models.py`.

### Changing hidden layer sizes (YAML only)

```yaml
algorithm:
  actor:
    hidden_sizes: [256, 256]    # default: two layers of 256 neurons
  critic:
    hidden_sizes: [256, 256]
```

**Common configurations:**

| Config              | `hidden_sizes`        | Parameters | Use case                    |
|---------------------|-----------------------|------------|-----------------------------|
| Small               | `[128, 128]`          | ~30K       | Fast training, simple tracks|
| Default             | `[256, 256]`          | ~100K      | Most tracks                 |
| Large               | `[512, 256, 128]`     | ~200K      | Complex tracks              |
| Deep                | `[256, 256, 256, 256]`| ~260K      | Experimental                |

**Rule of thumb**: Start with the default `[256, 256]`. Only increase if
the agent plateaus on a complex track. Larger networks need more training data.

### Changing activation function (YAML only)

```yaml
algorithm:
  actor:
    activation: "ReLU"     # default
```

Available activations (defined in `src/trackmania_ghost_ai/agent.py:22`):

| Name         | Characteristics                                    |
|--------------|----------------------------------------------------|
| `ReLU`       | Default, fast, works well for most cases           |
| `Tanh`       | Bounded output, smoother gradients                 |
| `GELU`       | Smoother than ReLU, popular in modern architectures|
| `ELU`        | Handles negative inputs better than ReLU           |
| `LeakyReLU`  | Avoids dead neurons                                |
| `SiLU`       | Self-gated, good for deep networks                 |

### Adding a new activation (code change)

Edit `src/trackmania_ghost_ai/agent.py:22`:

```python
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "GELU": nn.GELU,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "SiLU": nn.SiLU,
    "Mish": nn.Mish,        # <-- add new ones here
}
```

### Separate actor/critic architectures

The actor and critic can use different sizes and activations:

```yaml
algorithm:
  actor:
    lr: 0.0003
    hidden_sizes: [256, 256]
    activation: "ReLU"
  critic:
    lr: 0.00005
    hidden_sizes: [512, 256]   # wider critic
    activation: "GELU"         # different activation
```

This is valid because `MLPActorCritic` in `models.py:184` constructs them
independently. The actor maps observations to actions; the critic maps
(observation, action) pairs to Q-values. They can have different capacities.

---

## 4. Observation Space Parameters

### What the agent sees

The LIDAR observation is a fixed structure from tmrl:

```
obs = (speed[1], lidar[img_hist_len, 19], prev_act_1[3], prev_act_2[3])
```

**Total flat dimension** = `speed_dim + img_hist_len * lidar_beams + act_buf_len * 3`

Default: `1 + 4 * 19 + 2 * 3 = 83`

### Parameters you can change

| Parameter                          | YAML key                           | Effect on obs dim |
|------------------------------------|------------------------------------|--------------------|
| Number of stacked LIDAR frames     | `environment.img_hist_len`         | +/- 19 per frame   |
| Number of previous actions         | `environment.rtgym.act_buf_len`    | +/- 3 per action   |

**Changing `img_hist_len`:**

```yaml
environment:
  img_hist_len: 8   # 8 frames instead of 4
  # New obs dim: 1 + 8*19 + 2*3 = 159
```

**Changing `act_buf_len`:**

```yaml
environment:
  rtgym:
    act_buf_len: 3   # 3 previous actions instead of 2
    # New obs dim: 1 + 4*19 + 3*3 = 86
```

The model automatically adapts -- `_obs_dim()` in `models.py:53` computes the
input dimension from the observation space at runtime. No code changes needed.

### Switching to image-based observations

To use screenshots instead of LIDAR (CNN instead of MLP), change the
interface:

```yaml
environment:
  interface: "TM20FULL"       # full screenshot observations
  window_width: 256           # smaller window for faster capture
  window_height: 128
```

This requires a CNN-based actor/critic instead of MLP. You would need to
replace `MLPActorCritic` with tmrl's `VanillaCNNActorCritic` in `agent.py`.

### Adding the `preprocess_obs` hook

The identity function in `src/trackmania_ghost_ai/environment.py:27` is a hook
for custom observation processing:

```python
def preprocess_obs(obs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    speed, lidar, act1, act2 = obs

    # Example: normalise speed to [0, 1]
    speed = speed / 500.0  # assume max ~500 km/h

    # Example: clip LIDAR to max range
    lidar = np.clip(lidar, 0.0, 1.0)

    return (speed, lidar, act1, act2)
```

---

## 5. Action Space Parameters

The action space is fixed by the tmrl interface: 3 continuous values in
`[-1, 1]`. You cannot add or remove action dimensions without modifying tmrl
itself.

However, you can influence *how* the agent uses these actions through SAC
parameters:

### Action smoothness

```yaml
algorithm:
  alpha: 0.2                  # lower = more deterministic actions
  target_entropy: -9.0        # lower = less action randomness
```

### Action scale clipping

The `SquashedGaussianMLPActor` in `models.py:76` uses `tanh` squashing, which
naturally constrains outputs to `[-act_limit, +act_limit]`. The `act_limit`
is read from the action space (default 1.0).

---

## 6. SAC Algorithm Parameters

These are the core learning parameters in the `algorithm:` section.

### Discount factor (`gamma`)

```yaml
algorithm:
  gamma: 0.995   # default
```

| Value  | Effect                                                    |
|--------|-----------------------------------------------------------|
| 0.99   | Short-sighted: prioritises immediate reward               |
| 0.995  | Default: good balance of short and long-term planning     |
| 0.999  | Far-sighted: plans many steps ahead, slower to converge   |

**For track racing**: Higher gamma (0.995-0.999) is better because the agent
needs to plan corners many steps in advance.

### Target network coefficient (`polyak`)

```yaml
algorithm:
  polyak: 0.995   # default
```

Controls how slowly the target Q-network updates. Higher = more stable but
slower adaptation.

| Value  | Effect                              |
|--------|-------------------------------------|
| 0.99   | Faster adaptation, less stable      |
| 0.995  | Default balance                     |
| 0.999  | Very stable, very slow adaptation   |

### Entropy coefficient (`alpha`)

```yaml
algorithm:
  alpha: 0.37              # initial value
  learn_entropy_coef: true # auto-tune during training
  target_entropy: -7.0     # target (null = auto: -3 for 3D action)
```

| `alpha` | Behaviour                                              |
|---------|--------------------------------------------------------|
| 0.01    | Nearly deterministic -- exploits known strategies      |
| 0.1     | Mild exploration                                       |
| 0.37    | Default -- moderate exploration                        |
| 0.8     | Heavy exploration -- tries many different strategies   |
| 2.0     | Very random -- useful only at the start of training    |

With `learn_entropy_coef: true`, SAC auto-tunes alpha toward `target_entropy`.

### Learning rates

```yaml
algorithm:
  actor:
    lr: 0.0003    # policy network learning rate
  critic:
    lr: 0.00005   # Q-network learning rate
  entropy:
    lr: 0.0003    # entropy coefficient learning rate
```

**Important**: The critic LR is intentionally lower (5e-5 vs 3e-4). The
critic needs to be more stable than the actor -- if the critic changes too
fast, the actor chases a moving target.

| Situation                    | Adjustment                            |
|------------------------------|---------------------------------------|
| Training is unstable         | Lower all LRs by 3-5x                |
| Training is too slow         | Increase actor LR to 1e-3             |
| Agent oscillates             | Lower actor LR, keep critic stable    |
| Fine-tuning a trained model  | Lower all LRs by 10x                 |

---

## 7. Training Loop Parameters

```yaml
training:
  epochs: 200                   # number of outer loops
  rounds: 50                    # inner loops per epoch
  steps: 100                    # gradient updates per round
  start_training: 1000          # wait for this many samples
  update_model_interval: 100    # broadcast weights every N steps
  update_buffer_interval: 100   # fetch samples every N steps
```

### Total gradient steps

`epochs * rounds * steps = 200 * 50 * 100 = 1,000,000` gradient updates.

### How to adjust

| Goal                      | Change                                      |
|---------------------------|---------------------------------------------|
| Quick experiment          | `epochs: 20, rounds: 20, steps: 50`        |
| Full training run         | `epochs: 200, rounds: 50, steps: 100`      |
| Extended training         | `epochs: 500, rounds: 100, steps: 200`     |
| More frequent evaluation  | Lower `update_model_interval` to 50         |
| Larger batches of samples | Lower `update_buffer_interval` to 50        |

### `start_training`

The number of transitions collected before the first gradient step. This
ensures the replay buffer has enough diversity.

| Value  | Effect                                      |
|--------|---------------------------------------------|
| 500    | Start training quickly (may overfit early)  |
| 1000   | Default                                      |
| 5000   | More diverse initial data, slower start      |

---

## 8. Replay Buffer Parameters

```yaml
memory:
  capacity: 1000000   # 1M transitions
  batch_size: 256     # samples per gradient step
  min_samples: 1000   # minimum before training
```

### Capacity

| Value     | RAM usage (approx) | Use case                           |
|-----------|--------------------|------------------------------------|
| 100,000   | ~200 MB            | Quick experiments, limited RAM     |
| 1,000,000 | ~2 GB              | Default                            |
| 5,000,000 | ~10 GB             | Long training runs, diverse data   |

### Batch size

| Value | Effect                                      |
|-------|---------------------------------------------|
| 64    | Noisier gradients, faster per step          |
| 256   | Default -- good balance                     |
| 512   | Smoother gradients, more GPU memory needed  |
| 1024  | Very smooth, requires large replay buffer   |

---

## 9. Creating a New Config Profile

Best practice: create separate configs for different driving styles or tracks.

```bash
# Copy the default
cp configs/training.yaml configs/aggressive-cornering.yaml
```

Edit the copy:

```yaml
# configs/aggressive-cornering.yaml
run:
  name: "aggressive-cornering"    # unique name for weights

reward:
  velocity_weight: 2.5
  collision_penalty: -2.0         # tolerate some contact

algorithm:
  alpha: 0.5                      # more exploration
  target_entropy: -5.0
  actor:
    hidden_sizes: [512, 256]      # bigger network
    lr: 0.0005
```

Train:

```bash
tm-train --config configs/aggressive-cornering.yaml
```

Weights will be saved under the `aggressive-cornering` run name, separate
from the default profile.

---

## 10. Parameter Tuning Strategies

### Start coarse, refine later

1. **Phase 1**: Use default parameters. Train for 50 epochs. Does the agent
   learn to drive forward? If not, check the reward trajectory recording.

2. **Phase 2**: Adjust reward weights. Is it too cautious? Increase
   `velocity_weight`. Crashing too much? Increase `collision_penalty`.

3. **Phase 3**: Tune SAC parameters. Stuck in a local optimum? Increase
   `alpha`. Training unstable? Lower learning rates.

4. **Phase 4**: Try larger networks only if Phase 3 plateaus.

### Grid search example

Create multiple configs and train each:

```bash
for alpha in 0.1 0.37 0.8; do
  for lr in 0.0001 0.0003 0.001; do
    cat configs/training.yaml | \
      sed "s/alpha: 0.37/alpha: $alpha/" | \
      sed "s/lr: 0.0003/lr: $lr/" > \
      configs/sweep_a${alpha}_lr${lr}.yaml

    # Update run name
    sed -i "s/ghost-ai-default/sweep-a${alpha}-lr${lr}/" \
      configs/sweep_a${alpha}_lr${lr}.yaml

    tm-train --config configs/sweep_a${alpha}_lr${lr}.yaml &
  done
done
```

### Parameters ranked by impact

1. **Reward weights** -- largest impact on behaviour (what to optimise)
2. **`gamma`** -- how far ahead the agent plans
3. **`alpha` / `target_entropy`** -- exploration vs exploitation
4. **Learning rates** -- training stability and speed
5. **Network size** -- model capacity (rarely the bottleneck)
6. **Batch size / buffer size** -- training efficiency

---

## Quick Reference Card

| Parameter                         | Location              | Default        | Range           |
|-----------------------------------|-----------------------|----------------|-----------------|
| `reward.velocity_weight`          | `training.yaml`       | 1.0            | 0.0 - 10.0     |
| `reward.progress_weight`          | `training.yaml`       | 5.0            | 0.0 - 20.0     |
| `reward.collision_penalty`        | `training.yaml`       | -5.0           | -50.0 - 0.0    |
| `algorithm.gamma`                 | `training.yaml`       | 0.995          | 0.95 - 0.999   |
| `algorithm.alpha`                 | `training.yaml`       | 0.37           | 0.01 - 2.0     |
| `algorithm.target_entropy`        | `training.yaml`       | -7.0           | -10.0 - -1.0   |
| `algorithm.actor.lr`              | `training.yaml`       | 0.0003         | 1e-5 - 1e-2    |
| `algorithm.critic.lr`             | `training.yaml`       | 0.00005        | 1e-6 - 1e-3    |
| `algorithm.actor.hidden_sizes`    | `training.yaml`       | [256, 256]     | [64]-[1024,512] |
| `algorithm.actor.activation`      | `training.yaml`       | "ReLU"         | see table above |
| `environment.img_hist_len`        | `training.yaml`       | 4              | 1 - 16          |
| `environment.rtgym.act_buf_len`   | `training.yaml`       | 2              | 1 - 4           |
| `environment.rtgym.time_step_duration` | `training.yaml`  | 0.05           | 0.02 - 0.2     |
| `memory.capacity`                 | `training.yaml`       | 1,000,000      | 10K - 10M       |
| `memory.batch_size`               | `training.yaml`       | 256            | 32 - 1024       |
| `training.epochs`                 | `training.yaml`       | 200            | 10 - 1000       |
