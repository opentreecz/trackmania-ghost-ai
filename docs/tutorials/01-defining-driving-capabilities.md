# Defining New Driving Capabilities

This tutorial explains how to teach the AI agent new driving behaviours --
drifting, air-braking, optimal racing lines, cautious driving, or aggressive
cornering -- by shaping the reward function, tuning the observation window, and
adjusting the exploration/exploitation balance.

> In reinforcement learning the agent does not learn explicit rules.
> It learns whatever behaviour **maximises cumulative reward**.
> To change *how* it drives, you change *what* it is rewarded for.

## Table of Contents

1. [How Driving Behaviour Emerges](#1-how-driving-behaviour-emerges)
2. [Reward Shaping (the primary lever)](#2-reward-shaping-the-primary-lever)
3. [YAML-Only Recipes](#3-yaml-only-recipes)
4. [Code-Level Reward Customisation](#4-code-level-reward-customisation)
5. [Recording a Reward Trajectory](#5-recording-a-reward-trajectory)
6. [Observation Tuning](#6-observation-tuning)
7. [Exploration vs Exploitation](#7-exploration-vs-exploitation)
8. [Training Duration and Curriculum](#8-training-duration-and-curriculum)
9. [Verifying the New Capability](#9-verifying-the-new-capability)

---

## 1. How Driving Behaviour Emerges

The agent receives an observation every 50 ms (20 Hz) and outputs three
continuous values:

| Output   | Range      | Meaning               |
|----------|------------|-----------------------|
| `gas`    | `[-1, 1]`  | Throttle (1 = full)   |
| `brake`  | `[-1, 1]`  | Brake force           |
| `steer`  | `[-1, 1]`  | Left/right steering   |

The observation it sees:

| Component         | Shape     | What the agent "sees"                    |
|-------------------|-----------|------------------------------------------|
| Speed             | `(1,)`    | How fast it is going                     |
| LIDAR history     | `(4, 19)` | 4 frames of 19 distance beams (walls)   |
| Previous action 1 | `(3,)`    | What it did 2 steps ago                  |
| Previous action 2 | `(3,)`    | What it did 1 step ago                   |

The agent has no concept of "drifting" or "air-braking". It only knows:
*"If I output these 3 numbers given this observation, I get more reward."*

To make it drift, you reward the conditions that arise during drifting (high
speed through corners, lateral velocity). To make it brake precisely, you
penalise crashing and reward smooth progress through tight sections.

---

## 2. Reward Shaping (the primary lever)

All reward weights live in `configs/training.yaml` under the `reward:` section:

```yaml
# configs/training.yaml
reward:
  velocity_weight: 1.0       # reward for going fast
  progress_weight: 5.0       # reward for advancing along the track
  checkpoint_bonus: 10.0     # one-time bonus per checkpoint
  finish_bonus: 100.0        # bonus for completing the track
  collision_penalty: -5.0    # penalty for hitting walls
  off_track_penalty: -2.0    # penalty for leaving the road
```

These weights are applied in `src/trackmania_ghost_ai/environment.py:83`
inside the `shaped_reward()` function:

```python
def shaped_reward(raw_reward, info, cfg):
    r = raw_reward
    rc = cfg.reward

    # Velocity bonus
    speed = info.get("speed", 0.0)
    r += rc.velocity_weight * speed / 1000.0

    # Progress bonus
    progress = info.get("progress", 0.0)
    r += rc.progress_weight * progress

    # Checkpoint / finish bonuses
    if info.get("checkpoint_reached", False):
        r += rc.checkpoint_bonus
    if info.get("finished", False):
        r += rc.finish_bonus

    # Penalties
    if info.get("collision", False):
        r += rc.collision_penalty
    if info.get("off_track", False):
        r += rc.off_track_penalty

    return r
```

### The core principle

| You want...           | Increase...                        | Decrease...                  |
|-----------------------|------------------------------------|------------------------------|
| Faster driving        | `velocity_weight`                  | `collision_penalty` (less)   |
| Safer driving         | `collision_penalty` (more negative)| `velocity_weight`            |
| Track completion      | `progress_weight`, `finish_bonus`  | --                           |
| Aggressive cornering  | `velocity_weight`                  | `off_track_penalty` (less)   |
| Clean racing lines    | `progress_weight`                  | `collision_penalty` (more)   |

---

## 3. YAML-Only Recipes

No code changes needed -- just edit `configs/training.yaml` and retrain.

### Recipe A: Speed demon (maximum velocity, accept crashes)

```yaml
reward:
  velocity_weight: 3.0        # triple the speed reward
  progress_weight: 2.0        # lower progress emphasis
  checkpoint_bonus: 5.0
  finish_bonus: 50.0
  collision_penalty: -1.0     # barely penalise crashes
  off_track_penalty: -0.5
```

### Recipe B: Cautious driver (never crash, finish reliably)

```yaml
reward:
  velocity_weight: 0.3        # speed matters less
  progress_weight: 8.0        # track completion is king
  checkpoint_bonus: 20.0      # big checkpoint rewards
  finish_bonus: 200.0
  collision_penalty: -15.0    # heavy crash penalty
  off_track_penalty: -10.0
```

### Recipe C: Balanced racer (Author Medal hunter)

```yaml
reward:
  velocity_weight: 1.5
  progress_weight: 5.0
  checkpoint_bonus: 10.0
  finish_bonus: 100.0
  collision_penalty: -5.0
  off_track_penalty: -3.0
```

### Recipe D: Drift-friendly (reward speed through corners)

```yaml
reward:
  velocity_weight: 2.5        # high speed is heavily rewarded
  progress_weight: 4.0
  checkpoint_bonus: 10.0
  finish_bonus: 100.0
  collision_penalty: -3.0     # moderate -- drifting near walls is ok
  off_track_penalty: -1.0     # slight off-track is tolerated

algorithm:
  alpha: 0.5                  # more exploration to discover drift moves
  target_entropy: -5.0        # encourage action variety
```

### Using a custom config

```bash
# Copy the default config
cp configs/training.yaml configs/drift_training.yaml

# Edit the copy with your recipe
# ...

# Train with it
tm-train --config configs/drift_training.yaml
```

---

## 4. Code-Level Reward Customisation

For behaviours that cannot be expressed with simple weights, modify the
`shaped_reward()` function in `src/trackmania_ghost_ai/environment.py`.

### Example: Reward smooth steering (penalise jerky inputs)

```python
def shaped_reward(raw_reward, info, cfg):
    r = raw_reward
    rc = cfg.reward

    # ... existing reward components ...

    # NEW: Penalise large steering changes (jerk)
    prev_steer = info.get("prev_steer", 0.0)
    curr_steer = info.get("curr_steer", 0.0)
    steer_jerk = abs(curr_steer - prev_steer)
    r -= 0.5 * steer_jerk  # penalise abrupt direction changes

    return r
```

### Example: Reward maintaining high speed through curves

```python
    # NEW: Bonus for speed when LIDAR detects a curve ahead
    lidar = info.get("lidar", None)
    if lidar is not None:
        # If left/right beams are short (walls close = curve), reward speed
        left_beams = lidar[:5].mean()
        right_beams = lidar[-5:].mean()
        is_curve = min(left_beams, right_beams) < 0.5  # normalised
        if is_curve:
            speed = info.get("speed", 0.0)
            r += 0.002 * speed  # extra speed bonus in curves
```

### Example: Time penalty (encourage fastest possible completion)

```python
    # NEW: Small per-step penalty to encourage finishing quickly
    r -= 0.01  # every 50 ms costs a tiny bit
```

### Adding new reward parameters to the config

To make your custom rewards configurable via YAML, add fields to `RewardConfig`
in `src/trackmania_ghost_ai/config.py:78`:

```python
@dataclass
class RewardConfig:
    velocity_weight: float = 1.0
    progress_weight: float = 5.0
    checkpoint_bonus: float = 10.0
    finish_bonus: float = 100.0
    collision_penalty: float = -5.0
    off_track_penalty: float = -2.0

    # NEW custom parameters
    steer_jerk_penalty: float = 0.5
    curve_speed_bonus: float = 0.002
    time_penalty_per_step: float = 0.01
```

Then reference them in `shaped_reward()`:

```python
    r -= rc.steer_jerk_penalty * steer_jerk
    r += rc.curve_speed_bonus * speed
    r -= rc.time_penalty_per_step
```

And add to your YAML:

```yaml
reward:
  # ... existing ...
  steer_jerk_penalty: 0.5
  curve_speed_bonus: 0.002
  time_penalty_per_step: 0.01
```

---

## 5. Recording a Reward Trajectory

tmrl's default reward is based on **track progress** -- it measures how far
along a pre-recorded trajectory the car has advanced. You need to record this
trajectory for each new track.

```bash
# Record yourself driving the track from start to finish
python -m tmrl --record-reward
```

This saves `~/TmrlData/reward/reward.pkl` -- an array of 3D positions along
the track. The reward function then measures: *"How many trajectory points did
the car pass since the last step?"*

**Tips for a good trajectory recording:**
- Drive the full track at moderate speed (not your fastest)
- Stay centred on the road
- Complete the full lap without shortcuts
- The trajectory defines what "forward progress" means

---

## 6. Observation Tuning

The observation window affects what the agent can perceive and therefore what
behaviours it can learn.

### LIDAR frame stacking (`img_hist_len`)

```yaml
environment:
  img_hist_len: 4    # default: 4 consecutive LIDAR frames
```

| Value | Effect                                                      |
|-------|-------------------------------------------------------------|
| 2     | Less temporal context, faster reactions, less anticipation  |
| 4     | Default -- good for most tracks                             |
| 8     | More history, better at predicting upcoming curves          |

More frames = larger observation (and slower training), but the agent can
"see further into the past" and anticipate better.

### Action buffer (`act_buf_len`)

```yaml
environment:
  rtgym:
    act_buf_len: 2   # default: 2 previous actions in observation
```

The agent sees its last N actions. This helps it learn smooth control -- it
knows what it just did and can avoid contradicting itself (e.g., full gas
immediately after full brake).

| Value | Effect                                            |
|-------|---------------------------------------------------|
| 1     | Only sees last action; more reactive, less smooth |
| 2     | Default -- sees 2 previous actions                |
| 3     | Smoother control, larger observation vector        |

### Control frequency (`time_step_duration`)

```yaml
environment:
  rtgym:
    time_step_duration: 0.05   # 20 Hz (default)
```

| Value   | Frequency | Effect                                        |
|---------|-----------|-----------------------------------------------|
| 0.10    | 10 Hz     | Coarser control, easier to train, less precise|
| 0.05    | 20 Hz     | Default -- good balance                        |
| 0.033   | 30 Hz     | Finer control, harder to train, more precise  |

Higher frequency gives more precise control (better for drifting and tight
tracks) but requires more training steps.

---

## 7. Exploration vs Exploitation

SAC has a built-in exploration mechanism via its entropy term. The key
parameters:

```yaml
algorithm:
  alpha: 0.37                # entropy coefficient (higher = more exploration)
  learn_entropy_coef: true   # auto-tune alpha during training
  target_entropy: -7.0       # target entropy level
```

### When to change these

| Situation                          | Change                               |
|------------------------------------|--------------------------------------|
| Agent finds one mediocre strategy  | Increase `alpha` to 0.5-0.8         |
| Agent is too random after training | Decrease `alpha` to 0.1-0.2         |
| Agent never discovers drifting     | `target_entropy: -4.0` (more random)|
| Agent drives erratically           | `target_entropy: -9.0` (more focused)|
| Let SAC figure it out              | `learn_entropy_coef: true` (default) |

**Key insight**: Unusual driving techniques like drifting require the agent to
explore "bad" actions first (turning sharply while accelerating feels wrong
initially). Higher entropy in early training helps discover these.

---

## 8. Training Duration and Curriculum

### How long to train for each capability

| Capability            | Estimated time        | Notes                         |
|-----------------------|-----------------------|-------------------------------|
| Basic forward driving | 1-2 hours             | First thing it learns         |
| Consistent turns      | 3-5 hours             | Needs many crash examples     |
| Track completion      | 5-10 hours            | Full lap without crashing     |
| Smooth racing lines   | 10-20 hours           | Fine-tuning after completion  |
| Drifting              | 15-30+ hours          | Requires high exploration     |
| Author Medal times    | 30+ hours             | Hyperparameter sweeps needed  |

*Times assume RTX 3080 trainer, one worker, LIDAR interface.*

### Curriculum learning (progressive difficulty)

Train in stages instead of all at once:

1. **Stage 1** -- Learn to drive forward (high `progress_weight`, low penalties)
2. **Stage 2** -- Learn to turn (increase `collision_penalty`)
3. **Stage 3** -- Learn speed (increase `velocity_weight`, load Stage 2 weights)
4. **Stage 4** -- Fine-tune (balanced weights, lower learning rates)

```bash
# Stage 1: basics
tm-train --config configs/stage1_forward.yaml

# Stage 2: load stage 1 weights, tighter penalties
# (copy weights from ~/TmrlData/weights/ first)
tm-train --config configs/stage2_turns.yaml

# Stage 3: speed
tm-train --config configs/stage3_speed.yaml
```

Between stages, the weights persist in `~/TmrlData/weights/`. Each new
training run picks up where the last left off.

---

## 9. Verifying the New Capability

After training, run evaluation episodes to observe the behaviour:

```bash
# Watch the AI drive (5 test episodes)
tm-evaluate --episodes 5
```

### What to look for

| Desired capability  | What you should see                              |
|---------------------|--------------------------------------------------|
| Faster driving      | Higher average speed in logs                     |
| Drifting            | Lateral sliding through corners, no wall contact |
| Cautious driving    | Slower but zero crashes, always finishes         |
| Smooth lines        | Minimal steering oscillation, clean arcs         |
| Air-braking         | Brief brake taps before jumps/drops              |

### Comparing configurations

Train multiple configs and compare their evaluation metrics:

```bash
# Train config A
tm-train --config configs/speed_demon.yaml
# Rename weights
cp ~/TmrlData/weights/ghost-ai-default.tmod ~/TmrlData/weights/speed_demon.tmod

# Train config B
tm-train --config configs/cautious.yaml
cp ~/TmrlData/weights/ghost-ai-default.tmod ~/TmrlData/weights/cautious.tmod

# Evaluate each
tm-evaluate --weights ~/TmrlData/weights/speed_demon.tmod --episodes 10
tm-evaluate --weights ~/TmrlData/weights/cautious.tmod --episodes 10
```

---

## Quick Reference

| I want to...                    | Edit this                                     |
|---------------------------------|-----------------------------------------------|
| Make the agent faster           | `reward.velocity_weight` (increase)           |
| Make the agent safer            | `reward.collision_penalty` (more negative)    |
| Encourage track completion      | `reward.progress_weight` (increase)           |
| Encourage exploration           | `algorithm.alpha` (increase)                  |
| Add a new reward signal         | `RewardConfig` dataclass + `shaped_reward()`  |
| Change how far it sees          | `environment.img_hist_len`                    |
| Change how often it acts        | `environment.rtgym.time_step_duration`        |
| Train on a new track            | `python -m tmrl --record-reward`              |
