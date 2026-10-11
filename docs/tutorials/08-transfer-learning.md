# Using Previous Drives and Transfer Learning

How to reuse experience from past training sessions, load pre-trained weights,
fine-tune models on new tracks, and bootstrap new agents from old ones.

## Table of Contents

1. [Types of Reusable Knowledge](#1-types-of-reusable-knowledge)
2. [Loading Pre-trained Weights](#2-loading-pre-trained-weights)
3. [Fine-tuning on a New Track](#3-fine-tuning-on-a-new-track)
4. [Reusing the Replay Buffer](#4-reusing-the-replay-buffer)
5. [Offline RL from Saved Data](#5-offline-rl-from-saved-data)
6. [Transfer Between Configurations](#6-transfer-between-configurations)
7. [Curriculum Learning](#7-curriculum-learning)
8. [Best Practices](#8-best-practices)

---

## 1. Types of Reusable Knowledge

There are three kinds of learned knowledge you can transfer:

| Type                | File                          | What it contains                    |
|---------------------|-------------------------------|-------------------------------------|
| **Policy weights**  | `.tmod` (1-3 MB)              | The neural network that decides how to drive |
| **Replay buffer**   | `.tcpt` checkpoint (100MB+)   | Collected driving experience (state/action/reward tuples) |
| **Reward trajectory** | `reward.pkl`                | Human-driven reference path for a specific track |

### When to reuse what

| Scenario                                  | Reuse                          |
|-------------------------------------------|--------------------------------|
| Same track, continue training             | Weights + buffer (auto-resume) |
| New track, similar road style             | Weights only (fine-tune)       |
| Same track, different reward function     | Buffer only (reset weights)    |
| Completely different track and style      | Nothing (train from scratch)   |
| Demo/showcase without training            | Pre-trained weights            |

---

## 2. Loading Pre-trained Weights

### From tmrl's included models

tmrl ships with pre-trained weights for the LIDAR interface:

```bash
# Copy the pre-trained model to your run's weight location
cp ~/TmrlData/weights/SAC_4_LIDAR_pretrained.tmod \
   ~/TmrlData/weights/ghost-ai-default.tmod

# Evaluate immediately (no training needed)
tm-evaluate --episodes 5
```

### From a previous training run

```bash
# Copy weights from one run to another
cp ~/TmrlData/weights/track-a-model.tmod \
   ~/TmrlData/weights/ghost-ai-default.tmod
```

### From a shared file

If someone gives you a `.tmod` file:

```bash
cp shared_model.tmod ~/TmrlData/weights/ghost-ai-default.tmod
tm-evaluate --episodes 5
```

### The worker loads weights automatically

When a worker starts, it checks for existing weights at
`~/TmrlData/weights/{RUN_NAME}.tmod`. If found, it loads them instead of
using random weights. This means any `.tmod` file you place there becomes
the starting policy.

---

## 3. Fine-tuning on a New Track

Fine-tuning = starting training from pre-learned weights instead of random.
The agent already knows basic driving; it just needs to adapt to the new
track's corners and rhythm.

### Step 1: Record the new track's reward trajectory

```bash
# Load the new track in TM2020
# Then record yourself driving it
python -m tmrl --record-reward
```

### Step 2: Copy the pre-trained weights

```bash
cp ~/TmrlData/weights/old-track-model.tmod \
   ~/TmrlData/weights/new-track-run.tmod
```

### Step 3: Configure for fine-tuning

Create a new config with lower learning rates (the model is already close
to good; large updates would destroy what it learned):

```yaml
# configs/fine-tune.yaml
run:
  name: "new-track-run"

algorithm:
  actor:
    lr: 0.00005       # 6x lower than default 0.0003
  critic:
    lr: 0.00001       # 5x lower than default 0.00005
  alpha: 0.2          # less exploration needed

training:
  epochs: 100         # fewer epochs needed
  start_training: 500 # can start sooner with pre-trained weights
```

### Step 4: Train

```bash
tm-train --config configs/fine-tune.yaml
```

The agent starts from competent driving and adapts to the new track.
Fine-tuning typically takes 2-5x less time than training from scratch.

### Typical fine-tuning results

| Starting point                | Time to complete new track |
|-------------------------------|----------------------------|
| Random weights (from scratch) | 5-10 hours                 |
| Pre-trained on similar track  | 1-3 hours                  |
| Pre-trained on same track     | <1 hour (just adapting)    |

---

## 4. Reusing the Replay Buffer

The replay buffer contains millions of (state, action, reward, next_state)
transitions. This experience is valuable -- it represents hours of driving.

### Keep the buffer, reset the model

If you want to try a different network architecture or algorithm settings
without re-collecting data:

Set `RESET_TRAINING: true` in `~/TmrlData/config/config.json`:

```json
{
  "RESET_TRAINING": true
}
```

This creates a **fresh model and optimiser** but **keeps the replay buffer**.
The new network immediately starts training on previously collected experience.

After the first training run starts, set it back to `false` to resume
normally next time.

### Save the replay buffer separately

The buffer is embedded in the checkpoint. To save it independently:

```python
import pickle

# Extract buffer from checkpoint
with open("~/TmrlData/checkpoints/my-run_t.tcpt", "rb") as f:
    training = pickle.load(f)

buffer_data = training.memory.data

with open("saved_buffer.pkl", "wb") as f:
    pickle.dump(buffer_data, f)
```

### Load a saved buffer

Place it at `~/TmrlData/dataset/data.pkl`. tmrl's memory class checks for
this file on initialisation and pre-loads it:

```bash
cp saved_buffer.pkl ~/TmrlData/dataset/data.pkl
```

---

## 5. Offline RL from Saved Data

Offline RL trains entirely from previously collected data, without running
the game. Useful when:

- You have a large dataset but limited game access
- You want to experiment with hyperparameters without running episodes
- You collected data from human play or a previous agent

### Step 1: Collect data

```bash
# Run the worker to collect data (or use existing checkpoint)
tm-worker --config configs/training.yaml
# Let it collect for a few hours, then Ctrl+C
```

### Step 2: Save the buffer

The checkpoint at `~/TmrlData/checkpoints/{RUN_NAME}_t.tcpt` contains
the replay buffer.

### Step 3: Train offline

Start the trainer without a worker. It will use whatever data is in the
checkpoint:

```bash
# Only run server + trainer (no worker)
tm-server --config configs/training.yaml &
tm-train --config configs/training.yaml
```

The trainer processes the existing buffer data. Without a worker, no new
data arrives, but the agent can still improve by better exploiting existing
experience.

**Caveat**: Purely offline RL degrades after many gradient steps because
the policy drifts away from the data distribution. It works best for
initial training or short fine-tuning, not extended runs.

---

## 6. Transfer Between Configurations

### Same observation space, different reward

Weights transfer directly because the network architecture matches:

```bash
# Train with reward config A
tm-train --config configs/reward_a.yaml
cp ~/TmrlData/weights/reward-a.tmod ~/TmrlData/weights/reward-b.tmod

# Fine-tune with reward config B (different reward weights)
tm-train --config configs/reward_b.yaml
```

### Same observation space, different network size

Weights do **not** transfer directly -- the tensor shapes are different.

```yaml
# Config A: hidden_sizes: [256, 256]
# Config B: hidden_sizes: [512, 256]
# These are incompatible -- cannot load A's weights into B's network
```

Options:
- Train from scratch with the new architecture
- Use partial weight loading (advanced; `strict=False` in PyTorch)

### Different observation space

If `img_hist_len` or `act_buf_len` changes, the input dimension changes,
and the first layer weights are incompatible. Train from scratch.

### Different action space

The action space is fixed at 3 (`[gas, brake, steer]`) in the tmrl
interface. This cannot be changed without modifying tmrl itself.

---

## 7. Curriculum Learning

Train in progressive stages, each building on the last:

### Stage 1: Learn to drive forward

```yaml
# configs/stage1.yaml
run:
  name: "curriculum-stage1"
reward:
  velocity_weight: 2.0
  progress_weight: 8.0
  collision_penalty: -1.0     # gentle penalties
training:
  epochs: 50
```

```bash
tm-train --config configs/stage1.yaml
```

### Stage 2: Learn to avoid walls

```yaml
# configs/stage2.yaml
run:
  name: "curriculum-stage2"    # same name = auto-resume!
reward:
  velocity_weight: 1.5
  progress_weight: 5.0
  collision_penalty: -8.0     # harsh penalties now
training:
  epochs: 100
```

**Wait** -- copy the weights first if using a different run name:

```bash
cp ~/TmrlData/weights/curriculum-stage1.tmod \
   ~/TmrlData/weights/curriculum-stage2.tmod
tm-train --config configs/stage2.yaml
```

Or use the **same** `run.name` across stages. tmrl auto-resumes from the
checkpoint. Just change the reward config and restart training.

### Stage 3: Optimise speed

```yaml
# configs/stage3.yaml
run:
  name: "curriculum-stage2"    # continue from stage 2
reward:
  velocity_weight: 3.0
  collision_penalty: -5.0
algorithm:
  actor:
    lr: 0.0001                 # lower LR for fine-tuning
training:
  epochs: 200
```

### Why curriculum learning works

| Stage    | Agent learns...      | Because...                              |
|----------|----------------------|-----------------------------------------|
| Stage 1  | Forward movement     | High progress reward, low penalties     |
| Stage 2  | Turning, wall avoidance | Harsh crash penalty forces caution   |
| Stage 3  | Speed optimisation   | High velocity reward, balanced penalties|

Each stage starts from the previous stage's learned policy, so it does not
need to re-learn basic skills.

---

## 8. Best Practices

### Do

- **Always record a fresh reward trajectory** when switching tracks
- **Lower learning rates** when fine-tuning (5-10x lower)
- **Keep backups** of good weight files before starting new training
- **Use unique run names** for different experiments
- **Save weight snapshots** periodically during long training runs

### Don't

- **Don't change network architecture** and try to load old weights
- **Don't change observation space** (img_hist_len, act_buf_len) and load old weights
- **Don't train offline for too long** -- policy drift degrades quality
- **Don't overwrite good weights** without a backup
- **Don't mix weights trained on LIDAR** with a FULL (image) config or vice versa

### Transfer learning cheat sheet

```
Same track, same config, continue training:
  -> Just run tm-train again (auto-resume from checkpoint)

Same track, different reward weights:
  -> Copy .tmod, lower LR, train

New track, same style:
  -> Record new reward.pkl, copy .tmod, lower LR, train

New track, different style:
  -> Record new reward.pkl, train from scratch

Different network architecture:
  -> Train from scratch (weights incompatible)
```
