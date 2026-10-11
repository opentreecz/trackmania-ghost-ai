# Training Workflow

Deep dive into how the training loop works, what each phase does, how long
things take, and how to diagnose problems.

## Table of Contents

1. [The Training Loop Explained](#1-the-training-loop-explained)
2. [Epochs, Rounds, and Steps](#2-epochs-rounds-and-steps)
3. [Data Flow](#3-data-flow)
4. [Training Duration Estimates](#4-training-duration-estimates)
5. [Common Pitfalls](#5-common-pitfalls)
6. [Troubleshooting Guide](#6-troubleshooting-guide)
7. [Performance Optimisation](#7-performance-optimisation)

---

## 1. The Training Loop Explained

Training has three concurrent processes:

```
Worker (game)              Server (relay)              Trainer (GPU)
     |                          |                           |
     |-- collect episode ------>|                           |
     |   (obs, act, rew, ...)   |-- forward samples ------>|
     |                          |                           |-- gradient step
     |                          |                           |-- gradient step
     |                          |                           |-- ...
     |                          |<-- updated weights -------|
     |<-- broadcast weights ----|                           |
     |                          |                           |
     |-- next episode -------->|                           |
     |   ...                    |   ...                     |   ...
```

### Worker cycle

1. Reset the race (respawn car at start)
2. For each timestep (50 ms):
   - Capture screenshot, compute LIDAR
   - Feed observation to the actor network
   - Get action [gas, brake, steer]
   - Send action to the game via virtual gamepad
   - Record (obs, action, reward, next_obs, done) in buffer
3. When episode ends (crash or `ep_max_length` reached):
   - Send the episode buffer to the server
   - Check for updated weights from the server
   - Start the next episode

### Trainer cycle

1. Wait for `start_training` samples in the replay buffer
2. For each epoch:
   - For each round:
     - For each step:
       - Sample a batch from the replay buffer
       - Compute critic loss (TD error on twin Q-networks)
       - Compute actor loss (maximise Q-value + entropy)
       - Update networks via Adam optimiser
       - Soft-update target networks (polyak averaging)
     - Check for new samples from server; add to replay buffer
   - Save checkpoint
   - Broadcast updated weights to workers

---

## 2. Epochs, Rounds, and Steps

```yaml
training:
  epochs: 200          # outer loop
  rounds: 50           # middle loop
  steps: 100           # inner loop (gradient updates)
  start_training: 1000 # wait for this many transitions
```

### Total gradient steps

`epochs x rounds x steps = 200 x 50 x 100 = 1,000,000`

### What happens at each level

| Level  | What happens                                              | Frequency          |
|--------|-----------------------------------------------------------|--------------------|
| Step   | One gradient update on one batch from the replay buffer   | Every ~1 ms (GPU)  |
| Round  | 100 gradient steps, then check for new samples            | Every ~100 ms      |
| Epoch  | 50 rounds, then save checkpoint + broadcast weights       | Every ~5 seconds   |

### Adjusting for your hardware

| Hardware                  | Recommended                              |
|---------------------------|------------------------------------------|
| RTX 4090 (fast GPU)       | Default or increase `steps` to 200       |
| RTX 3060 (mid GPU)        | Default works well                       |
| GTX 1060 (slow GPU)       | Reduce: `steps: 50, rounds: 30`          |
| CPU only                  | `epochs: 50, rounds: 20, steps: 50`      |

---

## 3. Data Flow

### Observation (what the agent sees)

```
TM2020 game window
    |
    v
Screenshot capture (mss library, every 50ms)
    |
    v
LIDAR computation (19 rays from screen centre to black borders)
    |
    v
Observation tuple: (speed[1], lidar[4,19], prev_act1[3], prev_act2[3])
    |
    v
Actor neural network: obs -> [gas, brake, steer]
    |
    v
Virtual gamepad (vgamepad) -> game input
```

### Training data (what feeds the gradient)

```
Worker collects: (obs, action, reward, next_obs, terminated, truncated)
    |
    v
Sent to Server via tlspyo (TCP, pickled Python objects)
    |
    v
Server relays to Trainer
    |
    v
Appended to replay buffer (GenericTorchMemory, in RAM)
    |
    v
Random batch sampled for each gradient step
```

### Weight synchronisation

```
Trainer trains actor network
    |
    v
Every update_model_interval steps:
    actor.save("weights.tmod")
    binary = read("weights.tmod")
    server.broadcast(binary, to="workers")
    |
    v
Worker receives, writes to disk, calls actor.load()
    |
    v
Next episode uses updated policy
```

---

## 4. Training Duration Estimates

### By capability (LIDAR, RTX 3080, single worker)

| Milestone              | Time          | Gradient steps |
|------------------------|---------------|----------------|
| First forward movement | 5-15 min      | ~5,000         |
| Consistent turns       | 1-3 hours     | ~50,000        |
| First track completion | 3-5 hours     | ~150,000       |
| Reliable completion    | 5-10 hours    | ~300,000       |
| Optimised lap times    | 10-20 hours   | ~600,000       |
| Competitive times      | 20-50+ hours  | ~1,000,000+    |

### By hardware

| Setup                          | Relative speed | 50K steps takes |
|--------------------------------|----------------|-----------------|
| RTX 4090 + fast CPU            | 2x             | ~25 min         |
| RTX 3080 + modern CPU          | 1x (baseline)  | ~50 min         |
| RTX 3060 + mid CPU             | 0.7x           | ~70 min         |
| GTX 1060 + older CPU           | 0.3x           | ~3 hours        |
| CPU only (no CUDA)             | 0.05x          | ~16 hours       |

### Multiple workers

Adding more workers (each running the game) increases data collection:

| Workers | Data rate   | Effect                                  |
|---------|-------------|-----------------------------------------|
| 1       | 1x          | Default                                 |
| 2       | ~1.8x       | Almost twice the experience per hour    |
| 4       | ~3.2x       | Diminishing returns on data diversity   |

---

## 5. Common Pitfalls

### Pitfall 1: Worker not sending data

**Symptom**: `memory_len` stays at 0, trainer never starts.

**Causes**:
- Game window not focused (click it after starting the worker)
- OpenPlanet plugin not loaded (press F3, reload the plugin)
- Wrong window position (must be top-left, windowed mode)
- Game is paused or in a menu

### Pitfall 2: Training is unstable (loss exploding)

**Symptom**: `loss_critic` or `loss_actor` jumps to very large values.

**Fix**:
```yaml
algorithm:
  actor:
    lr: 0.0001        # reduce from 0.0003
  critic:
    lr: 0.00002       # reduce from 0.00005
```

### Pitfall 3: Agent stuck in local optimum

**Symptom**: `return_train` plateaus and stops improving.

**Fix**:
```yaml
algorithm:
  alpha: 0.6           # increase exploration
  target_entropy: -4.0  # encourage more randomness
```

### Pitfall 4: Agent forgets what it learned

**Symptom**: Performance suddenly drops after improving.

**Causes**:
- Learning rate too high (reduce by 3-5x)
- Replay buffer too small (old good experiences get overwritten)
- Batch size too small (noisy gradients)

**Fix**:
```yaml
memory:
  capacity: 2000000     # double the buffer
  batch_size: 512       # larger batches
algorithm:
  actor:
    lr: 0.0001          # slower learning
```

### Pitfall 5: "Timestep timeout" warnings

**Symptom**: Worker prints timestep timeout warnings.

**Cause**: The game + screenshot capture + LIDAR computation takes longer
than `time_step_duration` (50 ms).

**Fix**:
- Reduce TM2020 graphics quality
- Set max FPS to 30 in game settings
- Use a faster GPU for rendering
- Increase `time_step_duration` to 0.1 (10 Hz, less precise control)

---

## 6. Troubleshooting Guide

| Problem                            | Check                                    | Solution                          |
|------------------------------------|------------------------------------------|-----------------------------------|
| Worker crashes on startup          | Python version >= 3.10?                  | Upgrade Python                    |
| "Connection refused"               | Is the server running?                   | Start server first                |
| "No module named vgamepad"         | Is vgamepad installed?                   | `pip install vgamepad`            |
| Car not moving                     | Is the game window focused?              | Click the game window             |
| LIDAR all zeros                    | Is the track black-bordered?             | Use tmrl-test or tmrl-train maps  |
| Memory not growing                 | Is the worker connected to the server?   | Check server logs for connections |
| Training not starting              | `memory_len` > `start_training`?         | Wait for more samples             |
| NaN in losses                      | LR too high? gamma too close to 1?       | Lower LR, gamma to 0.99          |
| Weights not updating               | `update_model_interval` too high?        | Reduce to 50                      |
| Out of memory (RAM)                | Buffer too large?                        | Reduce `memory.capacity`          |
| Out of memory (VRAM)               | Batch too large? Network too big?        | Reduce batch_size or hidden_sizes |

---

## 7. Performance Optimisation

### Speed up training

1. **Use CUDA**: Ensure `torch.cuda.is_available()` returns True
2. **Larger batches**: `batch_size: 512` (if VRAM allows)
3. **More gradient steps per round**: `steps: 200`
4. **Multiple workers**: Run 2-3 workers on separate machines

### Speed up data collection

1. **Lower game resolution**: Reduce TM2020 graphics
2. **Faster game FPS**: Uncap FPS in game settings
3. **Smaller window**: Minimum 958x488 for LIDAR

### Reduce memory usage

1. **Smaller replay buffer**: `capacity: 500000`
2. **Smaller network**: `hidden_sizes: [128, 128]`
3. **Smaller batch**: `batch_size: 128`

### Distributed training

Split across machines for maximum efficiency:

```
Machine A (has the game):  tm-worker
Machine B (has the GPU):   tm-server + tm-train
```

This lets the GPU machine focus entirely on training without the overhead
of running the game.
