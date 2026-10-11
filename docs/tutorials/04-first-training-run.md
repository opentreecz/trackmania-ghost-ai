# First Training Run

Step-by-step guide to train the AI from scratch on a track and watch it learn
to drive.

## Table of Contents

1. [Before You Start](#1-before-you-start)
2. [Record a Reward Trajectory](#2-record-a-reward-trajectory)
3. [Configure the Training](#3-configure-the-training)
4. [Prepare the Game Window](#4-prepare-the-game-window)
5. [Launch Training (All-in-One)](#5-launch-training-all-in-one)
6. [Launch Training (Three Terminals)](#6-launch-training-three-terminals)
7. [What Happens During Training](#7-what-happens-during-training)
8. [Monitor Progress](#8-monitor-progress)
9. [When to Stop](#9-when-to-stop)
10. [What You Should See](#10-what-you-should-see)

---

## 1. Before You Start

Ensure you have completed the [Prerequisites and Installation](./03-prerequisites-installation.md)
tutorial. You should have:

- [x] TM2020 installed and running
- [x] OpenPlanet with the TMRL plugin loaded
- [x] `pip install -e ".[dev]"` completed
- [x] `python -m tmrl --install` done (TmrlData created)
- [x] Track map copied to TM2020's My Maps folder
- [x] `python -m tmrl --check-environment` passes

---

## 2. Record a Reward Trajectory

The agent needs a reference trajectory to measure track progress. You drive
the track once to create it.

```bash
python -m tmrl --record-reward
```

**How to record:**

1. The game should be running with the track loaded
2. After running the command, click the TM2020 window to give it focus
3. Drive the full track from start to finish at moderate speed
4. Stay roughly centred on the road
5. Complete the full lap without shortcuts
6. The recording stops when you cross the finish line

This saves `~/TmrlData/reward/reward.pkl` -- an array of 3D positions along
your driven path.

**Tips:**
- Drive at 60-70% of your best speed, not maximum
- A clean, centred line is better than a fast one
- If you make a mistake, re-run the command and drive again
- The trajectory does not need to be perfect -- it just defines "forward"

---

## 3. Configure the Training

The default config works well for a first run:

```bash
cat configs/training.yaml
```

Key settings to verify:

```yaml
run:
  name: "my-first-run"       # change this to identify your run

environment:
  interface: "TM20LIDAR"     # LIDAR observations (not images)
  rtgym:
    time_step_duration: 0.05  # 20 Hz (default, good)
    ep_max_length: 2000       # ~100 seconds per episode

algorithm:
  gamma: 0.995
  alpha: 0.37

training:
  epochs: 50                  # start with 50 for a first run
  rounds: 50
  steps: 100
```

For your very first run, 50 epochs is enough to see if training is working.
You can always resume later.

---

## 4. Prepare the Game Window

1. Launch TM2020
2. Load your track: Play > Local > My Maps > select track
3. Set **windowed mode** (Settings > Graphics > Display > Windowed)
4. Resize to **958 x 488** pixels
5. Position at the **top-left corner** of your screen
6. Switch to **cockpit view**: press `3`
7. Hide the car model: press `Alt+C` (or via OpenPlanet settings)
8. Press **F3** and verify TMRL plugin shows "waiting for connection"

The window must stay visible and at the top-left. Do not minimise it.

---

## 5. Launch Training (All-in-One)

The simplest way -- runs server, worker, and trainer in a single process:

```bash
tm-train --config configs/training.yaml --verbose
```

After launching:
1. **Immediately click the TM2020 window** to give it focus
2. The worker needs the game window focused to capture screenshots
3. You should see the car start moving within a few seconds

The agent will initially drive randomly (exploring). This is expected.

---

## 6. Launch Training (Three Terminals)

For more control and better logging, use three separate terminals:

### Terminal 1: Server (relay)

```bash
tm-server --config configs/training.yaml --verbose
```

Wait until you see "Server listening on port 6666".

### Terminal 2: Trainer (gradient updates)

```bash
tm-train --config configs/training.yaml --verbose
```

The trainer waits for the server, then waits for samples from the worker.

### Terminal 3: Worker (plays the game)

```bash
tm-worker --config configs/training.yaml --verbose
```

**Immediately click the TM2020 window after starting the worker.**

---

## 7. What Happens During Training

### Phase 1: Random exploration (first 5-15 minutes)

- The agent outputs random actions (gas, brake, steer)
- It crashes constantly -- this is expected
- It is collecting experience in the replay buffer
- No gradient updates yet (waiting for `start_training` samples)

### Phase 2: Learning begins (15-60 minutes)

- The trainer starts gradient updates
- The agent begins to drive forward inconsistently
- Episodes get slightly longer (fewer immediate crashes)
- You will see "model weights have been updated" in the worker log

### Phase 3: Basic driving (1-3 hours)

- The agent can drive forward and make some turns
- It still crashes on sharp corners
- Episode returns start increasing in the logs

### Phase 4: Track completion (3-10 hours)

- The agent completes the track occasionally
- It learns to slow down for corners
- Lap times start appearing in the logs

### Phase 5: Optimisation (10+ hours)

- The agent reliably completes laps
- It optimises speed through corners
- Lap times slowly improve

---

## 8. Monitor Progress

### Terminal output

The trainer logs per-round statistics:

```
Epoch 12 | Round 25 | loss_actor: 0.34 | loss_critic: 0.21 | return_train: 45.2 | memory_len: 15420
```

Key metrics:
- `return_train`: Average episode reward (should increase over time)
- `return_test`: Test episode reward (more stable, less noisy)
- `memory_len`: Number of transitions in the replay buffer
- `loss_actor` / `loss_critic`: Neural network losses (should decrease)

### Weights & Biases (optional)

Enable W&B for a visual dashboard:

```yaml
# configs/training.yaml
run:
  wandb:
    enabled: true
    project: "trackmania-ghost-ai"
```

Or pass the flag:

```bash
python -m tmrl --trainer --wandb
```

Then visit [wandb.ai](https://wandb.ai) to see real-time graphs of:
- Episode returns over time
- Loss curves
- Episode lengths
- Memory size growth

### Quick health checks

| Sign                               | Meaning                            |
|------------------------------------|------------------------------------|
| `return_train` increasing          | Agent is learning                  |
| `return_train` flat for 10+ epochs | Possibly stuck; tune alpha/rewards |
| `memory_len` not growing           | Worker not sending samples         |
| `loss_critic` exploding            | Reduce learning rates              |
| `loss_actor` oscillating wildly    | Reduce actor LR                    |

---

## 9. When to Stop

### Manual stop

Press `Ctrl+C` in the trainer terminal. Weights are saved automatically in
`~/TmrlData/weights/`.

### The agent finishes laps

Once you see consistent track completions in the test episodes, you have a
working agent. You can:

- Stop and evaluate: `tm-evaluate --episodes 10`
- Continue training for faster lap times
- Switch to a more aggressive reward config

### Time-based

For a first run:

| Duration    | Expected result                            |
|-------------|--------------------------------------------|
| 30 minutes  | Agent drives forward, crashes on turns     |
| 2 hours     | Agent makes some turns, inconsistent       |
| 5 hours     | Agent completes laps occasionally          |
| 10 hours    | Agent completes laps reliably              |

*Times for RTX 3080 trainer, one worker, LIDAR interface, tmrl practice track.*

---

## 10. What You Should See

### In the game window

- **First 5 minutes**: Car drives randomly, crashes into walls, respawns
- **After 30 minutes**: Car drives forward, sometimes turns
- **After 2 hours**: Car navigates most corners
- **After 5+ hours**: Car drives a full lap

### In the terminal

```
# Early training (random)
Episode 1  | return: 2.3  | length: 45    | crashed
Episode 2  | return: 1.8  | length: 32    | crashed

# After some learning
Episode 50 | return: 18.7 | length: 312   | crashed
Episode 51 | return: 22.1 | length: 445   | crashed

# Getting better
Episode 200| return: 67.4 | length: 1200  | finished
Episode 201| return: 71.2 | length: 1150  | finished
```

### After training

```bash
# Watch the trained agent drive
tm-evaluate --episodes 5

# Your weights are at:
ls ~/TmrlData/weights/
# my-first-run.tmod        (worker weights)
# my-first-run_t.tmod      (trainer weights)
```

---

## Quick Checklist

```
[ ] Game running, track loaded, windowed, top-left, cockpit view
[ ] OpenPlanet TMRL plugin active ("waiting for connection")
[ ] Reward trajectory recorded (reward.pkl exists)
[ ] Config file ready with unique run name
[ ] Training launched (tm-train or 3 terminals)
[ ] Game window clicked/focused after worker starts
[ ] Car is moving (even randomly = success)
[ ] return_train increasing over time
```
