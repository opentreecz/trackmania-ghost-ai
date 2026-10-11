# Replay and Evaluation

How to watch the trained AI drive, run test episodes, compare models, and
record ghost replays.

## Table of Contents

1. [Running Evaluation Episodes](#1-running-evaluation-episodes)
2. [Test vs Train Mode](#2-test-vs-train-mode)
3. [Watching the AI Drive](#3-watching-the-ai-drive)
4. [Comparing Models](#4-comparing-models)
5. [Recording Ghost Replays](#5-recording-ghost-replays)
6. [Recording Video](#6-recording-video)
7. [Analysing Performance](#7-analysing-performance)

---

## 1. Running Evaluation Episodes

### Using the CLI

```bash
# Run 5 test episodes with the current weights
tm-evaluate --episodes 5

# With a specific weight file
tm-evaluate --weights ~/TmrlData/weights/my-model.tmod --episodes 10

# With a custom config
tm-evaluate --config configs/training.yaml --episodes 5
```

### Using tmrl directly

```bash
# Pure test mode (no training, no server needed)
python -m tmrl --test
```

This runs the worker in standalone mode -- no server or trainer required.
It loads weights from `~/TmrlData/weights/{RUN_NAME}.tmod` and drives the
car using the deterministic policy.

---

## 2. Test vs Train Mode

The actor network behaves differently in test vs train mode:

### Train mode (`test=False`)

The policy **samples** from a Gaussian distribution:

```python
# In SquashedGaussianMLPActor.forward()
pi_dist = Normal(mu, std)
pi_action = pi_dist.rsample()  # random sample
```

This introduces deliberate randomness for exploration. The car will
occasionally make unexpected moves -- this is how it discovers new strategies.

### Test mode (`test=True`)

The policy uses the **mean** action (deterministic):

```python
pi_action = mu  # no randomness
```

The car drives as well as the current policy can, without any exploration
noise. This is what you see during evaluation.

### The difference in practice

| Aspect         | Train mode                    | Test mode                       |
|----------------|-------------------------------|---------------------------------|
| Actions        | Sampled (random component)    | Deterministic (mean only)       |
| Behaviour      | Exploratory, sometimes erratic| Smooth, consistent              |
| Purpose        | Collect diverse experience    | Evaluate true performance       |
| Lap times      | Varies widely                 | Consistent                      |

### Test episode interval during training

During training, the worker periodically runs a test episode:

```yaml
# In tm-worker --test-interval 10
# Every 10 training episodes, one test episode runs
```

This gives you a clean performance measure while training continues.

---

## 3. Watching the AI Drive

### Setup

1. Launch TM2020 with the track loaded
2. Position the window (top-left, windowed, cockpit view)
3. Ensure the TMRL OpenPlanet plugin is active

### Run the AI

```bash
# Standalone test mode (simplest)
python -m tmrl --test

# Or via our CLI
tm-evaluate --episodes 10
```

4. Click the TM2020 window immediately
5. Watch the AI drive

### What to observe

- **Steering smoothness**: Does it oscillate or drive clean lines?
- **Corner approach**: Does it brake before corners or crash into walls?
- **Speed management**: Does it go full throttle on straights?
- **Recovery**: When it clips a wall, does it recover or spiral?
- **Consistency**: Are all laps similar, or does quality vary?

---

## 4. Comparing Models

### Side-by-side evaluation

Evaluate different models and compare their returns:

```bash
# Model A: speed-focused training
cp ~/TmrlData/weights/speed-model.tmod ~/TmrlData/weights/ghost-ai-default.tmod
python -m tmrl --test
# Note the average return and lap times

# Model B: safety-focused training
cp ~/TmrlData/weights/safe-model.tmod ~/TmrlData/weights/ghost-ai-default.tmod
python -m tmrl --test
# Compare
```

### Metrics to compare

| Metric                | What it tells you                            |
|-----------------------|----------------------------------------------|
| Average return        | Overall performance score                    |
| Lap time              | How fast it completes the track              |
| Completion rate       | Fraction of episodes that finish the track   |
| Crash count           | How often it hits walls                      |
| Average episode length| Longer = survives longer = drives better     |

### Comparing training runs over time

Save weight snapshots at regular intervals:

```bash
# During training, every N epochs copy the weights
cp ~/TmrlData/weights/my-run.tmod checkpoints/my-run_epoch050.tmod
cp ~/TmrlData/weights/my-run.tmod checkpoints/my-run_epoch100.tmod
cp ~/TmrlData/weights/my-run.tmod checkpoints/my-run_epoch150.tmod
```

Then evaluate each to see how the agent improved:

```bash
for epoch in 050 100 150 200; do
  cp checkpoints/my-run_epoch${epoch}.tmod ~/TmrlData/weights/ghost-ai-default.tmod
  echo "=== Epoch $epoch ==="
  python -m tmrl --test
done
```

---

## 5. Recording Ghost Replays

TrackMania has a native replay system. You can save the AI's drives as
in-game ghost replays.

### Enable replay saving

In `~/TmrlData/config/config.json`, set:

```json
{
  "interface_kwargs": {
    "save_replays": true
  }
}
```

Or pass it when constructing the interface.

### How it works

When an episode completes (car crosses the finish line), the tmrl plugin
signals OpenPlanet to save the replay. The replay is saved in TM2020's
native `.Replay.Gbx` format in:

```
Documents/Trackmania/Replays/
```

### Viewing replays

1. In TM2020: Play > Local > My Replays
2. Select the replay to watch
3. The replay shows the exact inputs the AI used
4. You can view it from any camera angle

### Replays as ghost opponents

Load a saved replay as a ghost to race against:
1. Start a race on the same track
2. The ghost appears as a translucent car
3. You can race against your AI's best time

---

## 6. Recording Video

tmrl does not record video directly. Use external tools.

### Method 1: TM2020 Replay Editor (best quality)

1. Save a replay (see above)
2. Open the Replay Editor in TM2020
3. Place cameras, adjust angles
4. Export to video at any resolution

This produces the highest quality output because it uses the game's
renderer.

### Method 2: OBS Studio (real-time)

1. Install [OBS Studio](https://obsproject.com/)
2. Add a "Window Capture" source pointing to TM2020
3. Start recording
4. Run the AI: `python -m tmrl --test`
5. Stop recording when done

### Method 3: ffmpeg (command line, Linux)

```bash
# Record the X11 display (works with Xvfb too)
ffmpeg -f x11grab -framerate 30 -video_size 958x488 \
  -i :0.0+0,0 -c:v libx264 -preset fast \
  ai_drive.mp4

# Start the AI in another terminal
python -m tmrl --test

# Press Ctrl+C in ffmpeg terminal to stop
```

### Method 4: Windows Game Bar

Press `Win+G` to open Game Bar, click the record button. Simple but
limited options.

---

## 7. Analysing Performance

### From the terminal output

During evaluation, the worker prints per-episode stats:

```
Episode 1 | return: 67.4 | length: 1200 | status: finished
Episode 2 | return: 71.2 | length: 1150 | status: finished
Episode 3 | return: 12.3 | length: 234  | status: crashed
```

### Key indicators

| Indicator              | Good sign                  | Bad sign                    |
|------------------------|----------------------------|-----------------------------|
| Return                 | Increasing over episodes   | Random or decreasing        |
| Episode length         | Close to `ep_max_length`   | Very short (early crash)    |
| Completion rate        | >80% episodes finish       | <20% finish                 |
| Return variance        | Low (consistent)           | High (unpredictable)        |

### When to retrain

| Observation                        | Action                                    |
|------------------------------------|-------------------------------------------|
| Never completes the track          | Train longer, or check reward trajectory  |
| Completes but slowly               | Increase `velocity_weight`                |
| Fast but crashes often             | Increase `collision_penalty`              |
| Good on straights, bad on corners  | More training, or increase `img_hist_len` |
| Inconsistent (good then bad)       | Lower `alpha`, more training steps        |
