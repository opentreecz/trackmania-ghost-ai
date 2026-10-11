# Saving and Loading Models

How model weights, checkpoints, and training state are saved, loaded, and
managed across training sessions.

## Table of Contents

1. [What Gets Saved](#1-what-gets-saved)
2. [Automatic Saving During Training](#2-automatic-saving-during-training)
3. [File Locations](#3-file-locations)
4. [Manual Save and Load](#4-manual-save-and-load)
5. [Checkpoint vs Weights](#5-checkpoint-vs-weights)
6. [Resuming Training](#6-resuming-training)
7. [Managing Multiple Models](#7-managing-multiple-models)
8. [Exporting for Deployment](#8-exporting-for-deployment)

---

## 1. What Gets Saved

There are two types of saved state:

### Policy weights (`.tmod` files)

The neural network parameters -- the actor's learned policy. Small files
(~1-3 MB). This is what the worker loads to drive the car.

```
~/TmrlData/weights/ghost-ai-default.tmod       # worker weights
~/TmrlData/weights/ghost-ai-default_t.tmod     # trainer weights
```

Format: Standard PyTorch `state_dict` saved with `torch.save()`.

### Training checkpoints (`.tcpt` files)

The full training state: model weights + optimiser state + replay buffer +
epoch counter. Large files (100 MB - 2 GB depending on buffer size).

```
~/TmrlData/checkpoints/ghost-ai-default_t.tcpt
```

Format: Python pickle of the entire `TrainingOffline` instance.

---

## 2. Automatic Saving During Training

tmrl saves automatically at multiple points:

### Weights broadcast (every `update_model_interval` steps)

Every 100 gradient steps (default), the trainer:
1. Saves the actor weights to `~/TmrlData/weights/{RUN_NAME}_t.tmod`
2. Broadcasts the weights to all workers via the server
3. Workers save to `~/TmrlData/weights/{RUN_NAME}.tmod`

```yaml
training:
  update_model_interval: 100   # broadcast every 100 gradient steps
```

### Checkpoint save (every epoch)

After each epoch, the trainer pickles the entire `TrainingOffline` instance
(model + optimiser + replay buffer + epoch counter) to:

```
~/TmrlData/checkpoints/{RUN_NAME}_t.tcpt
```

This is an atomic write (temp file + rename) with signal protection to
prevent corruption from `Ctrl+C`.

### Weight history (timestamped snapshots)

The worker optionally saves timestamped copies:

```
~/TmrlData/weights/{RUN_NAME}_11_10_2026_14_30_00.tmod
```

These accumulate over time, letting you go back to any point in training.

---

## 3. File Locations

| File                              | Location                               | Size     | Contains              |
|-----------------------------------|----------------------------------------|----------|-----------------------|
| Worker weights                    | `~/TmrlData/weights/{NAME}.tmod`       | 1-3 MB   | Actor state_dict      |
| Trainer weights                   | `~/TmrlData/weights/{NAME}_t.tmod`     | 1-3 MB   | Actor state_dict      |
| Weight history                    | `~/TmrlData/weights/{NAME}_{ts}.tmod`  | 1-3 MB   | Actor state_dict      |
| Training checkpoint               | `~/TmrlData/checkpoints/{NAME}_t.tcpt` | 100MB-2GB| Full training state   |
| Pre-trained LIDAR weights         | `~/TmrlData/weights/SAC_4_LIDAR_pretrained.tmod` | ~1 MB | Pre-trained policy |
| Replay buffer (offline)           | `~/TmrlData/dataset/data.pkl`          | varies   | Collected transitions |

The `{NAME}` is your `run.name` from `configs/training.yaml`.

---

## 4. Manual Save and Load

### Save weights from Python

```python
from tmrl.actor import TorchActorModule

# Assuming you have an actor instance
actor.save("path/to/my_weights.tmod")
```

### Load weights from Python

```python
# Load into an existing actor
actor = actor.load("path/to/my_weights.tmod", device="cpu")
```

### Save a full checkpoint from Python

```python
import torch

state = {
    "epoch": 42,
    "model_state_dict": actor.state_dict(),
    "config": cfg,
}
torch.save(state, "checkpoints/my_checkpoint.pt")
```

### Load a checkpoint

```python
import torch

checkpoint = torch.load("checkpoints/my_checkpoint.pt", map_location="cpu")
actor.load_state_dict(checkpoint["model_state_dict"])
epoch = checkpoint["epoch"]
```

### Using the project's save_checkpoint utility

```python
from trackmania_ghost_ai.pipeline import save_checkpoint

# Saves to configs/checkpoint.dir with auto-pruning
save_checkpoint(trainer, cfg, epoch=42)
```

This saves to `checkpoints/{run_name}_epoch0042.pt` and prunes old
checkpoints to keep only the last `checkpoint.keep_last` files.

---

## 5. Checkpoint vs Weights

| Aspect          | Weights (`.tmod`)          | Checkpoint (`.tcpt`)             |
|-----------------|----------------------------|----------------------------------|
| Contains        | Actor neural network only  | Actor + critic + optimiser + buffer + epoch |
| Size            | 1-3 MB                     | 100 MB - 2 GB                    |
| Use case        | Evaluation, deployment     | Resume training exactly           |
| Portable        | Yes (any machine)          | Sensitive to Python/torch version |
| Resume training | Partially (loses optimiser)| Fully (exact state)               |

### When to use which

- **"I want to watch the AI drive"** -- load weights (`.tmod`)
- **"I want to continue training from where I left off"** -- use checkpoint (`.tcpt`)
- **"I want to fine-tune on a different track"** -- load weights into a fresh training run
- **"I want to share my trained model"** -- share the `.tmod` file

---

## 6. Resuming Training

### Automatic resume

tmrl resumes automatically. When you start training, it checks for:

1. `~/TmrlData/checkpoints/{RUN_NAME}_t.tcpt` -- if exists, loads the full
   training state (model + buffer + epoch counter)
2. `~/TmrlData/weights/{RUN_NAME}.tmod` -- the worker loads existing weights
   on startup

So simply running `tm-train` with the same `run.name` resumes from where
you stopped.

### Fresh start with same name

To force a fresh start (discard previous training), either:

- Delete the checkpoint: `rm ~/TmrlData/checkpoints/{RUN_NAME}_t.tcpt`
- Or change `run.name` in the config

### Resume with reset optimiser

Set `RESET_TRAINING: true` in `~/TmrlData/config/config.json`. This creates
a fresh model and optimiser but **keeps the replay buffer**, giving you a
new network that immediately trains on previously collected experience.

---

## 7. Managing Multiple Models

### Use different run names

```yaml
# configs/track_a.yaml
run:
  name: "track-a-speed"

# configs/track_b.yaml
run:
  name: "track-b-cautious"
```

Each gets its own weights and checkpoints:

```
~/TmrlData/weights/track-a-speed.tmod
~/TmrlData/weights/track-a-speed_t.tmod
~/TmrlData/weights/track-b-cautious.tmod
~/TmrlData/weights/track-b-cautious_t.tmod
```

### Backup important models

```bash
# Create a named backup
cp ~/TmrlData/weights/my-run.tmod ~/TmrlData/weights/my-run_best_v1.tmod

# Or use the project's checkpoint directory
mkdir -p checkpoints/archive
cp ~/TmrlData/weights/my-run.tmod checkpoints/archive/my-run_epoch200.tmod
```

### List all saved models

```bash
ls -lh ~/TmrlData/weights/*.tmod
ls -lh ~/TmrlData/checkpoints/*.tcpt
```

---

## 8. Exporting for Deployment

### Export weights for sharing

The `.tmod` file is self-contained and portable:

```bash
cp ~/TmrlData/weights/my-run.tmod ./exported_model.tmod
```

Anyone with the same project can load it:

```bash
cp exported_model.tmod ~/TmrlData/weights/shared-model.tmod
# Set run.name to "shared-model" in config
tm-evaluate --episodes 5
```

### Using pre-trained weights

tmrl ships with pre-trained weights:

```bash
# Use the pre-trained LIDAR model
cp ~/TmrlData/weights/SAC_4_LIDAR_pretrained.tmod \
   ~/TmrlData/weights/ghost-ai-default.tmod

# Evaluate it
tm-evaluate --episodes 5
```

### Weight file format details

The `.tmod` file is a standard `torch.save(state_dict)` output:

```python
import torch

# Inspect a weight file
state = torch.load("model.tmod", map_location="cpu")
for key, tensor in state.items():
    print(f"{key}: {tensor.shape}")
```

Typical output for a `[256, 256]` MLP:

```
net.0.weight: torch.Size([256, 83])    # first hidden layer
net.0.bias: torch.Size([256])
net.2.weight: torch.Size([256, 256])   # second hidden layer
net.2.bias: torch.Size([256])
mu_layer.weight: torch.Size([3, 256])  # action mean
mu_layer.bias: torch.Size([3])
log_std_layer.weight: torch.Size([3, 256])  # action std
log_std_layer.bias: torch.Size([3])
```
