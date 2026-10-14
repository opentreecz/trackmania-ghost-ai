# Trackmania Ghost AI

Autonomous AI agent for **TrackMania Nations / TM2020** circuits, powered by
reinforcement learning. The agent uses LIDAR-based observations (no images)
processed through a multi-layer perceptron (MLP) and is trained with
**Soft Actor-Critic (SAC)** via the [tmrl](https://github.com/trackmania-rl/tmrl)
framework.

## Architecture

```
+--------------------+                +----------------+                +--------------------+
|   RolloutWorker    |   samples      |     Server     |   weights      |      Trainer       |
|   (data collect)   |  ------------> |     (relay)    | ------------->  |   (SAC gradient)   |
|                    |                |                |                |                    |
|   TM20 + LIDAR    |   weights      |    TCP/TLS     |   samples      |   MLP ActorCritic  |
|   rtgym env        |  <------------ |                | <-------------  |   ReplayBuffer     |
+--------------------+                +----------------+                +--------------------+
```

### Observation space (LIDAR, no images)

| Component        | Shape    | Description                          |
|------------------|----------|--------------------------------------|
| Speed            | `(1,)`   | Normalised scalar speed              |
| LIDAR history    | `(4,19)` | 4 consecutive frames of 19 beams     |
| Previous action 1| `(3,)`   | `[gas, brake, steer]` from t-2       |
| Previous action 2| `(3,)`   | `[gas, brake, steer]` from t-1       |

**Total flat dimension: 83**

### Action space

Continuous `[gas, brake, steer]` in `[-1, 1]`.

## Project Structure

```
trackmania-ghost-ai/
├── configs/
│   └── training.yaml          # All hyperparameters and settings
├── scripts/
│   ├── train.sh               # Launch training (all-in-one)
│   └── setup_tmrl.sh          # First-time tmrl/TmrlData setup
├── src/
│   └── trackmania_ghost_ai/
│       ├── __init__.py
│       ├── agent.py           # SAC training agent construction
│       ├── cli.py             # Console entry points
│       ├── config.py          # YAML config loader + dataclasses
│       ├── environment.py     # Environment wrapper and reward shaping
│       ├── memory.py          # Replay buffer setup
│       ├── models.py          # MLP Actor, Critic, ActorCritic
│       └── pipeline.py        # Orchestrator (Server/Worker/Trainer)
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   └── test_models.py
├── LICENSE                    # AGPL-3.0
├── pyproject.toml
└── README.md
```

## Prerequisites

- **Python** >= 3.10
- **TrackMania Nations Forever** or **Trackmania 2020** installed
- **OpenPlanet** plugin (for TM2020) with the tmrl plugin
- Windows (the game + tmrl interface require Windows; training can be remote)

## Installation

```bash
# Clone the repository
git clone https://github.com/opentreecz/trackmania-ghost-ai.git
cd trackmania-ghost-ai

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### First-time tmrl setup

tmrl stores its runtime data in `~/TmrlData/`. Run the setup helper:

```bash
bash scripts/setup_tmrl.sh
```

Or manually:

```bash
python -c "import tmrl; tmrl.init()"
```

## Usage

### Train (all-in-one, single machine)

```bash
tm-train --config configs/training.yaml
```

This starts the **Server**, **RolloutWorker**, and **Trainer** in a single
process using threads.

### Distributed training (multi-machine)

On the **server** machine:
```bash
tm-server --config configs/training.yaml
```

On the **worker** machine (with the game running):
```bash
tm-worker --config configs/training.yaml
```

On the **trainer** machine (GPU recommended):
```bash
tm-train --config configs/training.yaml
```

### Evaluate a trained model

```bash
tm-evaluate --weights checkpoints/ghost-ai-default_epoch0200.pt --episodes 10
```

## Configuration

All hyperparameters live in `configs/training.yaml`. Key sections:

| Section       | What it controls                            |
|---------------|---------------------------------------------|
| `run`         | Run name, seed, device, W&B logging         |
| `environment` | LIDAR interface, window size, rtgym timing  |
| `algorithm`   | SAC hyperparameters (gamma, polyak, alpha)   |
| `memory`      | Replay buffer capacity and batch size        |
| `training`    | Epochs, rounds, gradient steps               |
| `networking`  | Server IP/port for distributed training      |
| `checkpoint`  | Save directory, interval, retention          |

## Tutorials

Comprehensive guides in [`docs/tutorials/`](docs/tutorials/):

| # | Tutorial | Description |
|---|---------|-------------|
| 01 | [Defining Driving Capabilities](docs/tutorials/01-defining-driving-capabilities.md) | Reward shaping, YAML recipes, custom rewards |
| 02 | [Writing Driving Parameters](docs/tutorials/02-writing-driving-parameters.md) | Full parameter reference, network architecture, tuning |
| 03 | [Prerequisites & Installation](docs/tutorials/03-prerequisites-installation.md) | System setup, TM2020, OpenPlanet, TmrlData |
| 04 | [First Training Run](docs/tutorials/04-first-training-run.md) | Step-by-step from zero to a driving agent |
| 05 | [Saving & Loading Models](docs/tutorials/05-saving-loading-models.md) | Weights, checkpoints, resume, export |
| 06 | [Training Workflow](docs/tutorials/06-training-workflow.md) | Epochs/rounds/steps, duration, troubleshooting |
| 07 | [Replay & Evaluation](docs/tutorials/07-replay-evaluation.md) | Test episodes, watching AI drive, ghost replays |
| 08 | [Transfer Learning](docs/tutorials/08-transfer-learning.md) | Pre-trained weights, fine-tuning, curriculum |
| 09 | [Platform: Windows](docs/tutorials/09-platform-windows.md) | Native setup, WSL2 hybrid |
| 10 | [Platform: Linux](docs/tutorials/10-platform-linux.md) | Proton, headless trainer, Docker |
| 11 | [Platform: Console](docs/tutorials/11-platform-console.md) | PS/Xbox limitations, hardware approaches |
| 12 | [Headless & Distributed](docs/tutorials/12-headless-distributed.md) | Multi-machine, Docker, systemd |
| 13 | [Recording Videos](docs/tutorials/13-recording-videos.md) | Replay Editor, OBS, ffmpeg |
| 14 | [Advanced Topics](docs/tutorials/14-advanced-topics.md) | Custom networks, CNN, REDQ, custom rewards |

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## License

This project is licensed under the **GNU Affero General Public License v3.0**.
See [LICENSE](LICENSE) for details.
