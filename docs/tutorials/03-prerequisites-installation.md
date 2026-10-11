# Prerequisites and Installation

Complete setup guide for getting Trackmania Ghost AI running on your machine.

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Install TrackMania 2020](#2-install-trackmania-2020)
3. [Install OpenPlanet](#3-install-openplanet)
4. [Install Python and the Project](#4-install-python-and-the-project)
5. [Initialise tmrl](#5-initialise-tmrl)
6. [TmrlData Directory Structure](#6-tmrldata-directory-structure)
7. [Install the Track Maps](#7-install-the-track-maps)
8. [Verify the Setup](#8-verify-the-setup)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. System Requirements

### Minimum (worker machine -- runs the game)

| Component | Requirement                              |
|-----------|------------------------------------------|
| OS        | Windows 10/11 (native) or Linux (Proton) |
| CPU       | 4 cores, 3.0 GHz+                        |
| RAM       | 8 GB                                     |
| GPU       | GTX 660 / equivalent (DirectX 11)        |
| Disk      | 10 GB free (game + TmrlData)             |
| Display   | Monitor or virtual display (Xvfb)        |

### Recommended (trainer machine -- runs gradient updates)

| Component | Requirement                          |
|-----------|--------------------------------------|
| OS        | Linux or Windows                     |
| GPU       | NVIDIA RTX 3060+ with CUDA           |
| RAM       | 16 GB (replay buffer lives in RAM)   |
| VRAM      | 4 GB+                                |

The trainer does **not** need the game installed. In a distributed setup,
the worker (Windows, has the game) and trainer (Linux GPU server) can be
separate machines.

---

## 2. Install TrackMania 2020

### Windows (recommended)

1. Install from [Steam](https://store.steampowered.com/app/2225070/Trackmania/),
   [Epic Games Store](https://store.epicgames.com/), or
   [Ubisoft Store](https://store.ubisoft.com/)
2. Launch once and complete the initial setup
3. The free Starter Access tier is sufficient for training

### Linux (experimental)

1. Install Steam and enable **Proton Experimental** for all titles:
   - Steam > Settings > Compatibility > Enable Steam Play for all titles
2. Install Trackmania via Steam
3. Install `protontricks`:
   ```bash
   pip install protontricks
   # or: sudo apt install protontricks
   ```
4. Install the Ubisoft launcher inside the Proton prefix:
   ```bash
   protontricks 2225070 uplay
   ```
5. Launch the game once through Steam to complete setup

See the [Platform Guide - Linux](./platform-linux.md) tutorial for the full
Linux setup including OpenPlanet under Proton.

---

## 3. Install OpenPlanet

OpenPlanet is a modding framework for TM2020. The tmrl plugin uses it to
extract game data (speed, position, screenshots).

### Windows

1. Install [Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Download the installer from [openplanet.dev](https://openplanet.dev/)
3. Run the installer -- it auto-detects TM2020's location
4. Launch TM2020. Press **F3** to verify OpenPlanet is active

### Linux (Proton)

```bash
# Add DLL override to Steam launch options:
# Right-click Trackmania > Properties > Launch Options:
WINEDLLOVERRIDES="dinput8=n,b" %command%

# Install VC++ runtime
protontricks 2225070 vcrun2022

# Download and install OpenPlanet via protontricks
protontricks-launch --appid 2225070 OpenplanetInstaller.exe
```

### Install the tmrl plugin

After `python -m tmrl --install` (Step 5), the plugin file
`TMRL_GrabData.op` is placed at `~/TmrlData/resources/Plugins/`.

Copy it to the OpenPlanet plugins directory:

- **Windows**: `%USERPROFILE%\OpenplanetNext\Plugins\`
- **Linux**: `~/.steam/steam/steamapps/compatdata/2225070/pfx/drive_c/users/steamuser/OpenplanetNext/Plugins/`

Verify in-game: Press **F3** > Developer > (Re)load plugin > TMRL Grab Data.
The log should say *"waiting for incoming connection"*.

---

## 4. Install Python and the Project

### Python

Install Python 3.10 or later:

- **Windows**: [python.org](https://www.python.org/downloads/) (check "Add to PATH")
- **Linux**: `sudo apt install python3 python3-pip python3-venv`

### Clone and install

```bash
git clone https://github.com/opentreecz/trackmania-ghost-ai.git
cd trackmania-ghost-ai

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Install with all dependencies
pip install -e ".[dev]"
```

This installs:
- `tmrl` (the Trackmania RL framework)
- `torch` (PyTorch for neural networks)
- `numpy`, `gymnasium`, `rtgym`
- `wandb` (optional logging)
- `pytest`, `ruff`, `mypy` (dev tools)

And registers the CLI commands: `tm-train`, `tm-server`, `tm-worker`,
`tm-evaluate`.

### GPU support (CUDA)

If you have an NVIDIA GPU, install the CUDA version of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

## 5. Initialise tmrl

```bash
# Method 1: use the setup script
bash scripts/setup_tmrl.sh

# Method 2: manual
python -m tmrl --install
```

This creates the `~/TmrlData/` directory tree, copies default configs,
maps, and the OpenPlanet plugin.

On Windows, it also installs the **ViGEmBus** virtual gamepad driver
(required for sending controller inputs to the game).

---

## 6. TmrlData Directory Structure

After initialisation:

```
~/TmrlData/
├── checkpoints/                # Training checkpoints (.tcpt)
│   └── {RUN_NAME}_t.tcpt      # Full training state (model + buffer + epoch)
├── config/
│   └── config.json             # tmrl's native config (we override via YAML)
├── dataset/
│   └── data.pkl                # Optional offline replay buffer
├── resources/
│   ├── Plugins/
│   │   └── TMRL_GrabData.op   # OpenPlanet plugin (copy to OpenplanetNext/)
│   ├── tmrl-test.Map.Gbx      # Pre-made test track
│   └── tmrl-train.Map.Gbx     # Pre-made training track
├── reward/
│   └── reward.pkl              # Recorded trajectory for reward function
└── weights/
    ├── {RUN_NAME}.tmod         # Worker's current policy weights
    ├── {RUN_NAME}_t.tmod       # Trainer's current policy weights
    └── SAC_4_LIDAR_pretrained.tmod  # Pre-trained LIDAR weights (from install)
```

---

## 7. Install the Track Maps

Copy the tmrl practice maps to your TM2020 maps folder:

### Windows

```powershell
copy "%USERPROFILE%\TmrlData\resources\tmrl-test.Map.Gbx" ^
     "%USERPROFILE%\Documents\Trackmania\Maps\My Maps\"
copy "%USERPROFILE%\TmrlData\resources\tmrl-train.Map.Gbx" ^
     "%USERPROFILE%\Documents\Trackmania\Maps\My Maps\"
```

### Linux

```bash
MAPS_DIR=~/.steam/steam/steamapps/compatdata/2225070/pfx/drive_c/users/steamuser/Documents/Trackmania/Maps/My\ Maps
mkdir -p "$MAPS_DIR"
cp ~/TmrlData/resources/tmrl-*.Map.Gbx "$MAPS_DIR/"
```

### Track requirements for LIDAR

The LIDAR interface (`TM20LIDAR`) requires tracks with **plain road and
black borders**. The pre-installed `tmrl-test` and `tmrl-train` maps meet
this requirement. Custom tracks must also have black-bordered roads for the
LIDAR ray-casting to detect wall distances correctly.

---

## 8. Verify the Setup

### Check Python packages

```bash
python -c "
import tmrl; print('tmrl:', tmrl.__version__)
import torch; print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
import trackmania_ghost_ai; print('project:', trackmania_ghost_ai.__version__)
"
```

### Check the CLI

```bash
tm-train --help
tm-evaluate --help
```

### Check the game connection

1. Launch TM2020
2. Load `tmrl-test` map (Play > Local > My Maps > tmrl-test)
3. Switch to **cockpit view** (press `3`)
4. Hide the car (press `Alt+C` or use OpenPlanet settings)
5. Position the window at the **top-left corner** of your screen
6. Set to **windowed mode** (958 x 488 pixels)
7. Press **F3** and verify the TMRL plugin shows "waiting for connection"

### Run a quick environment check

```bash
python -m tmrl --check-environment
```

This launches a worker that connects to the game, captures a screenshot,
computes LIDAR readings, and displays the result. Drive the car manually and
verify the LIDAR visualisation makes sense.

---

## 9. Troubleshooting

| Problem                                  | Solution                                         |
|------------------------------------------|--------------------------------------------------|
| `ModuleNotFoundError: tmrl`              | Activate your venv: `source .venv/bin/activate`  |
| `No module named 'vgamepad'`             | Windows: reinstall tmrl. Linux: `pip install vgamepad` + uinput access |
| OpenPlanet not loading                   | Reinstall after TM2020 updates                   |
| "Connection refused" from worker         | Reload the TMRL plugin (F3 menu in game)         |
| LIDAR shows all zeros                    | Wrong track (needs black borders) or wrong camera|
| `DLL load failed: win32gui`              | Install pywin32 via conda: `conda install pywin32` |
| Window not detected                      | Must be windowed mode at top-left corner         |
| "Timestep timeout" warnings              | GPU too slow; reduce TM graphics quality or set max FPS to 30 |
| `torch.cuda.is_available()` returns False| Install CUDA toolkit + PyTorch CUDA build        |
| ViGEmBus install failed (Windows)        | Download manually: [github.com/ViGEm/ViGEmBus](https://github.com/ViGEm/ViGEmBus/releases) |
