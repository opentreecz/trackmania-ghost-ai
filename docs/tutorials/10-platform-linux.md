# Platform Guide: Linux

Linux support covers two scenarios: running the full pipeline (game + worker +
trainer) via Steam Proton, or running only the server + trainer on a headless
GPU server while the worker runs on a Windows machine.

---

## Scenario A: Full Pipeline on Linux (Proton)

Experimental since tmrl v0.6.0. The game runs via Steam Proton/Wine.

### 1. Install Steam and Trackmania

```bash
# Install Steam
sudo apt install steam

# Enable Proton for all titles:
# Steam > Settings > Compatibility > Enable Steam Play for all titles
# Select "Proton Experimental"

# Install Trackmania via Steam (app ID 2225070)
```

### 2. Install Ubisoft launcher

```bash
pip install protontricks
protontricks 2225070 uplay
```

### 3. Install OpenPlanet

```bash
# Install VC++ runtime in Proton prefix
protontricks 2225070 vcrun2022

# Download OpenPlanet installer from https://openplanet.dev/
# Install into the Proton prefix
protontricks-launch --appid 2225070 OpenplanetInstaller.exe
```

Add DLL override to Steam launch options:
- Right-click Trackmania > Properties > Launch Options:
```
WINEDLLOVERRIDES="dinput8=n,b" %command%
```

### 4. Install Python and the project

```bash
sudo apt install python3 python3-pip python3-venv xdotool

git clone https://github.com/opentreecz/trackmania-ghost-ai.git
cd trackmania-ghost-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 5. Initialise tmrl

```bash
python -m tmrl --install
```

### 6. Set up virtual gamepad access

```bash
# Grant access to /dev/uinput (needed by vgamepad on Linux)
sudo chmod +0666 /dev/uinput

# Make it persistent with a udev rule
echo 'KERNEL=="uinput", MODE="0666"' | sudo tee /etc/udev/rules.d/99-uinput.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 7. Copy the plugin and maps

```bash
# OpenPlanet plugins directory (inside Proton prefix)
PROTON_USER=~/.steam/steam/steamapps/compatdata/2225070/pfx/drive_c/users/steamuser
cp ~/TmrlData/resources/Plugins/TMRL_GrabData.op \
   "$PROTON_USER/OpenplanetNext/Plugins/"

# Maps
MAPS_DIR="$PROTON_USER/Documents/Trackmania/Maps/My Maps"
mkdir -p "$MAPS_DIR"
cp ~/TmrlData/resources/tmrl-*.Map.Gbx "$MAPS_DIR/"
```

### 8. Launch and train

```bash
# Launch TM2020 via Steam
# Load tmrl-test map, windowed mode, top-left, cockpit view

# In another terminal
source .venv/bin/activate
tm-train --config configs/training.yaml --verbose
```

### Known issues on Linux

- **OpenPlanet breaks after TM2020 auto-updates** -- reinstall OpenPlanet each time
- **vgamepad button mappings** may differ from Windows (`libevdev` vs ViGEmBus)
- **xdotool** is required for window positioning -- install it
- Game performance may be lower under Proton than native Windows

---

## Scenario B: Headless Trainer/Server on Linux (No Game)

The server and trainer are pure Python -- they do **not** need TM2020,
OpenPlanet, vgamepad, or a display. This is the recommended setup for GPU
training servers.

### Requirements

| Component      | Needed |
|----------------|--------|
| TM2020         | No     |
| OpenPlanet     | No     |
| vgamepad       | No     |
| Display/Xvfb   | No     |
| NVIDIA GPU     | Recommended |
| Python 3.10+   | Yes    |
| PyTorch + CUDA | Yes    |

### Setup

```bash
git clone https://github.com/opentreecz/trackmania-ghost-ai.git
cd trackmania-ghost-ai
python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Run server + trainer

```bash
# Configure networking for cross-machine
# Edit configs/training.yaml:
#   networking:
#     server_ip: "0.0.0.0"     # listen on all interfaces
#     local_mode: false

tm-server --config configs/training.yaml &
tm-train --config configs/training.yaml --verbose
```

The worker runs on a separate Windows machine (see
[Headless & Distributed Training](./12-headless-distributed.md)).

### Docker

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["tm-train", "--config", "configs/training.yaml"]
```

```bash
docker build -t trackmania-trainer .
docker run --gpus all trackmania-trainer
```

---

## Scenario C: Virtual Display with Xvfb (Worker Without Monitor)

For running the game on a Linux machine without a physical display (e.g.,
a cloud VM with a GPU).

```bash
# Install Xvfb
sudo apt install xvfb

# Start a virtual display
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# Launch TM2020 via Steam on the virtual display
steam -applaunch 2225070 &

# xdotool can still control the window
xdotool search --name "Trackmania" windowactivate

# Start the worker
tm-worker --config configs/training.yaml
```

**Caveats**:
- Performance depends on GPU passthrough (needs a real GPU for DirectX)
- `mss` (screenshot library) works with Xvfb
- No visual feedback -- monitor training via logs or W&B

---

## Troubleshooting (Linux-Specific)

| Problem                              | Solution                                     |
|--------------------------------------|----------------------------------------------|
| `vgamepad` permission denied         | `sudo chmod +0666 /dev/uinput` + udev rule   |
| OpenPlanet not loading (Proton)      | Add `WINEDLLOVERRIDES="dinput8=n,b"` to launch options |
| `xdotool` not found                  | `sudo apt install xdotool`                   |
| Game crashes under Proton            | Try `Proton Experimental` instead of stable   |
| `protontricks` fails                 | `pip install --upgrade protontricks`          |
| Black screen on launch               | Enable Proton Experimental in Steam settings  |
| `CUDA not available` on server       | Install NVIDIA driver + CUDA toolkit          |
