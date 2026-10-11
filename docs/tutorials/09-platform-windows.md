# Platform Guide: Windows

Native Windows setup for all three components: worker, server, and trainer.

---

## Full Setup (Worker + Trainer on One Machine)

### 1. Install TrackMania 2020

Install from Steam, Epic, or Ubisoft Store. Free Starter Access is sufficient.

### 2. Install OpenPlanet

1. Install [Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Download from [openplanet.dev](https://openplanet.dev/) and run the installer
3. Launch TM2020, press **F3** to confirm OpenPlanet loads

### 3. Install Python and the project

```powershell
# Install Python 3.10+ from python.org (check "Add to PATH")

git clone https://github.com/opentreecz/trackmania-ghost-ai.git
cd trackmania-ghost-ai

python -m venv .venv
.venv\Scripts\activate

pip install -e ".[dev]"
```

### 4. Initialise tmrl

```powershell
python -m tmrl --install
```

This creates `%USERPROFILE%\TmrlData\` and installs the **ViGEmBus** virtual
gamepad driver (accept the driver installation prompt).

### 5. Copy the tmrl plugin and maps

```powershell
copy "%USERPROFILE%\TmrlData\resources\Plugins\TMRL_GrabData.op" ^
     "%USERPROFILE%\OpenplanetNext\Plugins\"

copy "%USERPROFILE%\TmrlData\resources\tmrl-test.Map.Gbx" ^
     "%USERPROFILE%\Documents\Trackmania\Maps\My Maps\"
copy "%USERPROFILE%\TmrlData\resources\tmrl-train.Map.Gbx" ^
     "%USERPROFILE%\Documents\Trackmania\Maps\My Maps\"
```

### 6. Configure the game window

1. Launch TM2020 and load the `tmrl-test` map
2. Settings > Graphics > Display > **Windowed**
3. Resize to **958 x 488** pixels
4. Position at the **top-left corner** of your screen
5. Press `3` for cockpit view
6. Press `Alt+C` to hide the car
7. Press **F3** > Developer > reload TMRL Grab Data plugin

### 7. Train

```powershell
tm-train --config configs\training.yaml --verbose
```

Click the TM2020 window immediately after starting.

---

## WSL2 Hybrid Setup (Worker on Windows, Trainer on WSL2)

This gives you Linux GPU training while the game runs natively on Windows.

### Windows side (worker)

Follow steps 1-6 above. Then run only the worker:

```powershell
tm-worker --config configs\training.yaml
```

### WSL2 side (server + trainer)

```bash
# In WSL2 terminal
cd /mnt/c/Users/YourName/git/trackmania-ghost-ai

# Install CUDA toolkit in WSL2 (if NVIDIA GPU)
# Follow: https://docs.nvidia.com/cuda/wsl-user-guide/

pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Run server + trainer
tm-server --config configs/training.yaml &
tm-train --config configs/training.yaml
```

Network: WSL2 and Windows share `localhost`, so the default
`server_ip: "127.0.0.1"` works without changes.

---

## Troubleshooting (Windows-Specific)

| Problem                          | Solution                                          |
|----------------------------------|---------------------------------------------------|
| ViGEmBus install fails           | Download manually from [github.com/ViGEm/ViGEmBus/releases](https://github.com/ViGEm/ViGEmBus/releases) |
| `DLL load failed: win32gui`      | `conda install pywin32` or `pip install pywin32`  |
| OpenPlanet missing after update  | Reinstall OpenPlanet (TM2020 updates break it)    |
| Window not at top-left           | Drag to corner, or use `Win+Left` then `Win+Up`  |
| Game window loses focus          | Click TM2020 window after starting the worker     |
| Antivirus blocks vgamepad        | Add an exception for the ViGEmBus driver          |
