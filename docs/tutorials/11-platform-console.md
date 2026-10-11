# Platform Guide: PlayStation, Xbox, and Console

Honest assessment of console platforms for Trackmania AI training.

---

## TL;DR

**Console platforms are not practically viable for RL training.** Use the
PC version (Windows or Linux). The rest of this document explains why and
what the theoretical approaches are.

---

## Console Availability

TrackMania 2020 is available on:

| Platform           | Available | AI training viable |
|--------------------|-----------|--------------------|
| **Windows PC**     | Yes       | Yes (recommended)  |
| **Linux PC**       | Via Proton| Yes (experimental) |
| **PlayStation**    | Yes       | No                 |
| **Xbox**           | Yes       | No                 |
| **Amazon Luna**    | Yes       | No                 |
| **Nintendo Switch**| No        | N/A                |

---

## Why Consoles Don't Work

The tmrl framework requires three things that consoles cannot provide:

### 1. Screenshot capture from the game process

tmrl uses the `mss` library to capture screenshots directly from the screen.
Consoles have no API for programmatic screen capture from external software.

### 2. Programmatic controller input injection

tmrl uses `vgamepad` (ViGEmBus on Windows, libevdev on Linux) to send
virtual controller inputs. Console USB stacks require authenticated
controllers -- random USB devices are rejected.

### 3. OpenPlanet plugin for game data

OpenPlanet extracts speed, position, and other telemetry from TM2020's
memory. There is no modding framework on consoles.

---

## Theoretical Approach: Hardware-in-the-Loop

If you absolutely must use a console, the theoretical approach is:

```
Console (PS5/Xbox)
    |
    | HDMI output
    v
Capture Card (Elgato, AverMedia)
    |
    | Video frames to PC
    v
PC: Computer Vision (LIDAR simulation from frames)
    |
    | Computed actions
    v
Input Device (Titan Two / CronusZen)
    |
    | Authenticated controller signals
    v
Console USB port
```

### Components needed

| Component              | Cost       | Purpose                              |
|------------------------|------------|--------------------------------------|
| HDMI capture card      | $100-200   | Capture game video to PC             |
| Titan Two / CronusZen | $80-100    | Inject controller inputs to console  |
| PC with GPU            | $500+      | Process frames, run neural network   |

### Titan Two / CronusZen

These devices sit between a controller and the console. They:
- Handle controller authentication transparently
- Accept commands from a PC via USB serial
- Support PS4, PS5, Xbox One, Xbox Series, Switch

Serial API example (pseudocode):
```python
import serial

device = serial.Serial("/dev/ttyUSB0", 115200)

# Send controller state: gas=0.8, brake=0.0, steer=0.3
device.write(b"GAS:0.8;BRAKE:0.0;STEER:0.3\n")
```

### Latency problem

The fundamental issue is latency:

| Step                       | Latency      |
|----------------------------|--------------|
| HDMI capture               | 5-15 ms      |
| Frame transfer to PC       | 5-10 ms      |
| Neural network inference   | 5-10 ms      |
| Serial command to Titan Two| 2-5 ms       |
| USB to console             | 1-4 ms       |
| **Total round-trip**       | **18-44 ms** |

tmrl operates at 20 Hz (50 ms per step). With 18-44 ms of hardware latency
eating most of that budget, the agent has almost no time margin. At 10 Hz
(100 ms steps) it becomes more feasible but control precision suffers.

### Has anyone done this?

- A few research groups have used capture card + input injection for
  retro console RL (NES, SNES via RetroArch)
- No known successful implementation for modern console games at the
  precision needed for racing
- The established approach for game RL is always: use the PC version

---

## Recommended Alternative: Xbox Game Pass PC

If you have an Xbox but not a Windows PC license for TM2020:

1. TM2020 is available on PC through Ubisoft Connect (free Starter Access)
2. You only need a Windows PC -- no Xbox subscription required
3. The free tier gives you access to all tracks for training

---

## PS Remote Play (Does Not Work)

Sony's PS Remote Play streams the PS5 display to a PC and sends controller
inputs back. Theoretically you could:

1. Capture the Remote Play video stream
2. Process with computer vision
3. Send inputs back through the Remote Play protocol

**Why it doesn't work:**
- The protocol is proprietary and encrypted
- Latency is 50-100+ ms (too slow for 20 Hz control)
- Resolution is limited and compressed
- No public API for programmatic control
- Community projects (chiaki) are reverse-engineered and fragile

---

## Summary

| Approach              | Feasibility | Cost     | Latency   | Recommendation     |
|-----------------------|-------------|----------|-----------|--------------------|
| PC (Windows native)   | Works       | $0 extra | <5 ms     | **Use this**       |
| PC (Linux Proton)     | Works       | $0 extra | <10 ms    | Good alternative   |
| Console + capture card| Theoretical | $200-400 | 20-50 ms  | Not recommended    |
| Console + Remote Play | No          | $0       | 50-100 ms | Does not work      |

**Use the PC version.** It is free, has full API support via OpenPlanet,
and is the only platform where tmrl is tested and maintained.
