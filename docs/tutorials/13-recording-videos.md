# Recording Race Videos

How to capture the AI's driving as video files using the game's replay
system, screen recording, or command-line tools.

---

## Method 1: TrackMania Replay Editor (Best Quality)

TM2020 has a built-in replay system. Every completed race saves a
`.Replay.Gbx` file containing the exact inputs -- not video frames, but
a deterministic recording that the game engine re-renders at any quality.

### Enable automatic replay saving

In `~/TmrlData/config/config.json`:

```json
{
  "interface_kwargs": {
    "save_replays": true
  }
}
```

When an episode finishes (car crosses the finish line), the tmrl plugin
tells OpenPlanet to save the replay.

### Where replays are stored

```
Documents/Trackmania/Replays/
```

### Export to video

1. In TM2020: Play > Local > My Replays
2. Select the replay
3. Open the **Replay Editor**
4. Place cameras, adjust angles, set duration
5. Export to video (the game renders each frame at full quality)

**Advantages**: Full game quality, any resolution, any camera angle,
post-production editing. This is how competitive TM2020 videos are made.

---

## Method 2: OBS Studio (Real-Time, All Platforms)

[OBS Studio](https://obsproject.com/) captures the game window live.

### Windows

1. Install OBS Studio
2. Add source: **Window Capture** > select Trackmania
3. Configure output:
   - Settings > Output > Recording Path
   - Recording Format: MKV or MP4
   - Encoder: NVENC (NVIDIA) or x264
4. Click **Start Recording**
5. Run the AI: `python -m tmrl --test`
6. Click **Stop Recording** when done

### Linux

1. Install: `sudo apt install obs-studio`
2. Add source: **Screen Capture** (X11) or **Window Capture** (PipeWire)
3. Same workflow as Windows

### Recommended OBS settings for TM2020

| Setting           | Value                      |
|-------------------|----------------------------|
| Resolution        | 1920x1080 (or match game)  |
| FPS               | 60                         |
| Encoder           | NVENC H.264 (if NVIDIA)    |
| Rate control      | CQP                        |
| CQ level          | 18 (high quality)          |
| Recording format  | MKV (no corruption on crash)|

---

## Method 3: ffmpeg (Command Line)

### Linux (X11 / Xvfb)

```bash
# Record the full display at 30 FPS
ffmpeg -f x11grab -framerate 30 -video_size 958x488 \
  -i :0.0+0,0 -c:v libx264 -preset fast -crf 23 \
  ai_drive.mp4
```

Parameters:
- `-i :0.0+0,0` -- display :0, offset (0,0) = top-left
- `-video_size 958x488` -- match the game window
- `-framerate 30` -- 30 FPS recording
- `-crf 23` -- quality (lower = better, 18-28 is useful range)

### Linux (Wayland / PipeWire)

```bash
ffmpeg -f pipewire -framerate 30 -i default \
  -c:v libx264 -preset fast -crf 23 \
  ai_drive.mp4
```

### Windows (GDI grab)

```powershell
ffmpeg -f gdigrab -framerate 30 -offset_x 0 -offset_y 0 ^
  -video_size 958x488 -i desktop ^
  -c:v libx264 -preset fast -crf 23 ^
  ai_drive.mp4
```

### Stop recording

Press `Ctrl+C` in the ffmpeg terminal (or `q`).

---

## Method 4: Windows Game Bar

The simplest option on Windows, but limited:

1. Press `Win+G` to open Game Bar
2. Click the record button (or press `Win+Alt+R`)
3. The recording indicator appears in the corner
4. Run the AI
5. Press `Win+Alt+R` again to stop
6. Videos saved to `Videos/Captures/`

Limitations: fixed quality, only captures the active window, no post-editing.

---

## Tips for Good AI Recordings

### Before recording

- Run a few test episodes first to verify the model works
- Use test mode (`python -m tmrl --test`) for deterministic behaviour
- Set TM2020 graphics to maximum quality
- Consider switching to a third-person camera (press `1`) for better visuals

### During recording

- Let 2-3 full laps run for variety
- The first lap may be slightly different as the agent warms up
- Record longer than needed -- you can trim later

### After recording

- Trim with ffmpeg:
  ```bash
  ffmpeg -i ai_drive.mp4 -ss 00:00:10 -to 00:02:30 -c copy trimmed.mp4
  ```
- Add speed-up for long sections:
  ```bash
  ffmpeg -i ai_drive.mp4 -filter:v "setpts=0.5*PTS" fast.mp4
  ```
- Concatenate multiple clips:
  ```bash
  ffmpeg -f concat -i filelist.txt -c copy combined.mp4
  ```

---

## Comparison

| Method              | Quality     | Effort   | Camera control | Platforms       |
|---------------------|-------------|----------|----------------|-----------------|
| Replay Editor       | Excellent   | Medium   | Full           | Windows, Linux  |
| OBS Studio          | Good        | Low      | None (live)    | All             |
| ffmpeg              | Good        | Low      | None (live)    | Linux, Windows  |
| Windows Game Bar    | OK          | Minimal  | None (live)    | Windows only    |
