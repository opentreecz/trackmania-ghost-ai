# Headless Mode and Distributed Training

How to run training across multiple machines, run the trainer on a headless
server, and set up the network topology.

---

## Architecture Recap

```
Machine A (Windows, has the game)       Machine B (Linux, has the GPU)
+-------------------+                   +-------------------+
|   RolloutWorker   | --- samples --->  |      Server       |
|   (TM2020 + tmrl) | <-- weights ---   |      Trainer      |
+-------------------+                   +-------------------+
```

The **worker** needs the game. The **server** and **trainer** do not.

---

## Headless Trainer on Linux

### What runs headless (no display, no game)

| Component | Needs game | Needs display | Needs GPU        |
|-----------|------------|---------------|------------------|
| Server    | No         | No            | No               |
| Trainer   | No         | No            | Yes (recommended)|
| Worker    | Yes        | Yes           | Yes (for game)   |

### Setup on a headless Linux server

```bash
# Install project
git clone https://github.com/opentreecz/trackmania-ghost-ai.git
cd trackmania-ghost-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

No need to install tmrl plugins, TM2020, vgamepad, or OpenPlanet.

### Run server + trainer

```bash
# Configure for remote access
# configs/distributed.yaml
cat > configs/distributed.yaml << 'EOF'
run:
  name: "distributed-run"
  device: "cuda"

networking:
  server_ip: "0.0.0.0"    # listen on all interfaces
  server_port: 6666
  password: "your-strong-password"
  local_mode: false

training:
  epochs: 200
  rounds: 50
  steps: 100
EOF

# Start server in background
tm-server --config configs/distributed.yaml &

# Start trainer (main process)
tm-train --config configs/distributed.yaml --verbose
```

### Run as a systemd service

```ini
# /etc/systemd/system/tm-trainer.service
[Unit]
Description=Trackmania Ghost AI Trainer
After=network.target

[Service]
Type=simple
User=ml-user
WorkingDirectory=/home/ml-user/trackmania-ghost-ai
Environment=PATH=/home/ml-user/trackmania-ghost-ai/.venv/bin
ExecStart=/home/ml-user/trackmania-ghost-ai/.venv/bin/tm-train \
    --config configs/distributed.yaml --verbose
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable tm-trainer
sudo systemctl start tm-trainer
journalctl -u tm-trainer -f   # watch logs
```

---

## Distributed Training Setup

### Network topology

```
Worker (Windows)  --->  Server (Linux)  <---  Trainer (Linux)
   port 6666              port 6666              port 6666
   outbound               listening              outbound
```

The server listens on a port. Worker and trainer connect outbound.

### Step 1: Configure the server machine

On the server/trainer machine (Linux with GPU):

```yaml
# configs/distributed.yaml
networking:
  server_ip: "0.0.0.0"         # bind to all interfaces
  server_port: 6666
  password: "your-strong-password"
  security: null                # or TLS config path
  local_mode: false
```

Open the firewall:
```bash
sudo ufw allow 6666/tcp
# or: sudo firewall-cmd --add-port=6666/tcp --permanent
```

### Step 2: Configure the worker machine

On the worker machine (Windows with TM2020):

```yaml
# configs/worker.yaml
networking:
  server_ip: "192.168.1.100"   # IP of the server machine
  server_port: 6666
  password: "your-strong-password"
  local_mode: false
```

### Step 3: Launch in order

```bash
# Machine B (Linux): start server
tm-server --config configs/distributed.yaml

# Machine B (Linux): start trainer
tm-train --config configs/distributed.yaml

# Machine A (Windows): start worker
tm-worker --config configs/worker.yaml
# Click the TM2020 window!
```

### Port forwarding (over the internet)

If the machines are on different networks:

1. Forward port 6666 on the server machine's router
2. Use the server's public IP in the worker config
3. Consider enabling TLS for security

### TLS encryption

For training over the internet, enable TLS:

```yaml
networking:
  security: "path/to/tls-config"
```

tmrl uses `tlspyo` which can generate self-signed certificates. See the
[tlspyo documentation](https://github.com/MISTlab/tlspyo) for TLS setup.

---

## Multiple Workers

You can run multiple workers (each with their own game instance) to collect
data faster:

```
Worker 1 (PC 1) ---\
Worker 2 (PC 2) ----+---> Server ---> Trainer
Worker 3 (PC 3) ---/
```

Each worker needs:
- Its own TM2020 instance
- The same config (pointing to the server)
- A unique local port if on the same machine

```yaml
# Worker 2 config (if on same machine, change local port)
# In ~/TmrlData/config/config.json:
# "LOCAL_PORT_WORKER": 6670  (different from default 6660)
```

### Scaling expectations

| Workers | Data rate   | Trainer utilisation |
|---------|-------------|---------------------|
| 1       | 1x          | Often waiting       |
| 2       | ~1.8x       | Good utilisation    |
| 3-4     | ~3x         | Near full           |
| 5+      | Diminishing | Trainer bottleneck  |

---

## Docker Setup

### Trainer + Server in Docker

```dockerfile
# Dockerfile.trainer
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 6666

CMD ["sh", "-c", "tm-server --config configs/distributed.yaml & tm-train --config configs/distributed.yaml --verbose"]
```

```bash
docker build -f Dockerfile.trainer -t tm-trainer .
docker run --gpus all -p 6666:6666 tm-trainer
```

### Docker Compose

```yaml
# docker-compose.yaml
version: "3.8"

services:
  server:
    build:
      context: .
      dockerfile: Dockerfile.trainer
    command: tm-server --config configs/distributed.yaml
    ports:
      - "6666:6666"

  trainer:
    build:
      context: .
      dockerfile: Dockerfile.trainer
    command: tm-train --config configs/distributed.yaml --verbose
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - server
```

```bash
docker compose up
# Worker still runs on Windows natively
```

---

## Monitoring Remote Training

### SSH tunnel for W&B

If the trainer runs on a remote server:

```bash
# On the server
pip install wandb
wandb login

# In configs/distributed.yaml
run:
  wandb:
    enabled: true
    project: "trackmania-ghost-ai"
```

View at [wandb.ai](https://wandb.ai).

### SSH to check status

```bash
ssh ml-server "tail -f /home/ml-user/trackmania-ghost-ai/training.log"
```

### Check GPU utilisation

```bash
ssh ml-server "nvidia-smi"
```
