"""Orchestrator: builds and runs the tmrl training pipeline.

Ties together environment, memory, agent, and networking into the full
Server / Trainer / RolloutWorker pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import partial as _partial
from pathlib import Path
from typing import Any

import torch

from tmrl.networking import Server, Trainer, RolloutWorker
from tmrl.training_offline import TorchTrainingOffline

from trackmania_ghost_ai.agent import make_training_agent_cls
from trackmania_ghost_ai.config import Config
from trackmania_ghost_ai.environment import make_env_cls
from trackmania_ghost_ai.memory import make_memory_cls
from trackmania_ghost_ai.models import SquashedGaussianMLPActor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def resolve_device(device: str | None) -> str:
    """Pick training device: explicit string, or auto-detect."""
    if device:
        return device
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    logger.info("Auto-selected device: %s", dev)
    return dev


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------


def build_server(cfg: Config) -> Server:
    """Create the relay server."""
    return Server(
        security=cfg.networking.security,
        password=cfg.networking.password,
        port=cfg.networking.server_port,
    )


def build_worker(cfg: Config) -> RolloutWorker:
    """Create a rollout worker (data collector)."""
    env_cls = make_env_cls(cfg)

    actor_cls = _partial(
        SquashedGaussianMLPActor,
        hidden_sizes=tuple(cfg.algorithm.actor.hidden_sizes),
    )

    return RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=actor_cls,
        sample_compressor=None,
        device="cpu",
        server_ip=cfg.networking.server_ip,
        server_port=cfg.networking.server_port,
        password=cfg.networking.password,
    )


def build_training(cfg: Config) -> TorchTrainingOffline:
    """Create the training (offline) instance."""
    env_cls = make_env_cls(cfg)
    memory_cls = make_memory_cls(cfg)
    training_agent_cls = make_training_agent_cls(cfg)
    device = resolve_device(cfg.run.device)

    return TorchTrainingOffline(
        env_cls=env_cls,
        memory_cls=memory_cls,
        training_agent_cls=training_agent_cls,
        epochs=cfg.training.epochs,
        rounds=cfg.training.rounds,
        steps=cfg.training.steps,
        update_model_interval=cfg.training.update_model_interval,
        update_buffer_interval=cfg.training.update_buffer_interval,
        start_training=cfg.training.start_training,
        device=device,
    )


def build_trainer(cfg: Config) -> Trainer:
    """Create the trainer (connects training loop to server)."""
    training = build_training(cfg)
    return Trainer(
        training=training,
        server_ip=cfg.networking.server_ip,
        server_port=cfg.networking.server_port,
        password=cfg.networking.password,
    )


# ---------------------------------------------------------------------------
# Local all-in-one runner
# ---------------------------------------------------------------------------


def run_local(cfg: Config) -> None:
    """Run server, worker, and trainer all in one process (threads).

    Useful for development / single-machine training.
    """
    logger.info("Starting local training pipeline (all-in-one) ...")

    server = build_server(cfg)
    worker = build_worker(cfg)
    trainer = build_trainer(cfg)

    # Start server in background
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    time.sleep(1.0)  # give server time to bind

    # Start worker in background
    worker_thread = threading.Thread(
        target=worker.run,
        kwargs={"test_episode_interval": 10},
        daemon=True,
    )
    worker_thread.start()
    time.sleep(0.5)

    # Trainer runs in the main thread
    logger.info("Trainer starting (main thread) ...")
    trainer.run()


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    trainer: Trainer,
    cfg: Config,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save model weights and training state."""
    ckpt_dir = Path(cfg.checkpoint.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"{cfg.run.name}_epoch{epoch:04d}.pt"
    state = {
        "epoch": epoch,
        "config": cfg,
    }
    if extra:
        state.update(extra)

    torch.save(state, path)
    logger.info("Checkpoint saved: %s", path)

    # Prune old checkpoints
    existing = sorted(ckpt_dir.glob(f"{cfg.run.name}_epoch*.pt"))
    if len(existing) > cfg.checkpoint.keep_last:
        for old in existing[: len(existing) - cfg.checkpoint.keep_last]:
            old.unlink()
            logger.debug("Pruned old checkpoint: %s", old)

    return path
