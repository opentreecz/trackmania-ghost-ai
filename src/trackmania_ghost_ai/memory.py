"""Replay memory for off-policy training.

Wraps ``tmrl.custom.custom_memories.GenericTorchMemory`` with project
defaults and optional extensions (e.g. Prioritized Experience Replay).
"""

from __future__ import annotations

import logging
from functools import partial as _partial

from tmrl.custom.custom_memories import GenericTorchMemory

from trackmania_ghost_ai.config import Config

logger = logging.getLogger(__name__)


def make_memory_cls(cfg: Config):
    """Return a partially-initialised memory class.

    Parameters
    ----------
    cfg:
        Project configuration.

    Returns
    -------
    partial
        A ``functools.partial`` wrapping ``GenericTorchMemory`` ready for
        instantiation by the tmrl ``Trainer``.
    """
    memory_cls = _partial(
        GenericTorchMemory,
        memory_size=cfg.memory.capacity,
        batch_size=cfg.memory.batch_size,
        dataset_path="",  # empty = in-memory only
        imgs_obs=0,       # MLP mode: no image observations
        act_buf_len=cfg.environment.rtgym.act_buf_len,
        crc_debug=False,
        use_dataloader=False,
        pin_memory=False,
    )

    logger.info(
        "Memory: capacity=%d, batch_size=%d, min_samples=%d",
        cfg.memory.capacity,
        cfg.memory.batch_size,
        cfg.memory.min_samples,
    )
    return memory_cls
