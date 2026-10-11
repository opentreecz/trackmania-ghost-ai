"""Environment wrapper and observation utilities for Trackmania LIDAR interface.

Provides helper functions to set up the ``GenericGymEnv`` with the correct
``rtgym`` configuration for the TM20LIDAR interface.
"""

from __future__ import annotations

import logging
from functools import partial as _partial
from typing import Any

import numpy as np

from tmrl.envs import GenericGymEnv

from trackmania_ghost_ai.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def preprocess_obs(obs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Optional observation post-processing.

    The raw LIDAR observation tuple is:
        (speed [1], lidar [H, 19], prev_act_1 [3], prev_act_2 [3])

    This function is a hook for any additional normalisation or feature
    engineering.  By default it passes through unchanged.
    """
    return obs


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_rtgym_config(cfg: Config) -> dict[str, Any]:
    """Build the rtgym configuration dict from project config."""
    return {
        "interface": cfg.environment.interface,
        "time_step_duration": cfg.environment.rtgym.time_step_duration,
        "start_obs_capture": cfg.environment.rtgym.start_obs_capture,
        "act_buf_len": cfg.environment.rtgym.act_buf_len,
        "benchmark": cfg.environment.rtgym.benchmark,
        "ep_max_length": cfg.environment.rtgym.ep_max_length,
    }


def make_env_cls(cfg: Config):
    """Return a partially-initialised ``GenericGymEnv`` class.

    This is the ``env_cls`` expected by tmrl's ``RolloutWorker`` and
    ``TorchTrainingOffline``.
    """
    gym_kwargs = {
        "id": "real-time-gym-ts-v1",
        "gym_kwargs": {"config": make_rtgym_config(cfg)},
    }
    env_cls = _partial(GenericGymEnv, **gym_kwargs)

    logger.info(
        "Environment: interface=%s, dt=%.3f, act_buf=%d, ep_max=%d",
        cfg.environment.interface,
        cfg.environment.rtgym.time_step_duration,
        cfg.environment.rtgym.act_buf_len,
        cfg.environment.rtgym.ep_max_length,
    )
    return env_cls


# ---------------------------------------------------------------------------
# Reward shaping (optional wrapper)
# ---------------------------------------------------------------------------


def shaped_reward(
    raw_reward: float,
    info: dict[str, Any],
    cfg: Config,
) -> float:
    """Apply reward shaping on top of the environment's native reward.

    Parameters
    ----------
    raw_reward:
        Reward returned by the environment step.
    info:
        Info dict from the environment step.
    cfg:
        Project configuration (carries reward weights).

    Returns
    -------
    float
        Shaped reward.
    """
    r = raw_reward
    rc = cfg.reward

    # Velocity bonus
    speed = info.get("speed", 0.0)
    r += rc.velocity_weight * speed / 1000.0  # normalise roughly to [0, 1]

    # Progress bonus
    progress = info.get("progress", 0.0)
    r += rc.progress_weight * progress

    # Checkpoint / finish bonuses
    if info.get("checkpoint_reached", False):
        r += rc.checkpoint_bonus
    if info.get("finished", False):
        r += rc.finish_bonus

    # Penalties
    if info.get("collision", False):
        r += rc.collision_penalty
    if info.get("off_track", False):
        r += rc.off_track_penalty

    return r
