"""SAC training agent construction.

Wraps ``tmrl.custom.custom_algorithms.SpinupSacAgent`` with project-specific
model and hyper-parameter configuration.
"""

from __future__ import annotations

import logging
from functools import partial as _partial

import torch.nn as nn

from tmrl.custom.custom_algorithms import SpinupSacAgent

from trackmania_ghost_ai.config import Config
from trackmania_ghost_ai.models import MLPActorCritic

logger = logging.getLogger(__name__)

# Map config string -> torch activation class
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "GELU": nn.GELU,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "SiLU": nn.SiLU,
}


def _resolve_activation(name: str) -> type[nn.Module]:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS.keys())}."
        )
    return _ACTIVATIONS[name]


def make_training_agent_cls(cfg: Config):
    """Return a partially-initialised ``SpinupSacAgent``.

    Parameters
    ----------
    cfg:
        Project configuration.

    Returns
    -------
    partial
        Ready for instantiation by ``TorchTrainingOffline``.
    """
    alg = cfg.algorithm
    activation = _resolve_activation(alg.actor.activation)

    model_cls = _partial(
        MLPActorCritic,
        hidden_sizes=tuple(alg.actor.hidden_sizes),
        activation=activation,
    )

    training_agent_cls = _partial(
        SpinupSacAgent,
        model_cls=model_cls,
        gamma=alg.gamma,
        polyak=alg.polyak,
        alpha=alg.alpha,
        lr_actor=alg.actor.lr,
        lr_critic=alg.critic.lr,
        lr_entropy=alg.entropy.lr,
        learn_entropy_coef=alg.learn_entropy_coef,
        target_entropy=alg.target_entropy,
    )

    logger.info(
        "Agent: algorithm=%s, gamma=%.4f, polyak=%.4f, alpha=%.2f, "
        "lr_actor=%.1e, lr_critic=%.1e, hidden=%s",
        alg.name,
        alg.gamma,
        alg.polyak,
        alg.alpha,
        alg.actor.lr,
        alg.critic.lr,
        alg.actor.hidden_sizes,
    )
    return training_agent_cls
