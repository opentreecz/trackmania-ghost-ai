"""MLP-based actor and critic models for SAC on Trackmania LIDAR observations.

Architecture
------------
- **SquashedGaussianMLPActor**: Outputs a squashed-Gaussian action distribution.
  Used as the policy network in SAC.
- **MLPQFunction**: Twin Q-value estimators.
- **MLPActorCritic**: Bundles actor + 2 critics for the SAC training agent.

The observation space is a *tuple* of tensors:
    (speed [1], lidar [H, 19], prev_act_1 [3], prev_act_2 [3])
All sub-tensors are flattened and concatenated before being fed into the MLP.
"""

from __future__ import annotations

import logging
from math import prod
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from tmrl.actor import TorchActorModule

logger = logging.getLogger(__name__)

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mlp(
    sizes: Sequence[int],
    activation: type[nn.Module] = nn.ReLU,
    output_activation: type[nn.Module] = nn.Identity,
) -> nn.Sequential:
    """Build a simple feedforward MLP."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


def _obs_dim(observation_space) -> int:
    """Compute total flattened dimension from a Tuple observation space."""
    if hasattr(observation_space, "spaces"):
        return sum(prod(s.shape) for s in observation_space.spaces)
    return prod(observation_space.shape)


def _flatten_obs(obs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Flatten and concatenate a tuple of observation tensors."""
    flat = []
    for o in obs:
        if o.dim() == 1:
            flat.append(o)
        else:
            flat.append(o.flatten(start_dim=-len(o.shape) + (1 if o.dim() > 1 else 0)))
    return torch.cat(flat, dim=-1)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


class SquashedGaussianMLPActor(TorchActorModule):
    """MLP policy that outputs tanh-squashed Gaussian actions (SAC-style).

    Parameters
    ----------
    observation_space:
        Gymnasium observation space (typically ``Tuple(Box, Box, Box, Box)``).
    action_space:
        Gymnasium action space (``Box``).
    hidden_sizes:
        Widths of the hidden layers.
    activation:
        Activation function class.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__(observation_space, action_space)

        self.dim_obs = _obs_dim(observation_space)
        self.dim_act = action_space.shape[0]
        self.act_limit = float(action_space.high[0])

        self.net = _build_mlp(
            [self.dim_obs, *hidden_sizes],
            activation=activation,
            output_activation=activation,
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], self.dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], self.dim_act)

        logger.info(
            "SquashedGaussianMLPActor: obs_dim=%d, act_dim=%d, hidden=%s",
            self.dim_obs,
            self.dim_act,
            list(hidden_sizes),
        )

    def forward(
        self,
        obs: tuple[torch.Tensor, ...],
        test: bool = False,
        with_logprob: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning (action, log_prob)."""
        x = _flatten_obs(obs)
        net_out = self.net(x)

        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_dist = Normal(mu, std)
        pi_action = mu if test else pi_dist.rsample()

        logp: torch.Tensor | None = None
        if with_logprob:
            logp = pi_dist.log_prob(pi_action).sum(dim=-1)
            # Correction for tanh squashing
            logp -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(dim=-1)

        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, logp

    def act(self, obs: tuple[torch.Tensor, ...], test: bool = False) -> np.ndarray:
        """Select an action given an observation (no grad)."""
        with torch.no_grad():
            action, _ = self.forward(obs, test=test, with_logprob=False)
            return action.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class MLPQFunction(nn.Module):
    """MLP Q-value function Q(s, a) -> scalar."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        dim_obs = _obs_dim(observation_space)
        dim_act = action_space.shape[0]
        self.q = _build_mlp([dim_obs + dim_act, *hidden_sizes, 1], activation=activation)

    def forward(
        self, obs: tuple[torch.Tensor, ...], act: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([_flatten_obs(obs), act], dim=-1)
        return self.q(x).squeeze(-1)


# ---------------------------------------------------------------------------
# ActorCritic bundle
# ---------------------------------------------------------------------------


class MLPActorCritic(nn.Module):
    """Actor-Critic module bundling the policy with twin Q-functions.

    This is the ``model_cls`` passed to ``SpinupSacAgent``.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.actor = SquashedGaussianMLPActor(
            observation_space, action_space, hidden_sizes, activation
        )
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
