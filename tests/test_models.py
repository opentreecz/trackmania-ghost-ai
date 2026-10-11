"""Tests for MLP model architecture."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gymnasium.spaces import Box, Tuple as GymTuple

from trackmania_ghost_ai.models import (
    MLPActorCritic,
    MLPQFunction,
    SquashedGaussianMLPActor,
    _build_mlp,
    _obs_dim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lidar_obs_space():
    """Mimics the TM20LIDAR observation space: (speed, lidar, act1, act2)."""
    return GymTuple(
        (
            Box(low=-np.inf, high=np.inf, shape=(1,)),       # speed
            Box(low=-np.inf, high=np.inf, shape=(4, 19)),    # lidar history
            Box(low=-1.0, high=1.0, shape=(3,)),             # prev action 1
            Box(low=-1.0, high=1.0, shape=(3,)),             # prev action 2
        )
    )


@pytest.fixture
def action_space():
    return Box(low=-1.0, high=1.0, shape=(3,))


@pytest.fixture
def sample_obs():
    """A single observation as tuple of tensors (batch_size=1)."""
    return (
        torch.randn(1, 1),
        torch.randn(1, 4, 19),
        torch.randn(1, 3),
        torch.randn(1, 3),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_build_mlp_output_shape(self):
        net = _build_mlp([10, 64, 32, 1])
        x = torch.randn(5, 10)
        out = net(x)
        assert out.shape == (5, 1)

    def test_obs_dim_tuple_space(self, lidar_obs_space):
        # 1 + 4*19 + 3 + 3 = 83
        assert _obs_dim(lidar_obs_space) == 83

    def test_obs_dim_box_space(self, action_space):
        assert _obs_dim(action_space) == 3


class TestSquashedGaussianMLPActor:
    def test_construction(self, lidar_obs_space, action_space):
        actor = SquashedGaussianMLPActor(lidar_obs_space, action_space)
        assert actor.dim_obs == 83
        assert actor.dim_act == 3

    def test_forward_train(self, lidar_obs_space, action_space, sample_obs):
        actor = SquashedGaussianMLPActor(lidar_obs_space, action_space)
        action, logp = actor.forward(sample_obs, test=False, with_logprob=True)
        assert action.shape == (1, 3)
        assert logp is not None
        assert logp.shape == (1,)

    def test_forward_test(self, lidar_obs_space, action_space, sample_obs):
        actor = SquashedGaussianMLPActor(lidar_obs_space, action_space)
        action, logp = actor.forward(sample_obs, test=True, with_logprob=False)
        assert action.shape == (1, 3)
        assert logp is None

    def test_act_returns_numpy(self, lidar_obs_space, action_space, sample_obs):
        actor = SquashedGaussianMLPActor(lidar_obs_space, action_space)
        action = actor.act(sample_obs, test=True)
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)

    def test_actions_in_bounds(self, lidar_obs_space, action_space, sample_obs):
        actor = SquashedGaussianMLPActor(lidar_obs_space, action_space)
        for _ in range(20):
            action = actor.act(sample_obs, test=False)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)


class TestMLPQFunction:
    def test_output_shape(self, lidar_obs_space, action_space, sample_obs):
        q = MLPQFunction(lidar_obs_space, action_space)
        act = torch.randn(1, 3)
        val = q(sample_obs, act)
        assert val.shape == (1,)


class TestMLPActorCritic:
    def test_has_actor_and_two_critics(self, lidar_obs_space, action_space):
        ac = MLPActorCritic(lidar_obs_space, action_space)
        assert hasattr(ac, "actor")
        assert hasattr(ac, "q1")
        assert hasattr(ac, "q2")

    def test_forward_pass(self, lidar_obs_space, action_space, sample_obs):
        ac = MLPActorCritic(lidar_obs_space, action_space)
        action, logp = ac.actor.forward(sample_obs, test=False)
        q1_val = ac.q1(sample_obs, action)
        q2_val = ac.q2(sample_obs, action)
        assert q1_val.shape == (1,)
        assert q2_val.shape == (1,)
