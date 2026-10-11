"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from trackmania_ghost_ai.config import Config, load_config


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestConfigDefaults:
    """Config should have sensible defaults even without a file."""

    def test_default_config_creates_without_error(self):
        cfg = Config()
        assert cfg.run.name == "ghost-ai-default"
        assert cfg.algorithm.name == "SAC"

    def test_default_gamma(self):
        cfg = Config()
        assert cfg.algorithm.gamma == 0.995

    def test_default_observation_dim(self):
        cfg = Config()
        # speed(1) + lidar(4*19) + prev_actions(2*3) = 83
        assert cfg.observation.flat_dim == 83

    def test_default_action_dim(self):
        cfg = Config()
        assert cfg.action.dim == 3

    def test_default_hidden_sizes(self):
        cfg = Config()
        assert cfg.algorithm.actor.hidden_sizes == [256, 256]


class TestConfigLoading:
    """Loading from YAML files."""

    def test_load_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, Config)
        assert cfg.run.seed == 42

    def test_load_project_config(self):
        """Load the actual project config file."""
        project_config = Path(__file__).parent.parent / "configs" / "training.yaml"
        if not project_config.exists():
            pytest.skip("Project config not found")

        cfg = load_config(project_config)
        assert cfg.run.name == "ghost-ai-default"
        assert cfg.algorithm.gamma == 0.995
        assert cfg.memory.batch_size == 256

    def test_load_partial_override(self, tmp_path):
        """Partial YAML should override only specified keys."""
        partial_yaml = tmp_path / "partial.yaml"
        partial_yaml.write_text(
            "run:\n  name: custom-run\n  seed: 123\n"
        )
        cfg = load_config(partial_yaml)
        assert cfg.run.name == "custom-run"
        assert cfg.run.seed == 123
        # Other defaults should remain
        assert cfg.algorithm.gamma == 0.995
