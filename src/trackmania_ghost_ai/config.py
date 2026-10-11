"""Configuration loader and dataclasses for Trackmania Ghost AI."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "training.yaml"


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "trackmania-ghost-ai"
    entity: str | None = None


@dataclass
class RunConfig:
    name: str = "ghost-ai-default"
    seed: int = 42
    device: str | None = None  # None = auto
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class RtgymConfig:
    time_step_duration: float = 0.05
    start_obs_capture: float = 0.04
    act_buf_len: int = 2
    ep_max_length: int = 2000
    benchmark: bool = False


@dataclass
class EnvironmentConfig:
    interface: str = "TM20LIDAR"
    window_width: int = 958
    window_height: int = 488
    img_hist_len: int = 4
    rtgym: RtgymConfig = field(default_factory=RtgymConfig)


@dataclass
class ObservationConfig:
    lidar_beams: int = 19
    speed_dim: int = 1

    @property
    def flat_dim(self) -> int:
        """Total flattened observation dimension.

        speed(1) + lidar(img_hist_len * beams) + prev_actions(act_buf_len * 3).
        Default: 1 + 4*19 + 2*3 = 83.
        """
        return self.speed_dim + 4 * self.lidar_beams + 2 * 3


@dataclass
class ActionConfig:
    dim: int = 3
    low: float = -1.0
    high: float = 1.0


@dataclass
class RewardConfig:
    velocity_weight: float = 1.0
    progress_weight: float = 5.0
    checkpoint_bonus: float = 10.0
    finish_bonus: float = 100.0
    collision_penalty: float = -5.0
    off_track_penalty: float = -2.0


@dataclass
class NetworkConfig:
    lr: float = 0.0003
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "ReLU"


@dataclass
class AlgorithmConfig:
    name: str = "SAC"
    gamma: float = 0.995
    polyak: float = 0.995
    alpha: float = 0.37
    learn_entropy_coef: bool = True
    target_entropy: float | None = -7.0
    actor: NetworkConfig = field(default_factory=NetworkConfig)
    critic: NetworkConfig = field(
        default_factory=lambda: NetworkConfig(lr=0.00005)
    )
    entropy: NetworkConfig = field(
        default_factory=lambda: NetworkConfig(lr=0.0003)
    )


@dataclass
class MemoryConfig:
    capacity: int = 1_000_000
    batch_size: int = 256
    min_samples: int = 1000


@dataclass
class TrainingConfig:
    epochs: int = 200
    rounds: int = 50
    steps: int = 100
    start_training: int = 1000
    update_model_interval: int = 100
    update_buffer_interval: int = 100


@dataclass
class NetworkingConfig:
    server_ip: str = "127.0.0.1"
    server_port: int = 6666
    password: str = "trackmania-ghost-ai"
    security: str | None = None
    local_mode: bool = True


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    save_interval: int = 10
    keep_last: int = 5


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Top-level project configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    networking: NetworkingConfig = field(default_factory=NetworkingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dict to a nested dataclass instance."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}

    for key, value in data.items():
        if key not in field_types:
            logger.warning("Unknown config key %r in %s, ignoring.", key, cls.__name__)
            continue

        ft = field_types[key]
        # Resolve string type annotations for nested dataclasses
        if isinstance(ft, str):
            ft = eval(ft)  # noqa: S307 – only our own annotations

        if isinstance(value, dict) and hasattr(ft, "__dataclass_fields__"):
            kwargs[key] = _dict_to_dataclass(ft, value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML config file.  Falls back to ``configs/training.yaml``.

    Returns
    -------
    Config
        Fully-parsed configuration dataclass.
    """
    path = Path(path) if path else DEFAULT_CONFIG

    if not path.exists():
        logger.info("Config file %s not found, using defaults.", path)
        return Config()

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return _dict_to_dataclass(Config, raw)
