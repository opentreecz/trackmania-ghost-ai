"""CLI entry points for Trackmania Ghost AI.

Provides the console scripts defined in ``pyproject.toml``:
    tm-train, tm-worker, tm-server, tm-evaluate
"""

from __future__ import annotations

import argparse
import logging
import sys

from trackmania_ghost_ai.config import load_config


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to YAML config file (default: configs/training.yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


# ---------------------------------------------------------------------------
# tm-train
# ---------------------------------------------------------------------------


def train() -> None:
    """Run the full training pipeline (server + worker + trainer)."""
    parser = _base_parser("Train the Trackmania Ghost AI agent")
    parser.add_argument(
        "--local",
        action="store_true",
        default=True,
        help="Run all components in one process (default: True)",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)

    cfg = load_config(args.config)

    from trackmania_ghost_ai.pipeline import run_local

    run_local(cfg)


# ---------------------------------------------------------------------------
# tm-server
# ---------------------------------------------------------------------------


def server() -> None:
    """Run the tmrl relay server only."""
    parser = _base_parser("Run the tmrl relay server")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    cfg = load_config(args.config)

    from trackmania_ghost_ai.pipeline import build_server

    srv = build_server(cfg)
    logging.getLogger(__name__).info(
        "Server listening on port %d ...", cfg.networking.server_port
    )
    srv.run()


# ---------------------------------------------------------------------------
# tm-worker
# ---------------------------------------------------------------------------


def worker() -> None:
    """Run a rollout worker (data collector)."""
    parser = _base_parser("Run a rollout worker")
    parser.add_argument(
        "--test-interval",
        type=int,
        default=10,
        help="Run a test episode every N training episodes",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)

    cfg = load_config(args.config)

    from trackmania_ghost_ai.pipeline import build_worker

    wkr = build_worker(cfg)
    logging.getLogger(__name__).info("Worker connecting to %s:%d ...", cfg.networking.server_ip, cfg.networking.server_port)
    wkr.run(test_episode_interval=args.test_interval)


# ---------------------------------------------------------------------------
# tm-evaluate
# ---------------------------------------------------------------------------


def evaluate() -> None:
    """Evaluate a trained model (test episodes only, no training)."""
    parser = _base_parser("Evaluate a trained Trackmania Ghost AI agent")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to saved model weights (.pt)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes to run",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)

    cfg = load_config(args.config)

    from trackmania_ghost_ai.pipeline import build_worker

    logger = logging.getLogger(__name__)

    wkr = build_worker(cfg)

    if args.weights:
        import torch

        state = torch.load(args.weights, map_location="cpu")
        logger.info("Loaded weights from %s", args.weights)

    logger.info("Running %d evaluation episodes ...", args.episodes)
    wkr.run(test_episode_interval=1)
