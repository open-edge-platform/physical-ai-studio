# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Register the ``physicalai benchmark`` subcommand."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, cast

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

from physicalai.cli._help import print_benchmark_help  # noqa: PLC2701
from physicalai.cli._policy import load_policy  # noqa: PLC2701


class _WritableResults(Protocol):
    """Protocol for benchmark result persistence helpers."""

    def summary(self) -> str:
        """Return a human-readable summary string."""

    def to_json(self, path: Path) -> None:
        """Write results as JSON."""

    def to_csv(self, path: Path) -> None:
        """Write results as CSV."""


logger = logging.getLogger(__name__)

HELP = "Run benchmark evaluation."


def build_parser() -> ArgumentParser:
    """Build the ``benchmark`` parser.

    Returns:
        Parser for ``physicalai benchmark``.
    """
    from physicalai.benchmark.gyms import Benchmark  # noqa: PLC0415

    parser = ArgumentParser(prog="physicalai benchmark", description="Run benchmark evaluation on a trained policy.")
    parser.add_argument("--config", action=ActionConfigFile, help="YAML/JSON config file.")
    parser.add_subclass_arguments(Benchmark, "benchmark", required=True)
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Policy class path (e.g., physicalai.policies.ACT).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=(
            "Path to checkpoint file (.ckpt) or export directory. "
            "For Policy subclasses: path to Lightning checkpoint (.ckpt). "
            "For InferenceModel: path to export directory containing model files. "
            "If not provided, uses randomly initialized policy (Policy only)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/benchmark",
        help="Directory to save benchmark results.",
    )
    return parser


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Dispatch ``benchmark`` by instantiating benchmark and evaluating policy.

    Args:
        parser: Parser used to instantiate the benchmark.
        cfg: Parsed configuration namespace.

    Returns:
        Process exit code.
    """
    instantiated = cast("Namespace", parser.instantiate_classes(Namespace(benchmark=cfg.benchmark)))
    benchmark = instantiated.benchmark
    policy, device = load_policy(cfg.policy, cfg.ckpt_path)

    logger.info("Benchmark: %s", benchmark)
    logger.info("Policy: %s", type(policy).__name__)
    logger.info("Device: %s", device)

    try:
        results = benchmark.evaluate(policy=policy)
        _write_results(results, cfg.output_dir)
    finally:
        for gym in benchmark.gyms:
            gym.close()

    return 0


def _write_results(results: _WritableResults, output_dir: str) -> None:
    """Print and persist benchmark results.

    Args:
        results: Benchmark results object with summary and export helpers.
        output_dir: Directory where results files should be written.
    """
    print(results.summary())  # noqa: T201

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results.to_json(output_path / "results.json")
    results.to_csv(output_path / "results.csv")
    logger.info("Results saved to %s", output_path)


def print_help(prog: str) -> None:
    """Print lightweight help without building the full parser."""
    print_benchmark_help(prog, description=HELP)


def register() -> SubcommandSpec:
    """Return the ``benchmark`` subcommand spec.

    Returns:
        Registered spec for the shared CLI host.
    """
    return SubcommandSpec(name="benchmark", parser=build_parser(), dispatch=run, help=HELP)
