# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Register ``physicalai engine`` worker-environment management commands."""

from __future__ import annotations

from jsonargparse import ArgumentParser, Namespace
from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

from physicalai.engine import Engine, list_prepared, prepare, prune

HELP = "Prepare, inspect, and prune isolated framework worker environments."
_SIZE_BASE = 1024


def build_parser() -> ArgumentParser:
    """Build the worker environment management parser.

    Returns:
        Parser for ``physicalai engine``.
    """
    parser = ArgumentParser(prog="physicalai engine", description=HELP)
    parser.add_argument("action", choices=("prepare", "list", "prune"))
    parser.add_argument("framework", nargs="?", help="Engine identifier required by 'prepare'.")
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "cuda", "cu128", "xpu"),
        default="auto",
        help="Worker environment accelerator for 'prepare'.",
    )
    parser.add_argument("--offline", action="store_true", help="Prepare without network access.")
    return parser


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Execute environment management and print the result for CLI users.

    Returns:
        Zero after a successful command.

    Raises:
        ValueError: If ``prepare`` has no engine name or the selected engine has no worker environment.
    """
    del parser
    if cfg.action == "prepare":
        if cfg.framework is None:
            msg = "physicalai engine prepare requires an engine name."
            raise ValueError(msg)
        engine_class = Engine.get(cfg.framework)
        if engine_class.env is None:
            msg = f"Engine {cfg.framework!r} does not use a managed worker environment."
            raise ValueError(msg)
        python = prepare(engine_class.env, cfg.accelerator, offline=cfg.offline)
        print(f"Prepared {engine_class.framework} worker environment: {python}")  # noqa: T201
        return 0

    if cfg.action == "list":
        environments = list_prepared()
        if not environments:
            print("No prepared worker environments.")  # noqa: T201
        for environment in environments:
            print(  # noqa: T201
                f"{environment.framework}\t{environment.accelerator}\t"
                f"{environment.lock_hash[:12]}\t{_format_size(environment.size_bytes)}\t{environment.path}",
            )
        return 0

    removed = prune()
    for path in removed:
        print(f"Removed stale worker environment: {path}")  # noqa: T201
    if not removed:
        print("No stale worker environments.")  # noqa: T201
    return 0


def print_help(prog: str) -> None:
    """Print concise command help without preparing any engine environment."""
    print(  # noqa: T201
        f"usage: {prog} <prepare|list|prune> [framework] [--accelerator NAME] [--offline]\n\n"
        f"{HELP}\n\n"
        "Example: physicalai engine prepare lerobot --accelerator xpu",
    )


def register() -> SubcommandSpec:
    """Return the Runtime CLI subcommand registration.

    Returns:
        The ``engine`` subcommand spec.
    """
    return SubcommandSpec(name="engine", parser=build_parser(), dispatch=run, help=HELP)


def _format_size(size_bytes: int) -> str:
    """Format an environment size using a compact binary unit.

    Returns:
        The value with a binary unit suffix.
    """
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < _SIZE_BASE or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= _SIZE_BASE
    return f"{value:.1f} TiB"
