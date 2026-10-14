# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Register ``physicalai train <framework>`` with native argument passthrough."""

from __future__ import annotations

import argparse
from pathlib import Path

from jsonargparse import ArgumentParser, Namespace
from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

from physicalai.engine import Engine

HELP = "Train with a native policy framework."


def build_parser() -> ArgumentParser:
    """Build a parser that consumes Studio options before the framework name.

    Returns:
        Parser for ``physicalai train``.
    """
    parser = ArgumentParser(prog="physicalai train", description=HELP, add_help=False)
    parser.add_argument("--interpreter", type=Path, help="Use an existing framework Python environment.")
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "cuda", "cu128", "xpu"),
        default="auto",
        help="Worker environment accelerator (default: auto).",
    )
    parser.add_argument("framework", help="Installed engine identifier, for example 'lerobot'.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded unchanged to the framework.")
    return parser


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Run the framework's native command and return its exit code.

    Returns:
        The native framework command's exit code.
    """
    del parser
    engine_class = Engine.get(cfg.framework)
    return engine_class.cli(
        cfg.args,
        interpreter=cfg.interpreter,
        accelerator=cfg.accelerator,
    )


def print_help(prog: str) -> None:
    """Print lightweight help for ``physicalai train``."""
    print(  # noqa: T201
        f"usage: {prog} [--interpreter PYTHON] [--accelerator NAME] <framework> [native arguments]\n\n"
        f"{HELP}\n\n"
        "Example: physicalai train lerobot --policy.type=act --dataset.repo_id=lerobot/pusht",
    )


def register() -> SubcommandSpec:
    """Return the Runtime CLI subcommand registration.

    Returns:
        The ``train`` subcommand spec.
    """
    return SubcommandSpec(name="train", parser=build_parser(), dispatch=run, help=HELP)
