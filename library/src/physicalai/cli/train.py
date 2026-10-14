# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Register ``physicalai train <framework>`` with native argument passthrough."""

from __future__ import annotations

import argparse

from jsonargparse import ArgumentParser, Namespace
from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

from physicalai.engine import Engine

HELP = "Train with a native policy framework."


def build_parser() -> ArgumentParser:
    """Build a parser that consumes only the framework name and preserves its arguments.

    Returns:
        Parser with native framework flags stored as remainder tokens.
    """
    parser = ArgumentParser(prog="physicalai train", description=HELP, add_help=False)
    parser.add_argument("framework", help="Installed framework identifier, for example 'lerobot'.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded unchanged to the framework.")
    return parser


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Dispatch to the selected framework without interpreting native options.

    Returns:
        Process exit code from the Studio-owned train subcommand.
    """
    del parser
    Engine().train(cfg.args, framework=cfg.framework)
    return 0


def print_help(prog: str) -> None:
    """Print lightweight help for ``physicalai train``."""
    print(  # noqa: T201
        f"usage: {prog} <framework> [framework arguments]\n\n"
        f"{HELP}\n\n"
        "Example: physicalai train lerobot --policy.type=act --dataset.repo_id=lerobot/pusht",
    )


def register() -> SubcommandSpec:
    """Return the runtime CLI subcommand registration."""
    return SubcommandSpec(name="train", parser=build_parser(), dispatch=run, help=HELP)
