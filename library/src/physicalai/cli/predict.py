# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Register the ``physicalai predict`` subcommand."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

from physicalai.cli._dispatch import _build_lightning_parser, _dispatch  # noqa: PLC2701
from physicalai.cli._help import print_trainer_help  # noqa: PLC2701

if TYPE_CHECKING:
    from jsonargparse import ArgumentParser, Namespace

HELP = "Run prediction with a model."


def build_parser() -> ArgumentParser:
    """Build the ``predict`` parser.

    Returns:
        Parser for ``physicalai predict``.
    """
    return _build_lightning_parser("predict")


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Dispatch ``predict`` to ``Trainer.predict``.

    Args:
        parser: Parser used to instantiate components.
        cfg: Parsed configuration namespace.

    Returns:
        Process exit code.
    """
    return _dispatch("predict")(parser, cfg)


def print_help(prog: str) -> None:
    """Print lightweight help without building the full parser."""
    print_trainer_help(prog, description=HELP, method="predict")


def register() -> SubcommandSpec:
    """Return the ``predict`` subcommand spec.

    Returns:
        Registered spec for the shared CLI host.
    """
    return SubcommandSpec(name="predict", parser=build_parser(), dispatch=run, help=HELP)
