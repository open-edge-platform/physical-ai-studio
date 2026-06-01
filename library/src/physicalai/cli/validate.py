# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Register the ``physicalai validate`` subcommand."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

from physicalai.cli._dispatch import _build_lightning_parser, _dispatch  # noqa: PLC2701
from physicalai.cli._help import print_trainer_help  # noqa: PLC2701

if TYPE_CHECKING:
    from jsonargparse import ArgumentParser, Namespace

HELP = "Run validation on a model."


def build_parser() -> ArgumentParser:
    """Build the ``validate`` parser.

    Returns:
        Parser for ``physicalai validate``.
    """
    return _build_lightning_parser("validate")


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Dispatch ``validate`` to ``Trainer.validate``.

    Args:
        parser: Parser used to instantiate components.
        cfg: Parsed configuration namespace.

    Returns:
        Process exit code.
    """
    return _dispatch("validate")(parser, cfg)


def print_help(prog: str) -> None:
    """Print lightweight help without building the full parser."""
    print_trainer_help(prog, description=HELP, method="validate")


def register() -> SubcommandSpec:
    """Return the ``validate`` subcommand spec.

    Returns:
        Registered spec for the shared CLI host.
    """
    return SubcommandSpec(name="validate", parser=build_parser(), dispatch=run, help=HELP)
