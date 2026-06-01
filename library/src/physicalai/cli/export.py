# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Register the ``physicalai export`` subcommand."""

from __future__ import annotations

import logging

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

from physicalai.cli._help import print_export_help  # noqa: PLC2701
from physicalai.cli._policy import load_policy  # noqa: PLC2701

logger = logging.getLogger(__name__)

HELP = "Export a trained policy for deployment."


def build_parser() -> ArgumentParser:
    """Build the ``export`` parser.

    Returns:
        Parser for ``physicalai export``.
    """
    parser = ArgumentParser(prog="physicalai export", description=HELP)
    parser.add_argument("--config", action=ActionConfigFile, help="YAML/JSON config file.")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Policy class path (e.g., physicalai.policies.ACT).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="Export backend: onnx, openvino, executorch, or torch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where export artifacts are written.",
    )
    return parser


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Dispatch ``export`` by loading a policy and invoking its export contract.

    Args:
        parser: Parser used for the export subcommand.
        cfg: Parsed configuration namespace.

    Returns:
        Process exit code.

    Raises:
        TypeError: If the loaded policy does not implement ``export()``.
    """
    del parser
    policy, device = load_policy(cfg.policy, cfg.ckpt_path)
    if not hasattr(policy, "export"):
        msg = f"Policy '{cfg.policy}' does not support export()."
        raise TypeError(msg)

    logger.info("Policy: %s", type(policy).__name__)
    logger.info("Device: %s", device)
    policy.export(cfg.output_dir, backend=cfg.backend)
    logger.info("Export saved to %s", cfg.output_dir)
    return 0


def print_help(prog: str) -> None:
    """Print lightweight help without building the full parser."""
    print_export_help(prog, description=HELP)


def register() -> SubcommandSpec:
    """Return the ``export`` subcommand spec.

    Returns:
        Registered spec for the shared CLI host.
    """
    return SubcommandSpec(name="export", parser=build_parser(), dispatch=run, help=HELP)
