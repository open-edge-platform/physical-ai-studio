# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Lightweight CLI help text for studio subcommands."""

from __future__ import annotations

_TRAIN_HELP = """usage: {prog} --config CONFIG [options]

{description}

options:
  -h, --help              Show this help message and exit.
  --config CONFIG         YAML/JSON config file.

Common config sections:
  model                   Policy subclass configuration.
  data                    DataModule subclass configuration.
  trainer                 Trainer configuration.
  {method:<23} Trainer.{method} method arguments.

Use --print_config with a complete command to inspect the full jsonargparse
schema for the selected model, data module, trainer, and method arguments.
"""

_BENCHMARK_HELP = """usage: {prog} --config CONFIG [options]

{description}

options:
  -h, --help              Show this help message and exit.
  --config CONFIG         YAML/JSON config file.
  --benchmark CLASS       Benchmark class path.
  --policy POLICY         Policy class path, for example physicalai.policies.ACT.
  --ckpt_path PATH        Checkpoint file or export directory.
  --output_dir DIR        Directory where benchmark results are written.

Use --print_config with a complete command to inspect the full jsonargparse
schema for the selected benchmark.
"""

_EXPORT_HELP = """usage: {prog} --config CONFIG [options]

{description}

options:
  -h, --help              Show this help message and exit.
  --config CONFIG         YAML/JSON config file.
  --policy POLICY         Policy class path, for example physicalai.policies.ACT.
  --ckpt_path PATH        Lightning checkpoint path.
  --backend BACKEND       Export backend: onnx, openvino, executorch, or torch.
  --output_dir DIR        Directory where export artifacts are written.
"""


def print_trainer_help(prog: str, *, description: str, method: str) -> None:
    """Print lightweight help for a Trainer-backed command."""
    print(_TRAIN_HELP.format(prog=prog, description=description, method=method))  # noqa: T201


def print_benchmark_help(prog: str, *, description: str) -> None:
    """Print lightweight help for ``benchmark``."""
    print(_BENCHMARK_HELP.format(prog=prog, description=description))  # noqa: T201


def print_export_help(prog: str, *, description: str) -> None:
    """Print lightweight help for ``export``."""
    print(_EXPORT_HELP.format(prog=prog, description=description))  # noqa: T201
