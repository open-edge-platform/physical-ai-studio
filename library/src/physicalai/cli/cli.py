# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""Unified CLI for physicalai extending LightningCLI with benchmark support.

This module provides `CLI` - a custom CLI class that extends LightningCLI
to add benchmark evaluation capabilities while preserving all training features.

Design Pattern:
    | Class       | Extends              | Purpose                              |
    |-------------|----------------------|--------------------------------------|
    | Config      | -                    | Training configuration               |
    | DataModule  | LightningDataModule  | Data loading                         |
    | Policy      | LightningModule      | Model definition                     |
    | Trainer     | lightning.Trainer    | Training orchestration               |
    | CLI         | LightningCLI         | Command-line interface               |

Key features:
- All LightningCLI subcommands: fit, validate, test, predict
- Additional `benchmark` subcommand for policy evaluation
- Uses physicalai.train.Trainer by default
- Full class_path support for model (policy), data, and benchmark

Examples:
    # Train with YAML config file
    physicalai fit --config configs/train.yaml

    # Override config values from CLI
    physicalai fit \
        --config configs/train.yaml \
        --trainer.max_epochs 200 \
        --data.train_batch_size 64

    # Run benchmark evaluation
    physicalai benchmark --config configs/benchmark/libero.yaml

    # Benchmark with CLI overrides (shorthand syntax)
    physicalai benchmark \
        --config configs/benchmark/libero.yaml \
        --benchmark.num_episodes 50 \
        --policy physicalai.policies.ACT \
        --ckpt_path ./checkpoints/model.ckpt

    # Generate config template
    physicalai fit --print_config
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from physicalai.data import DataModule
from physicalai.policies.base import Policy
from physicalai.train.trainer import Trainer

if TYPE_CHECKING:
    from physicalai.benchmark import Benchmark

logger = logging.getLogger(__name__)


class CLI(LightningCLI):
    r"""Custom CLI extending LightningCLI with benchmark evaluation support.

    This CLI provides a unified interface for both training and evaluation:
    - Training: fit, validate, test, predict (inherited from LightningCLI)
    - Evaluation: benchmark (added by CLI)

    The benchmark subcommand uses class_path pattern for benchmarks and
    simple string arguments for policy class and checkpoint path.

    Example:
        Command line usage:

            # Training
            physicalai fit --config configs/train.yaml

            # Benchmark evaluation with config
            physicalai benchmark --config configs/benchmark/libero.yaml

            # Benchmark with CLI arguments
            physicalai benchmark \\
                --benchmark physicalai.benchmark.LiberoBenchmark \\
                --benchmark.task_suite libero_10 \\
                --policy physicalai.policies.ACT \\
                --ckpt_path ./checkpoints/model.ckpt

        YAML configuration for benchmark:

            benchmark:
              class_path: physicalai.benchmark.LiberoBenchmark
              init_args:
                task_suite: libero_10
                num_episodes: 20

            policy: physicalai.policies.ACT
            ckpt_path: ./checkpoints/model.ckpt
            output_dir: ./results/benchmark
    """

    def __init__(
        self,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize CLI with benchmark support."""
        # Store benchmark-related state
        self._benchmark: Benchmark | None = None
        self._benchmark_pretrained: str | None = None
        self._benchmark_output_dir: Path | None = None
        self._benchmark_verbose: bool = False

        super().__init__(*args, **kwargs)

    def instantiate_classes(self) -> None:
        """Override to skip instantiation for benchmark subcommand.

        The benchmark subcommand uses different arguments (benchmark/policy)
        than training (model/datamodule), so we handle instantiation
        separately in _run_benchmark().
        """
        if self.subcommand == "benchmark":
            # Skip automatic instantiation for benchmark - handled in _run_benchmark()
            return
        super().instantiate_classes()

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Define available subcommands including benchmark.

        Returns:
            Dict mapping subcommand names to sets of arguments to skip.
            The benchmark subcommand skips model/data/trainer since it uses
            its own benchmark and pretrained arguments.
        """
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
            "benchmark": {"model", "dataloaders", "datamodule", "trainer"},
        }

    def _add_subcommands(self, parser: LightningArgumentParser, **kwargs: Any) -> None:  # noqa: ANN401
        """Add subcommands including custom benchmark command."""
        from lightning.pytorch.cli import (  # noqa: PLC0415
            _get_short_description,
            class_from_function,
        )

        self._subcommand_parsers: dict[str, LightningArgumentParser] = {}
        parser_subcommands = parser.add_subcommands()

        trainer_class = (
            self.trainer_class if isinstance(self.trainer_class, type) else class_from_function(self.trainer_class)
        )

        for subcommand in self.subcommands():
            if subcommand == "benchmark":
                # Custom handling for benchmark subcommand
                description = "Run benchmark evaluation on a trained policy."
                subparser_kwargs = kwargs.get(subcommand, {})
                subparser_kwargs.setdefault("description", description)
                subcommand_parser = self._create_benchmark_parser(**subparser_kwargs)
            else:
                # Standard LightningCLI handling for training subcommands
                fn = getattr(trainer_class, subcommand)
                description = _get_short_description(fn) or f"Run {subcommand}"
                subparser_kwargs = kwargs.get(subcommand, {})
                subparser_kwargs.setdefault("description", description)
                subcommand_parser = self._prepare_subcommand_parser(trainer_class, subcommand, **subparser_kwargs)

            self._subcommand_parsers[subcommand] = subcommand_parser
            parser_subcommands.add_subcommand(subcommand, subcommand_parser, help=description)

    def _create_benchmark_parser(self, **kwargs: Any) -> LightningArgumentParser:  # noqa: ANN401, PLR6301
        """Create argument parser for benchmark subcommand.

        Returns:
            Parser configured with benchmark, policy class, checkpoint, and output arguments.
        """
        from physicalai.benchmark import Benchmark  # noqa: PLC0415

        parser = LightningArgumentParser(**kwargs)

        # Add benchmark with class_path pattern (like model in training)
        parser.add_subclass_arguments(
            Benchmark,
            nested_key="benchmark",
            required=True,
            help="Benchmark configuration. Use class_path for specialized benchmarks.",
        )

        # Policy class path as simple string (required)
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

        # Output directory
        parser.add_argument(
            "--output_dir",
            type=str,
            default="./results/benchmark",
            help="Directory to save benchmark results.",
        )

        return parser

    def _run_subcommand(self, subcommand: str) -> None:
        """Run the chosen subcommand with custom benchmark handling."""
        if subcommand == "benchmark":
            self._run_benchmark()
        else:
            super()._run_subcommand(subcommand)

    @staticmethod
    def _load_policy(
        policy_path: str,
        ckpt_path: str | None,
    ) -> tuple[Any, str]:
        """Load policy from class path and optional checkpoint.

        Args:
            policy_path: Fully qualified policy class path (e.g., physicalai.policies.ACT)
            ckpt_path: Path to checkpoint file or export directory (optional)

        Returns:
            Tuple of (policy instance, device string)

        Raises:
            ImportError: If policy class cannot be imported.
            ValueError: If InferenceModel is used without ckpt_path.
        """
        import importlib  # noqa: PLC0415

        from physicalai.devices import get_available_device  # noqa: PLC0415
        from physicalai.inference import InferenceModel  # noqa: PLC0415

        module_path, class_name = policy_path.rsplit(".", 1)
        try:
            policy_class = getattr(importlib.import_module(module_path), class_name)
        except (ImportError, AttributeError) as e:
            msg = f"Could not import policy class '{policy_path}'"
            raise ImportError(msg) from e

        is_inference_model = policy_class is InferenceModel or (
            isinstance(policy_class, type) and issubclass(policy_class, InferenceModel)
        )

        if is_inference_model:
            if not ckpt_path:
                msg = "InferenceModel requires --ckpt_path pointing to export directory"
                raise ValueError(msg)
            policy = InferenceModel.load(ckpt_path)
            return policy, policy.device

        device = get_available_device()
        if ckpt_path:
            policy = policy_class.load_from_checkpoint(ckpt_path)
        else:
            logger.warning("No checkpoint provided - using randomly initialized policy")
            policy = policy_class()

        policy.to(device)
        policy.eval()
        return policy, device

    def _run_benchmark(self) -> None:
        """Execute benchmark evaluation.

        Raises:
            ValueError: If benchmark configuration is not found.
        """
        from jsonargparse import Namespace  # noqa: PLC0415

        if self.subcommand is None:
            msg = "No subcommand specified"
            raise ValueError(msg)
        config = self.config.get(self.subcommand)
        if config is None:
            msg = "Benchmark configuration not found"
            raise ValueError(msg)

        benchmark_parser = self._subcommand_parsers["benchmark"]
        benchmark = benchmark_parser.instantiate_classes(
            Namespace(benchmark=config.benchmark),
        ).benchmark

        policy, device = self._load_policy(config.policy, config.ckpt_path)

        logger.info("Benchmark: %s", benchmark)
        logger.info("Policy: %s", type(policy).__name__)
        logger.info("Device: %s", device)

        try:
            results = benchmark.evaluate(policy=policy)

            print(results.summary())  # noqa: T201

            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results.to_json(output_dir / "results.json")
            results.to_csv(output_dir / "results.csv")
            logger.info("Results saved to %s", output_dir)
        finally:
            for gym in benchmark.gyms:
                gym.close()


def cli() -> None:
    """Entry point for physicalai CLI.

    Creates and runs CLI with default configuration for
    physicalai policies, data modules, and trainer.
    """
    CLI(
        model_class=Policy,
        datamodule_class=DataModule,
        trainer_class=Trainer,
        save_config_callback=None,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    cli()
