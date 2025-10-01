# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""Training CLI using LightningCLI with jsonargparse.

This CLI automatically parses configurations from:
- YAML/JSON files (--config)
- CLI arguments (--model.init_args.*)
- Python dataclasses (via from_config methods)
- Pydantic models (with automatic validation)

The jsonargparse `class_path` pattern enables dynamic class instantiation
from configuration files without code changes.

Examples:
    # Train with YAML config file
    getiaction fit --config configs/train.yaml

    # Override config values from CLI
    getiaction fit \
        --config configs/train.yaml \
        --trainer.max_epochs 200 \
        --data.init_args.train_batch_size 64

    # Specify classes directly from CLI
    getiaction fit \
        --model getiaction.policies.dummy.policy.Dummy \
        --data getiaction.data.lerobot.LeRobotDataModule

    # Generate config template
    getiaction fit --print_config

    # Fast development run
    getiaction fit \
        --config configs/train.yaml \
        --trainer.fast_dev_run=true

    # Alternative: Use as Python module
    python -m getiaction.cli.cli fit --config configs/train.yaml
"""

from lightning.pytorch.cli import LightningCLI

from getiaction.data import DataModule
from getiaction.policies.base import Policy


def cli() -> None:
    """Main CLI entry point.

    Creates a LightningCLI instance that automatically:
    - Parses YAML/JSON configs and CLI arguments
    - Validates configurations using type hints
    - Instantiates classes using the class_path pattern
    - Supports subclasses of Policy and DataModule
    """
    LightningCLI(
        model_class=Policy,
        datamodule_class=DataModule,
        save_config_callback=None,
        subclass_mode_model=True,  # Allow any Policy subclass
        subclass_mode_data=True,  # Allow any DataModule subclass
    )


if __name__ == "__main__":
    cli()
