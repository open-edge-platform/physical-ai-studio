# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""Training CLI using LightningCLI with jsonargparse.

This CLI provides 1-1 mapping between Python API and CLI configuration with full
support for both Lightning and getiaction Trainers via class_path pattern.

Key features:
- Use `model` for Lightning compatibility (refers to Policy in getiaction)
- Full class_path support for model (policy), data, AND trainer
- Compatible with both lightning.Trainer and getiaction.train.Trainer
- All standard LightningCLI features (checkpointing, logging, callbacks)

The jsonargparse `class_path` pattern enables dynamic class instantiation
from configuration files without code changes.

Examples:
    # Train with YAML config file
    getiaction fit --config configs/train.yaml

    # Override config values from CLI
    getiaction fit \
        --config configs/train.yaml \
        --trainer.init_args.max_epochs 200 \
        --data.init_args.train_batch_size 64

    # Use getiaction Trainer via class_path in config
    trainer:
      class_path: getiaction.train.Trainer
      init_args:
        max_epochs: 100

    # Use Lightning Trainer with logger
    trainer:
      class_path: lightning.Trainer
      init_args:
        max_epochs: 100
        limit_val_batches: 0
        logger:
          class_path: lightning.pytorch.loggers.TensorBoardLogger
          init_args:
            save_dir: experiments
            name: my_experiment

    # Generate config template
    getiaction fit --print_config

    # Fast development run
    getiaction fit \
        --config configs/train.yaml \
        --trainer.fast_dev_run true
"""

from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from getiaction.data import DataModule
from getiaction.policies.base import Policy
from getiaction.train.trainer import Trainer


class GetiActionCLI(LightningCLI):
    """Custom CLI for getiaction using getiaction.train.Trainer by default.

    This extends LightningCLI to use getiaction.train.Trainer as the default trainer,
    which provides automatic validation handling and other conveniences for embodied AI.

    Users can still override to use Lightning Trainer via class_path in config:
        trainer:
          class_path: lightning.Trainer
          init_args:
            max_epochs: 100
            limit_val_batches: 0
    """


def cli() -> None:
    """Main CLI entry point using LightningCLI with getiaction.train.Trainer as default.

    This provides:
    - getiaction.train.Trainer by default (automatic validation handling)
    - Full jsonargparse class_path support for all components
    - 1-1 mapping between CLI and Python API
    - Users can override trainer via class_path in config

    Examples:
        # Use default getiaction.train.Trainer (simple config)
        getiaction fit --config configs/act.yaml

        # Override to Lightning Trainer (class_path in config)
        trainer:
          class_path: lightning.Trainer
          init_args:
            max_epochs: 100
            limit_val_batches: 0
    """
    GetiActionCLI(
        model_class=Policy,
        datamodule_class=DataModule,
        trainer_class=Trainer,  # Use getiaction.train.Trainer by default
        save_config_callback=None,
        subclass_mode_model=True,  # Allow any Policy subclass via class_path
        subclass_mode_data=True,  # Allow any DataModule subclass via class_path
        auto_configure_optimizers=False,  # Policies configure their own optimizers
    )


if __name__ == "__main__":
    cli()
