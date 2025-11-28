# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""Training CLI using LightningCLI with jsonargparse.

This CLI provides 1-1 mapping between Python API and CLI configuration using
getiaction.train.Trainer by default (a Lightning Trainer subclass).

Key features:
- Use `model` for Lightning compatibility (refers to Policy in getiaction)
- Full class_path support for model (policy) and data
- Uses getiaction.train.Trainer by default (Lightning subclass with conveniences)
- All standard LightningCLI features (checkpointing, logging, callbacks)

Examples:
    # Train with YAML config file
    getiaction fit --config configs/train.yaml

    # Override config values from CLI
    getiaction fit \
        --config configs/train.yaml \
        --trainer.max_epochs 200 \
        --trainer.experiment_name my_experiment \
        --data.train_batch_size 64

    # Config file structure
    trainer:
      max_epochs: 100
      experiment_name: pusht_baseline  # getiaction convenience
      # All Lightning Trainer parameters available

    model:
      class_path: getiaction.policies.lerobot.ACT
      init_args:
        dim_model: 512

    data:
      class_path: getiaction.data.lerobot.LeRobotDataModule
      init_args:
        repo_id: lerobot/pusht

    # Generate config template
    getiaction fit --print_config

    # Fast development run
    getiaction fit \
        --config configs/train.yaml \
        --trainer.fast_dev_run true

Note:
    This CLI uses getiaction.train.Trainer, which is a Lightning Trainer subclass.
    You get all Lightning features plus getiaction conveniences (experiment_name,
    auto-callbacks). To use pure lightning.Trainer, use the `lightning` CLI directly.
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

    Since getiaction.train.Trainer is a subclass of lightning.Trainer, all Lightning
    Trainer features work normally. Users get getiaction conveniences (experiment_name,
    auto-callbacks) plus full Lightning compatibility.
    """


def cli() -> None:
    """Main CLI entry point using LightningCLI with getiaction.train.Trainer as default.

    This provides:
    - getiaction.train.Trainer by default (automatic validation handling, experiment naming)
    - Full jsonargparse class_path support for model and data components
    - 1-1 mapping between CLI and Python API
    - All Lightning Trainer features work (it's a Lightning Trainer subclass)

    Examples:
        # Use getiaction.train.Trainer with all its features
        getiaction fit --config configs/getiaction/act.yaml

        # Override trainer parameters from CLI
        getiaction fit \
          --config configs/getiaction/act.yaml \
          --trainer.experiment_name my_experiment \
          --trainer.max_epochs 200

        # Config file structure
        trainer:
          max_epochs: 100
          experiment_name: pusht_baseline  # getiaction convenience
          # All Lightning Trainer parameters available

    Note:
        The CLI uses getiaction.train.Trainer, which is a Lightning Trainer subclass.
        This means all Lightning features work, but you get getiaction conveniences like
        experiment_name and automatic callback injection. To use pure lightning.Trainer
        without these conveniences, use the `lightning` CLI directly.
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
