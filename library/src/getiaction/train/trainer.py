# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer with Lightning backend."""

# ruff: noqa: ANN401

from typing import Any

try:
    import lightning
except ImportError as e:
    msg = "Lightning not installed. Trainer requires Lightning.\nInstall with: pip install getiaction[train]"
    raise ImportError(msg) from e

from getiaction.train.callbacks import PolicyDatasetInteraction


class Trainer(lightning.Trainer):
    """Lightning Trainer subclass with getiaction-specific conveniences.

    This subclasses Lightning's Trainer to add:
    - Automatic PolicyDatasetInteraction callback injection
    - Better default directory structure (experiments/ instead of current directory)
    - Optional experiment naming for better organization

    By default, experiments are saved to:
        experiments/lightning_logs/version_N/     # If experiment_name=None
        experiments/{experiment_name}/version_N/   # If experiment_name provided

    This matches Lightning's behavior but with a cleaner default location.

    All Lightning Trainer features work normally. When paired with getiaction DataModules,
    validation is automatically skipped when no validation dataset is configured (DataModule
    returns empty dataloader).

    Examples:
        >>> # Basic usage - saves to experiments/lightning_logs/version_0/
        >>> trainer = Trainer(max_epochs=10)
        >>> trainer.fit(policy, datamodule)

        >>> # With custom experiment name - saves to experiments/pusht_act/version_0/
        >>> trainer = Trainer(max_epochs=10, experiment_name="pusht_act")
        >>> trainer.fit(policy, datamodule)

        >>> # Change root directory
        >>> trainer = Trainer(max_epochs=10, default_root_dir="my_experiments")
        >>> trainer.fit(policy, datamodule)
        >>> # Creates: my_experiments/lightning_logs/version_0/

        >>> # Disable logging (same as Lightning)
        >>> trainer = Trainer(max_epochs=10, logger=False)
        >>> trainer.fit(policy, datamodule)

        >>> # Use custom logger (same as Lightning)
        >>> from lightning.pytorch.loggers import WandbLogger
        >>> logger = WandbLogger(project="robot-learning")
        >>> trainer = Trainer(max_epochs=10, logger=logger)
        >>> trainer.fit(policy, datamodule)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        # getiaction-specific parameter
        experiment_name: str | None = None,
        # Hardware
        accelerator: str | Any = "auto",
        strategy: str | Any = "auto",
        devices: list[int] | str | int = "auto",
        num_nodes: int = 1,
        precision: Any = None,
        # Logging & Checkpointing
        logger: Any = None,  # Keep Lightning's default (None = auto-create TensorBoardLogger)
        callbacks: list | Any | None = None,
        default_root_dir: str | Any | None = "experiments",  # Changed from None to "experiments"
        enable_checkpointing: bool | None = None,
        enable_progress_bar: bool | None = None,
        enable_model_summary: bool | None = None,
        # Training control
        max_epochs: int | None = None,
        min_epochs: int | None = None,
        max_steps: int = -1,
        min_steps: int | None = None,
        max_time: Any = None,
        # Validation & Testing
        limit_train_batches: float | None = None,
        limit_val_batches: float | None = None,
        limit_test_batches: float | None = None,
        limit_predict_batches: float | None = None,
        val_check_interval: float | None = None,
        check_val_every_n_epoch: int | None = 1,
        num_sanity_val_steps: int | None = 0,  # Default to 0 for embodied AI
        # Optimization
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
        # Debugging & Development
        fast_dev_run: int | bool = False,
        overfit_batches: float = 0.0,
        log_every_n_steps: int | None = None,
        profiler: Any = None,
        detect_anomaly: bool = False,
        # Advanced
        deterministic: bool | Any | None = None,
        benchmark: bool | None = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        plugins: Any = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        barebones: bool = False,
        model_registry: str | None = None,
    ) -> None:
        """Initialize the getiaction Trainer with all Lightning Trainer parameters.

        This exposes all Lightning Trainer parameters for full control and IDE autocomplete support.
        The only modification is that num_sanity_val_steps defaults to 0 (instead of 2) since
        embodied AI training typically doesn't need sanity checks, and PolicyDatasetInteraction
        callback is automatically added.

        See Lightning Trainer documentation for detailed parameter descriptions:
        https://lightning.ai/docs/pytorch/stable/common/trainer.html

        Key Parameters:
            experiment_name: Name for this experiment (creates subdirectory in default_root_dir).
                           If provided, overrides Lightning's default "lightning_logs" name.
            default_root_dir: Root directory for experiments (default: "experiments" instead of current dir)
            accelerator: Hardware accelerator ('auto', 'cpu', 'gpu', 'tpu', 'ipu', 'mps')
            max_epochs: Maximum number of epochs to train
            logger: Logger instance. None (default) creates TensorBoardLogger automatically.
                   False disables logging. Or pass custom logger instance.
            callbacks: List of callbacks. PolicyDatasetInteraction is auto-added.
            num_sanity_val_steps: Number of validation sanity steps (default: 0)
            devices: Number/list of devices to use
            precision: Training precision ('32', '16', 'bf16', etc.)
            gradient_clip_val: Gradient clipping value
            log_every_n_steps: How often to log within training steps
            fast_dev_run: Quick run with 1 batch for debugging
        """
        # If experiment_name is provided and no custom logger, create TensorBoardLogger
        # This allows users to easily name their experiments while still using Lightning's patterns
        if experiment_name is not None and logger is None:
            from lightning.pytorch.loggers import TensorBoardLogger  # noqa: PLC0415

            logger = TensorBoardLogger(
                save_dir=default_root_dir if default_root_dir is not None else "experiments",
                name=experiment_name,
                default_hp_metric=False,
            )

        # Add PolicyDatasetInteraction callback automatically
        callbacks = [PolicyDatasetInteraction()] if callbacks is None else [*callbacks, PolicyDatasetInteraction()]

        # Call parent Lightning Trainer __init__ with all parameters
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            default_root_dir=default_root_dir,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            fast_dev_run=fast_dev_run,
            overfit_batches=overfit_batches,
            log_every_n_steps=log_every_n_steps,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            barebones=barebones,
            model_registry=model_registry,
        )
