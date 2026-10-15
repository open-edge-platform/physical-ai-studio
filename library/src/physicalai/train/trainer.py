# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer with Lightning backend."""

# ruff: noqa: ANN401

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence  # noqa: TC003
from pathlib import Path
from typing import Any, ClassVar

import lightning
import torch
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateMonitor
from lightning.pytorch.loggers import Logger  # noqa: TC002
from lightning.pytorch.strategies import DDPStrategy
from physicalai.config import instantiate

from physicalai.engine import Capabilities, Enablement, Engine, Run, TrainHooks, WorkerEnv
from physicalai.train.callbacks import PolicyDatasetInteraction


class Trainer(Engine, lightning.Trainer):
    """Lightning Trainer subclass with physicalai-specific conveniences.

    This subclasses Lightning's Trainer to add:
    - Automatic PolicyDatasetInteraction callback injection
    - Automatic LearningRateMonitor callback injection (per-step logging)
    - Better default directory structure (experiments/ instead of current directory)
    - Optional experiment naming for better organization

    By default, experiments are saved to:
        experiments/lightning_logs/version_N/     # If experiment_name=None
        experiments/{experiment_name}/version_N/   # If experiment_name provided

    This matches Lightning's behavior but with a cleaner default location.

    All Lightning Trainer features work normally. When paired with physicalai DataModules,
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

    framework: ClassVar[str] = "physicalai"
    env: ClassVar[WorkerEnv | None] = None
    enablements: ClassVar[tuple[Enablement, ...]] = ()
    training_accelerators: ClassVar[tuple[str, ...]] = ("cpu", "cuda", "xpu")

    def __init__(  # noqa: PLR0913
        self,
        *,
        # physicalai-specific parameters
        experiment_name: str | None = None,
        auto_scale_batch_size: bool = False,
        allow_tf32: bool = False,
        cudnn_benchmark: bool = True,
        # Hardware
        accelerator: str | Any = "auto",
        strategy: str | Any = "auto",
        devices: list[int] | str | int = "auto",
        num_nodes: int = 1,
        precision: Any = None,
        # Logging & Checkpointing
        logger: Logger | Iterable[Logger] | bool | None = None,  # Keep Lightning's default (None = auto TB logger)
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
        val_check_interval: int | float | None = None,  # noqa: PYI041
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
        """Initialize the physicalai Trainer with all Lightning Trainer parameters.

        This exposes all Lightning Trainer parameters for full control and IDE autocomplete support.
        The only modification is that num_sanity_val_steps defaults to 0 (instead of 2) since
        embodied AI training typically doesn't need sanity checks, and PolicyDatasetInteraction
        callback is automatically added.

        See Lightning Trainer documentation for detailed parameter descriptions:
        https://lightning.ai/docs/pytorch/stable/common/trainer.html

        Key Parameters:
            experiment_name: Name for this experiment (creates subdirectory in default_root_dir).
                           If provided, overrides Lightning's default "lightning_logs" name.
            auto_scale_batch_size: Add a ``BatchSizeFinder`` callback to find the
                largest batch size that fits in memory before training.
                ``False`` (default) disables it.  ``True`` enables exponential (power) scaling.
            allow_tf32: Enable TF32 precision for matmul on Ampere+ GPUs. Provides ~3x speedup
                for float32 operations with minimal accuracy loss. Defaults to False to match
                PyTorch's default. Consider enabling when using 32-true precision.
            cudnn_benchmark: Enable cuDNN benchmark mode to auto-tune convolution algorithms.
                Speeds up training when input sizes are constant. Defaults to True.
            default_root_dir: Root directory for experiments (default: "experiments" instead of current dir)
            accelerator: Hardware accelerator ('auto', 'cpu', 'gpu', 'tpu', 'ipu', 'mps')
            max_epochs: Maximum number of epochs to train
            logger: Logger instance. None (default) creates TensorBoardLogger automatically.
                   False disables logging. Or pass custom logger instance.
            callbacks: List of callbacks. PolicyDatasetInteraction is auto-added, and a
                LearningRateMonitor logging per step is added unless one is already provided.
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

        user_callbacks: list[Any]
        if callbacks is None:
            user_callbacks = []
        elif isinstance(callbacks, (list, tuple)):
            user_callbacks = list(callbacks)
        else:
            user_callbacks = [callbacks]

        # jsonargparse does not always instantiate callback list entries when the
        # parameter type is broad (list | Any). Support callback specs in YAML.
        normalized_callbacks: list[Any] = []
        for callback in user_callbacks:
            if isinstance(callback, dict) and "class_path" in callback:
                normalized_callbacks.append(instantiate(callback))
            else:
                normalized_callbacks.append(callback)

        callbacks = [*normalized_callbacks, PolicyDatasetInteraction()]

        # LearningRateMonitor raises if the trainer has no logger, so skip it when logging is disabled.
        logging_enabled = logger is not False and not barebones
        if logging_enabled and not any(isinstance(callback, LearningRateMonitor) for callback in callbacks):
            callbacks.append(LearningRateMonitor(logging_interval="step"))

        if auto_scale_batch_size:
            callbacks.append(BatchSizeFinder(mode="power"))

        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # When using auto strategy with multiple GPUs, default to DDP with
        # find_unused_parameters=True. Policies like Pi05 freeze large portions
        # of the model, so not all parameters participate in every training step.
        multi_gpu = (isinstance(devices, int) and devices > 1) or (isinstance(devices, list) and len(devices) > 1)
        if strategy == "auto" and multi_gpu:
            strategy = DDPStrategy(find_unused_parameters=True)

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

    @classmethod
    def cli(
        cls,
        argv: Sequence[str],
        *,
        interpreter: str | Path | None = None,
    ) -> int:
        """Alias the first-party native CLI to the existing ``fit`` command.

        Returns:
            The ``fit`` command's exit code.
        """
        del cls, interpreter
        from physicalai.cli.main import main  # noqa: PLC0415

        return main(["fit", *argv])

    @classmethod
    def run(
        cls,
        settings: Mapping[str, Any],
        *,
        dataset: Path,
        output_dir: Path,
        device: str,
        hooks: TrainHooks | None = None,
        interpreter: str | Path | None = None,
    ) -> Run:
        """Keep first-party Studio jobs on their established job runner for now."""
        del cls, settings, dataset, output_dir, device, hooks, interpreter
        msg = "First-party Studio training still uses run_training_job(); standalone Engine.run() is not available."
        raise NotImplementedError(msg)

    @classmethod
    def export(
        cls,
        source: Any,
        output_path: str | Path,
        backend: str | None = "openvino",
        *,
        policy_class: str | type[Any] | None = None,
        input_sample: dict[str, torch.Tensor] | None = None,
        post_export_hooks: list[Callable[[str], None]] | None = None,
        **export_kwargs: Any,
    ) -> Path:
        """Export a first-party policy or checkpoint for Runtime inference.

        Returns:
            The export destination.

        Raises:
            TypeError: If the selected policy does not implement ``export()``.
            ValueError: If a checkpoint is exported without a policy class or the run belongs to another engine.
        """
        from physicalai.policies.base.policy import Policy  # noqa: PLC0415

        if isinstance(source, Policy):
            policy = source
        else:
            checkpoint = _checkpoint_path(source, cls.framework)
            if policy_class is None:
                msg = "policy_class is required when exporting a checkpoint path."
                raise ValueError(msg)
            resolved_class = _resolve_policy_class(policy_class, Policy)
            # Lightning deserializes checkpoint state, so this path must be a trusted local artifact.
            policy = resolved_class.load_from_checkpoint(str(checkpoint))

        export_policy = getattr(policy, "export", None)
        if not callable(export_policy):
            msg = f"Policy {type(policy).__name__} does not implement export()."
            raise TypeError(msg)
        export_policy(
            output_path,
            backend or "openvino",
            input_sample=input_sample,
            post_export_hooks=post_export_hooks,
            **export_kwargs,
        )
        return Path(output_path)

    @classmethod
    def policies(cls) -> list[str]:
        """List first-party policies without importing their model modules.

        Returns:
            First-party policy identifiers.
        """
        from physicalai.policies import list_physicalai_policies  # noqa: PLC0415

        return list_physicalai_policies()

    @classmethod
    def config_schema(cls, policy: str) -> dict[str, Any]:
        """Return a JSON Schema for one first-party policy config dataclass.

        Raises:
            ValueError: If the policy or its config type is unavailable.
        """
        from importlib import import_module  # noqa: PLC0415

        from pydantic import TypeAdapter  # noqa: PLC0415

        from physicalai.policies import get_physicalai_policy_class  # noqa: PLC0415

        policy_class = get_physicalai_policy_class(policy)
        module_name = policy_class.__module__.rsplit(".", maxsplit=1)[0]
        if not module_name.startswith("physicalai.policies."):
            msg = f"Policy {policy!r} resolved outside the first-party policy package."
            raise ValueError(msg)
        config_module = import_module(f"{module_name}.config")
        config_class_name = f"{policy_class.__name__}Config"
        try:
            config_class = getattr(config_module, config_class_name)
        except AttributeError as exc:
            msg = f"Policy {policy!r} does not expose a {config_class_name} config type."
            raise ValueError(msg) from exc
        return TypeAdapter(config_class).json_schema()

    @classmethod
    def capabilities(cls, policy: str) -> Capabilities:
        """Return accelerators and export formats supported by one first-party policy."""
        from physicalai.policies import get_physicalai_policy_class  # noqa: PLC0415

        policy_class = get_physicalai_policy_class(policy)
        get_backends = getattr(policy_class, "get_supported_export_backends", None)
        export_formats = tuple(sorted(str(backend) for backend in get_backends())) if callable(get_backends) else ()
        return Capabilities(
            training_accelerators=cls.training_accelerators,
            export_formats=export_formats,
        )


def _checkpoint_path(source: object, framework: str) -> Path:
    if isinstance(source, Run):
        if source.framework != framework:
            msg = f"Run belongs to framework {source.framework!r}, not {framework!r}."
            raise ValueError(msg)
        if source.checkpoint is None:
            msg = "Run has no native checkpoint to export."
            raise ValueError(msg)
        return source.checkpoint
    if isinstance(source, (str, Path)):
        return Path(source)
    msg = f"Unsupported first-party export source {type(source).__name__}."
    raise TypeError(msg)


def _resolve_policy_class(policy_class: str | type[Any], policy_base: type[Any]) -> type[Any]:
    from physicalai.policies import get_physicalai_policy_class, list_physicalai_policies  # noqa: PLC0415

    if isinstance(policy_class, type):
        if not issubclass(policy_class, policy_base):
            msg = f"policy_class must subclass {policy_base.__name__}."
            raise TypeError(msg)
        return policy_class

    for name in list_physicalai_policies():
        candidate = get_physicalai_policy_class(name)
        class_path = f"{candidate.__module__}.{candidate.__name__}"
        if policy_class in {name, class_path}:
            return candidate
    msg = f"Unknown first-party policy class {policy_class!r}."
    raise ValueError(msg)
