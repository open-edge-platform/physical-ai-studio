# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""One training run, described by a spec and executed by one function.

:class:`TrainingJobSpec` is the complete configuration of a training run —
 policy, epoch budget, batch size, precision, device — and nothing else. It holds
no paths, no job identity, and no transport details, so the same spec can be
built in-process or sent over HTTP to a remote trainer and mean the same thing
in both places.

:func:`run_training_job` executes a spec: it builds the datamodule and policy,
runs Lightning, saves the final checkpoint into ``output_dir``, and exports the
policy to every backend it supports. Where the data comes from, where the
result goes, how progress is reported, and how cancellation is signalled are all
arguments — the runner owns none of that policy.

Example:
    >>> spec = TrainingJobSpec(policy="act", max_epochs=5, batch_size=8)
    >>> run_training_job(  # doctest: +SKIP
    ...     spec,
    ...     dataset_root="/data/snapshot",
    ...     output_dir="/models/abc",
    ...     cache_dir="/cache/abc",
    ...     report=lambda progress, message, extra: None,
    ...     should_stop=lambda: False,
    ... )
"""

from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from physicalai.policies.base import Policy
    from physicalai.train.callbacks import ReportFn, StopFn

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "model.ckpt"
"""Filename of the final checkpoint written into ``output_dir``."""

EXPORTS_DIRNAME = "exports"
"""Subdirectory of ``output_dir`` holding one directory per export backend."""

_DATASET_REPO_ID = "snapshot"
"""Placeholder repo id: datasets are always loaded from a local root here."""

PRETRAINED_BASE_CHECKPOINTS: dict[str, str] = {
    "pi05": "lerobot/pi05_base",
    "smolvla": "lerobot/smolvla_base",
}
"""Hub checkpoints used to initialize policies that only fine-tune from pretrained weights."""

_WEIGHTS_ONLY_RESUME_POLICIES = frozenset({"pi0"})
"""Policies whose checkpoints must be reloaded with ``weights_only=True``."""

_COMPILED_EXPORT_RELOAD_POLICIES = frozenset({"act", "smolvla"})
"""Policies that cannot be exported while ``torch.compile``d, so are reloaded first."""


class TrainingJobSpec(BaseModel):
    """Everything needed to train one policy, and nothing else.

    Deliberately free of paths, identifiers, and transport concerns: this is
    the shared contract between an in-process runner and a remote trainer, so
    it must describe *what* to train rather than *where*. Runner-local
    concerns (dataset location, output location, resume checkpoint) are
    arguments to :func:`run_training_job`.

    Example:
        >>> spec = TrainingJobSpec(policy="act", max_epochs=5)
        >>> spec.precision
        'bf16-mixed'
    """

    model_config = ConfigDict(extra="forbid")

    policy: str = Field(description="Policy name, e.g. 'act', 'pi0', 'pi05', 'smolvla', 'groot'.")
    policy_source: Literal["physicalai", "lerobot"] = Field(
        default="physicalai",
        description="Which implementation of the policy to train.",
    )
    max_epochs: int = Field(default=5, ge=1, description="Training epoch budget.")
    batch_size: int = Field(default=8, ge=1, description="Training batch size.")
    num_workers: int | Literal["auto"] = Field(default="auto", description="Dataloader worker count.")
    val_split: float = Field(default=0.1, ge=0.0, lt=1.0, description="Fraction of episodes held out for validation.")
    precision: str = Field(default="bf16-mixed", description="Lightning precision, e.g. '32-true' or 'bf16-mixed'.")
    compile_model: bool = Field(default=False, description="Whether to torch.compile the policy forward pass.")
    auto_scale_batch_size: bool = Field(default=False, description="Whether to search for the largest fitting batch.")
    device_type: str | None = Field(
        default=None,
        description="Accelerator to train on ('xpu', 'cuda', 'cpu', ...). None auto-detects.",
    )
    device_index: int | None = Field(
        default=None,
        ge=0,
        description="Zero-based index of the accelerator to train on. None lets Lightning pick one.",
    )


def build_policy(spec: TrainingJobSpec, *, resume_from: Path | str | None = None) -> Policy:
    """Construct the policy a spec describes.

    Args:
        spec: The training configuration.
        resume_from: Checkpoint to continue training from. When None the policy
            is initialized fresh (from its pretrained base weights where the
            policy requires them, see :data:`PRETRAINED_BASE_CHECKPOINTS`).

    Returns:
        The policy, compiled when ``spec.compile_model`` is set.
    """
    if resume_from is not None:
        return _load_policy_from_checkpoint(spec, Path(resume_from))

    from physicalai.policies import get_policy

    kwargs: dict[str, Any] = {"compile_model": spec.compile_model}
    if spec.policy_source == "physicalai":
        pretrained = PRETRAINED_BASE_CHECKPOINTS.get(spec.policy.lower())
        if pretrained is not None:
            kwargs["pretrained_name_or_path"] = pretrained
    return get_policy(spec.policy, source=spec.policy_source, **kwargs)


def run_training_job(
    spec: TrainingJobSpec,
    *,
    dataset_root: Path | str,
    output_dir: Path | str,
    cache_dir: Path | str,
    report: ReportFn,
    should_stop: StopFn,
    resume_from: Path | str | None = None,
) -> None:
    """Train one policy end to end: fit, checkpoint, and export.

    On success ``output_dir`` holds a checkpoint (:data:`CHECKPOINT_NAME`),
    the Lightning CSV logs, and an :data:`EXPORTS_DIRNAME` directory per
    successful export backend. That checkpoint is normally the best epoch,
    written by the ``ModelCheckpoint`` callback during ``fit``; this function
    only saves a checkpoint of its own as a fallback when the callback did
    not produce one (e.g. the monitored metric never logged), so it never
    overwrites a best checkpoint with the final-epoch weights (see
    :func:`_ensure_checkpoint_exists`).

    Cancellation is cooperative. ``should_stop`` is polled throughout training
    via :class:`~physicalai.train.callbacks.ProgressReportingCallback`; when it
    returns True this function stops after the fit loop unwinds and returns
    *without* writing a checkpoint or exporting, since a partially trained run
    has no artifact worth keeping. Callers distinguish a canceled run from a
    completed one by consulting ``should_stop`` again after this returns.

    Before export, the trainer, datamodule, and their references back into the
    policy are dropped and accelerator memory is released (see
    :func:`_detach_trainer` and :func:`_release_memory`): the trainer's
    optimizer state and dataloaders would otherwise stay resident throughout
    export, and export conversion (OpenVINO in particular) can need several GB
    on top of that.

    Args:
        spec: What to train.
        dataset_root: Local root of the LeRobot dataset to train on.
        output_dir: Destination for the trained model. Replaced if it exists.
        cache_dir: Scratch directory for checkpoints and logs during training;
            moved to ``output_dir`` on success.
        report: Telemetry sink, called with ``(progress, message, extra_info)``.
        should_stop: Cooperative cancellation probe.
        resume_from: Checkpoint to continue training from, e.g. a previously
            trained model's ``model.ckpt``. None trains from scratch.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from physicalai.data import LeRobotDataModule
    from physicalai.train.callbacks import ProgressReportingCallback
    from physicalai.train.trainer import Trainer

    from training.device import resolve_accelerator, resolve_devices, resolve_strategy

    accelerator = resolve_accelerator(spec.device_type)
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    datamodule = LeRobotDataModule(
        repo_id=_DATASET_REPO_ID,
        root=str(dataset_root),
        train_batch_size=spec.batch_size,
        num_workers=spec.num_workers,
        val_split=spec.val_split,
    )
    policy = build_policy(spec, resume_from=resume_from)

    trainer = Trainer(
        logger=CSVLogger(cache_dir.parent, name=cache_dir.stem),
        callbacks=[
            ModelCheckpoint(
                dirpath=cache_dir,
                filename=Path(CHECKPOINT_NAME).stem,
                save_top_k=1,
                monitor="val/loss",
                mode="min",
            ),
            ProgressReportingCallback(report=report, should_stop=should_stop),
        ],
        accelerator=accelerator,
        strategy=resolve_strategy(spec.device_type),
        devices=resolve_devices(spec.device_index),
        max_epochs=spec.max_epochs,
        auto_scale_batch_size=spec.auto_scale_batch_size,
        precision=spec.precision,
        check_val_every_n_epoch=1,
    )

    report(0, "Training model", {})
    trainer.fit(model=policy, datamodule=datamodule)
    if should_stop():
        logger.info("Training canceled; skipping checkpoint and export")
        return

    _ensure_checkpoint_exists(trainer, cache_dir)
    _publish(cache_dir, output_dir)

    export_policy = _export_policy(spec, policy, output_dir)
    _detach_trainer(export_policy, trainer)
    del trainer, datamodule, policy
    _release_memory()
    _export(export_policy, output_dir, report)


def _load_policy_from_checkpoint(spec: TrainingJobSpec, checkpoint: Path) -> Policy:
    """Restore a policy from a checkpoint, applying ``spec.compile_model``.

    Returns:
        The restored policy.
    """
    if spec.policy_source == "lerobot":
        from physicalai.policies.lerobot import LeRobotPolicy

        policy: Policy = LeRobotPolicy.load_from_checkpoint(checkpoint)
    else:
        from physicalai.policies import get_physicalai_policy_class

        policy_class = get_physicalai_policy_class(spec.policy)
        # Some policies store non-tensor objects Lightning cannot unpickle
        # safely by default; those are loaded weights-only.
        kwargs: dict[str, Any] = {"weights_only": True} if spec.policy.lower() in _WEIGHTS_ONLY_RESUME_POLICIES else {}
        policy = policy_class.load_from_checkpoint(str(checkpoint), **kwargs)

    if spec.compile_model:
        import torch

        compile_mode = getattr(policy.config, "compile_mode", "default")
        policy.forward = torch.compile(policy.forward, mode=compile_mode)  # type: ignore[method-assign]
    return policy


def _ensure_checkpoint_exists(trainer: Any, cache_dir: Path) -> None:
    """Save a final checkpoint only if the ``ModelCheckpoint`` callback did not already write one.

    The callback is the preferred source of :data:`CHECKPOINT_NAME`: it tracks
    the best monitored epoch, not just the last one. But it only saves when
    its monitored metric was actually logged (e.g. a run with no validation
    split never logs ``val/loss``), so this is a fallback for that case —
    saving the trainer's current (final-epoch) weights — rather than a second
    source of truth that could overwrite the callback's best checkpoint.
    """
    checkpoint = cache_dir / CHECKPOINT_NAME
    if not checkpoint.is_file():
        logger.warning("ModelCheckpoint callback did not write a checkpoint; saving final weights instead")
        trainer.save_checkpoint(checkpoint)


def _publish(cache_dir: Path, output_dir: Path) -> None:
    """Move the finished training cache into its final location."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.move(str(cache_dir), str(output_dir))


def _export_policy(spec: TrainingJobSpec, policy: Policy, output_dir: Path) -> Policy:
    """Return the policy to export, reloading it when compilation blocks export.

    ``torch.compile`` wraps ``forward`` in a way some export backends cannot
    trace, so for the affected policies the just-saved checkpoint is reloaded
    uncompiled. Falls back to the trained policy if that reload fails — a
    failed export is better than a failed job.

    Returns:
        The policy instance to hand to the export backends.
    """
    if not (spec.compile_model and spec.policy.lower() in _COMPILED_EXPORT_RELOAD_POLICIES):
        return policy
    try:
        logger.info("Reloading non-compiled policy for export")
        uncompiled = spec.model_copy(update={"compile_model": False})
        return _load_policy_from_checkpoint(uncompiled, output_dir / CHECKPOINT_NAME)
    except Exception:  # reload is best-effort; the trained policy is a valid fallback
        logger.warning("Failed to reload non-compiled policy for export; using trained policy", exc_info=True)
        return policy


def _detach_trainer(export_policy: Policy, trainer: Any) -> None:
    """Break the trainer<->policy<->datamodule reference cycle before export.

    Lightning wires ``policy._trainer = trainer``, ``trainer.datamodule =
    datamodule``, and ``trainer.strategy._lightning_module = policy`` during
    ``fit``, and never undoes it. When ``export_policy`` is the very policy
    that was just trained (the common case: no ``torch.compile`` reload), it
    still holds that ``_trainer`` reference, which keeps the trainer — and
    everything it holds: optimizer state, dataloaders, the strategy — alive
    and reachable no matter how many local names ``run_training_job`` deletes.
    ``gc.collect()`` only reclaims *unreachable* cycles, so without this the
    memory release below is a no-op on the export object it matters most for.

    Best-effort: a failure here must not abort the job.
    """
    try:
        export_policy._trainer = None
        if getattr(trainer, "strategy", None) is not None:
            trainer.strategy._lightning_module = None
        trainer.datamodule = None
    except Exception as exc:
        logger.warning("Could not detach trainer from policy: %s", exc)


def _release_memory() -> None:
    """Return held host and device memory to the OS/allocator before the next stage.

    Export conversion (OpenVINO in particular) allocates several GB above the
    model's resident size, on top of whatever the just-finished trainer and
    dataloaders still hold. A garbage-collection pass plus clearing every
    available accelerator's cache prevents that peak from stacking on top of
    memory this process no longer needs. Best-effort: a failure here must not
    abort the job.
    """
    gc.collect()
    try:
        import torch

        if torch.xpu.is_available():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if torch.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Could not release device cache: %s", exc)


def _export(policy: Policy, output_dir: Path, report: ReportFn) -> None:
    """Export the policy to every backend it declares support for."""
    from physicalai.export import ExportablePolicyMixin

    if not isinstance(policy, ExportablePolicyMixin):
        logger.info("Skipping export: policy does not support export backends")
        return

    for backend in policy.get_supported_export_backends():
        name = backend.value if hasattr(backend, "value") else str(backend)
        try:
            logger.info("Exporting model to %s format", name)
            report(99, f"Exporting to {name} format", {})
            policy.export(output_dir / EXPORTS_DIRNAME / name, backend=backend)
        except ImportError as exc:
            # An optional backend dependency is missing (e.g. executorch on xpu
            # builds). Skip it without a traceback so the run isn't mistaken
            # for a failure; the remaining backends still export.
            logger.warning("Skipping %s export: optional dependency missing (%s)", name, exc)
        except Exception:
            # Export is best-effort: one failing backend must not abort the job.
            logger.exception("Failed exporting model to %s format", name)
        finally:
            # Conversion peaks well above the model's resident size, so release
            # after every backend — not just before the first one.
            _release_memory()
