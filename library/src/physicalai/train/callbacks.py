# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Callbacks for training."""

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import lightning as L  # noqa: N812
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only, rank_zero_warn

from physicalai.train.utils import reformat_dataset_to_match_policy

logger = logging.getLogger(__name__)

ReportFn = Callable[[int, str | None, dict[str, object]], None]
"""Telemetry sink: ``(progress, message, extra_info)`` with progress in 0-100."""

StopFn = Callable[[], bool]
"""Cooperative cancellation probe: returns True when training should stop."""

SNAPFLOW_PROGRESS_BAR_KEY = "snapflow"
"""Progress-bar metric key logged while SnapFlow distillation is active.

The key is only logged once SnapFlow has been activated, so its *presence* in
the progress bar is the phase-2 indicator: the bar stays clean during phase 1.
"""

SNAPFLOW_CHECKPOINT_KEY = "snapflow"
"""Checkpoint dict key holding the SnapFlow phase metadata stamped at save time."""

_PARAM_COUNT_MILLIONS_THRESHOLD = 1_000_000


def _format_param_count(count: int) -> str:
    """Render a parameter count readably at both toy and VLA scale.

    Args:
        count: Number of parameters.

    Returns:
        A ``"311.9M"``-style string for real models, or an exact count for the
        small models used in tests, where ``0.0M`` would say nothing.
    """
    if count >= _PARAM_COUNT_MILLIONS_THRESHOLD:
        return f"{count / 1e6:.1f}M"
    return f"{count:,}"


@rank_zero_only
def _print_banner(trainer: L.Trainer, message: str) -> None:
    """Emit ``message`` to the console without garbling an active progress bar.

    Routed through the progress bar's ``print`` (``tqdm.write`` under the hood)
    when one is running, so the banner does not collide with the bar's own
    output. Falls back to ``logger.info`` otherwise. Only one sink is used, so
    the banner never shows up twice.

    Args:
        trainer: The active Lightning trainer.
        message: Pre-formatted, possibly multi-line banner text.
    """
    bar = trainer.progress_bar_callback
    # is_enabled lives on TQDMProgressBar/RichProgressBar, not on the ProgressBar
    # base class, and the base class's print() is a no-op. Default to False so an
    # exotic bar falls through to the logger rather than swallowing the banner.
    if bar is not None and getattr(bar, "is_enabled", False):
        bar.print(message)
        return
    logger.info(message)


@runtime_checkable
class SnapFlowCapable(Protocol):
    """A policy that can switch into SnapFlow self-distillation.

    Satisfied by any policy mixing in
    :class:`~physicalai.policies.mixins.SnapFlowPolicyMixin`, which covers
    :class:`~physicalai.policies.Pi05` and
    :class:`~physicalai.policies.SmolVLA`.

    Note:
        Used for static typing and documentation. Runtime detection goes through
        ``hasattr`` rather than ``isinstance``, because protocol ``isinstance``
        checks inspect the class and would reject duck-typed objects that carry
        the method on the instance.
    """

    def enable_snapflow(self, alpha: float, lambda_: float, num_inference_steps: int) -> None:
        """Activate the SnapFlow objective and freeze the VLM backbone."""
        ...


class ProgressReportingCallback(Callback):
    """Stream standardized training/validation telemetry to a sink.

    Forwards progress, step loss, and validation events through a ``report``
    callable so any consumer (a job store, an SSE stream, or logs) sees an
    identical telemetry schema, and honors cooperative cancellation via
    ``should_stop``. This keeps in-process and remote runners emitting the same
    data without duplicating the Lightning hook logic.

    ``report`` receives ``(progress, message, extra_info)`` where ``progress``
    is 0-100 and ``extra_info`` carries:

    - train batch: ``{"train/loss_step": float | None}``, plus
      ``{"global_step", "max_steps", "epoch"}`` on the logging cadence.
    - validation start: ``{"val_event": "start", "global_step", "max_steps"}``.
    - validation batch: ``{"val_event": "batch", "val_batch", "val/loss_step"}``
      (throttled to the logging cadence).
    - validation end: ``{"val_event": "end", "global_step", "val/loss",
      "val_elapsed_s"}``.

    Example:
        >>> cb = ProgressReportingCallback(report=sink, should_stop=lambda: False)
        >>> trainer = Trainer(callbacks=[cb])
    """

    def __init__(self, *, report: ReportFn, should_stop: StopFn) -> None:
        """Store the telemetry sink and cancellation probe.

        Args:
            report: Sink called with ``(progress, message, extra_info)``.
            should_stop: Returns True when training should stop cooperatively.
        """
        super().__init__()
        self._report = report
        self._should_stop = should_stop
        # Logging cadence in steps; resolved once Lightning knows the dataloader size.
        self._every_n_steps = 1
        self._val_start_t: float | None = None

    @staticmethod
    def _auto_every_n_steps(total_steps: int) -> int:
        """Pick a logging cadence in steps.

        Args:
            total_steps: Configured maximum number of steps.

        Returns:
            Cadence in steps. Targets ~1000 entries for budgets up to 100k steps,
            then caps at every 100 steps. Above 100k steps the cap dominates and
            the total entry count grows past 1000.
        """
        if total_steps <= 0:
            return 1
        return min(100, max(1, total_steps // 1000))

    @staticmethod
    def _to_scalar(value: object) -> float | None:
        """Coerce a metric to a float, handling tensors and plain scalars.

        Args:
            value: A 0-d tensor, a Python scalar, or None.

        Returns:
            The float value, or None when ``value`` is None.
        """
        if value is None:
            return None
        item = getattr(value, "item", None)
        if callable(item):
            return float(cast("float", item()))
        return float(value)  # type: ignore[arg-type]

    @staticmethod
    def _total_steps(trainer: L.Trainer) -> int | None:
        """Return the effective step budget, or None when it cannot be determined.

        ``estimated_stepping_batches`` accounts for a configured ``max_epochs``
        and becomes available after Lightning attaches the dataloader.
        """
        if trainer.max_steps > 0:
            return trainer.max_steps
        estimated_steps = trainer.estimated_stepping_batches
        return int(estimated_steps) if estimated_steps > 0 else None

    @staticmethod
    def _extract_loss(outputs: object) -> float | None:
        """Return a scalar loss from a step output, or None when unavailable.

        Handles a ``{"loss": tensor}`` mapping (training) or a bare loss tensor
        (eval-loss validation).
        """
        candidate = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
        detach = getattr(candidate, "detach", None)
        if detach is None:
            return None
        try:
            return detach().cpu().item()
        except (RuntimeError, ValueError):
            return None

    @staticmethod
    def _progress(trainer: L.Trainer) -> int:
        """Compute completion from the effective step budget.

        Args:
            trainer: The active Lightning trainer.

        Returns:
            Completion percentage clamped to 0-100. Returns 0 when the effective
            step budget is unavailable. Emits 100 only once all estimated steps
            complete so partial steps never round up to completion.
        """
        total_steps = ProgressReportingCallback._total_steps(trainer)
        if total_steps is None:
            return 0
        if trainer.global_step >= total_steps:
            return 100
        return min(99, int(trainer.global_step / total_steps * 100))

    def _check_stop(self, trainer: L.Trainer) -> None:
        """Stop the trainer cooperatively when cancellation was requested."""
        if self._should_stop():
            trainer.should_stop = True

    def on_fit_start(self, trainer: L.Trainer, _pl_module: L.LightningModule) -> None:
        """Resolve the logging cadence and honor a cancel requested before training."""
        self._every_n_steps = self._auto_every_n_steps(self._total_steps(trainer) or 0)
        self._check_stop(trainer)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        _pl_module: L.LightningModule,
        outputs: object,
        _batch: object,
        _batch_idx: int,
    ) -> None:
        """Report step progress and loss; honor cancellation."""
        global_step = trainer.global_step
        extra: dict[str, object] = {"train/loss_step": self._extract_loss(outputs)}
        # Attach the detailed cadence fields so consumers can throttle logs.
        if global_step <= 1 or global_step % self._every_n_steps == 0:
            extra["global_step"] = global_step
            extra["max_steps"] = self._total_steps(trainer)
            extra["epoch"] = trainer.current_epoch
        self._report(self._progress(trainer), None, extra)
        self._check_stop(trainer)

    def on_validation_start(self, trainer: L.Trainer, _pl_module: L.LightningModule) -> None:
        """Report the start of a validation pass; honor cancellation."""
        self._val_start_t = time.monotonic()
        self._report(
            self._progress(trainer),
            None,
            {"val_event": "start", "global_step": trainer.global_step, "max_steps": self._total_steps(trainer)},
        )
        self._check_stop(trainer)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        _pl_module: L.LightningModule,
        outputs: object,
        _batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002  # Lightning hook signature; unused here.
    ) -> None:
        """Report a throttled validation batch; honor cancellation."""
        current = batch_idx + 1
        if current == 1 or current % self._every_n_steps == 0:
            self._report(
                self._progress(trainer),
                None,
                {"val_event": "batch", "val_batch": current, "val/loss_step": self._extract_loss(outputs)},
            )
        self._check_stop(trainer)

    def on_validation_epoch_end(self, trainer: L.Trainer, _pl_module: L.LightningModule) -> None:
        """Report the validation summary with aggregated loss and elapsed time."""
        val_loss = trainer.callback_metrics.get("val/loss")
        val_loss_val = self._to_scalar(val_loss)
        elapsed = time.monotonic() - self._val_start_t if self._val_start_t is not None else 0.0
        self._report(
            self._progress(trainer),
            None,
            {
                "val_event": "end",
                "global_step": trainer.global_step,
                "val/loss": val_loss_val,
                "val_elapsed_s": elapsed,
            },
        )


class IterationTimer(Callback):
    """Log wall-clock time per training step in seconds.

    Logs ``train/iter_time_s`` on every training batch end.

    Example:
        >>> from physicalai.train.callbacks import IterationTimer
        >>> trainer = Trainer(callbacks=[IterationTimer()])
    """

    def on_train_batch_start(
        self,
        _trainer: L.Trainer,
        _pl_module: L.LightningModule,
        _batch: object,
        _batch_idx: int,
    ) -> None:
        """Record the batch start time."""
        self._start = time.perf_counter()

    def on_train_batch_end(
        self,
        _trainer: L.Trainer,
        pl_module: L.LightningModule,
        _outputs: object,
        _batch: object,
        _batch_idx: int,
    ) -> None:
        """Log elapsed time since batch start."""
        elapsed_s = time.perf_counter() - self._start
        pl_module.log("train/iter_time_s", elapsed_s, prog_bar=True)


class PolicyDatasetInteraction(Callback):
    """Callback to interact the policy and dataset before training starts."""

    @staticmethod
    def _interact_policy_dataset(trainer: L.Trainer, model: L.LightningModule) -> None:
        # Assumes trainer has a datamodule attached
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            reformat_dataset_to_match_policy(policy=model, datamodule=trainer.datamodule)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called at the start of `trainer.fit()`."""
        self._interact_policy_dataset(trainer, pl_module)


class SnapFlowPhaseCallback(Callback):
    """Enable SnapFlow self-distillation at a phase boundary within a single training run.

    At the configured boundary the callback transitions the policy from standard
    flow-matching (phase 1) into SnapFlow distillation (phase 2) without
    requiring a second CLI invocation:

    - Calls ``policy.enable_snapflow(alpha, lambda_, num_inference_steps)``
      which activates the mixed FM/consistency loss and freezes the VLM
      backbone via the policy's existing ``set_requires_grad`` primitive.
    - Reconfigures the optimizer so it only covers the now-trainable
      parameters (action expert + target-time embedding), giving a clean
      optimizer state for phase 2.

    The boundary is expressed either in optimizer steps (``start_step``) or in
    epochs (``start_epoch``); exactly one must be given. Use ``start_epoch``
    when the run is budgeted with ``Trainer(max_epochs=...)`` so the boundary
    does not have to be converted to steps by hand.

    The policy must expose an ``enable_snapflow`` method — both
    :class:`~physicalai.policies.SmolVLA` and
    :class:`~physicalai.policies.Pi05` satisfy this contract.

    Phase 2 is made visible in three ways, so a run never switches objectives
    silently:

    - A console banner is printed at the boundary through the progress bar
      (tqdm-safe), or through ``logger.info`` when no progress bar is running.
    - ``snapflow`` is logged as a progress-bar metric for every phase-2 batch.
      It is deliberately not logged during phase 1, so the key's presence in the
      bar is itself the phase indicator.
    - Every :class:`~lightning.pytorch.callbacks.ModelCheckpoint` filename
      template is prefixed with ``checkpoint_prefix`` from the boundary onwards,
      and the phase metadata is stamped into the checkpoint under a ``snapflow``
      key.

    Note:
        Reconfiguring the optimizer rebuilds the LR scheduler, so phase 2 starts
        with a fresh warmup. The cosine decay horizon is derived from
        ``Trainer.estimated_stepping_batches``, which reports the *total* run
        budget rather than the phase-2 remainder, so the phase-2 LR decays more
        slowly than a standalone phase-2 run would. Use two explicit
        ``physicalai fit`` runs if you need an exact phase-2 decay horizon.

    Note:
        The ``save_last`` checkpoint keeps its stock ``last.ckpt`` name across
        the boundary, so resuming always has a stable, phase-agnostic entry
        point. Use the checkpoint's ``snapflow`` metadata to tell which phase a
        given ``last.ckpt`` came from.

    Note:
        SnapFlow's shortcut target is bootstrapped from the model's own
        marginal-velocity predictions, so distilling an undertrained teacher
        distills noise. By default (``restore_best_teacher=True``), the
        callback restores the best-``val/loss`` weights from a monitored
        :class:`~lightning.pytorch.callbacks.ModelCheckpoint` *before* calling
        ``enable_snapflow()``, rather than distilling whatever the live
        in-memory model happens to be at the boundary. This requires a
        ``ModelCheckpoint(monitor=..., mode=...)`` among ``trainer.callbacks``;
        an unmonitored ``ModelCheckpoint`` (``monitor=None``) is ignored even if
        its ``best_model_path`` is populated, because ``best_model_path`` on an
        unmonitored checkpoint just means "most recent", not "best". If no
        monitored checkpoint is configured, or it has not saved yet, the
        callback warns and continues on the live weights. Set
        ``restore_best_teacher=False`` to always use the live weights and
        silence the warning.

    Note:
        ``val/loss`` is not comparable across the phase boundary: it measures
        full-denoising action MSE at ``num_inference_steps`` steps before
        activation, and at the (typically much lower, e.g. 1-NFE) SnapFlow
        ``num_inference_steps`` afterwards. By default
        (``scope_best_to_phase=True``), the callback resets every monitored
        ``ModelCheckpoint``'s best-tracking state at the boundary, so phase-2's
        best checkpoint is ranked only against other phase-2 checkpoints. The
        phase-1 best checkpoint file is preserved on disk (not deleted) as the
        distillation teacher's record. ``last.ckpt`` is unaffected. Set
        ``scope_best_to_phase=False`` to keep phase-1 scores in the running.

    Args:
        start_step: Optimizer step at which to activate SnapFlow. Mutually
            exclusive with ``start_epoch``.
        start_epoch: Epoch at which to activate SnapFlow, applied at the start
            of that epoch. Mutually exclusive with ``start_step``.
        alpha: Weight for the flow-matching loss branch (``L_FM``).
            Paper default: ``0.5``.
        lambda_: Scaling factor for the shortcut consistency loss
            (``L_shortcut``).  Paper default: ``0.1``.
        num_inference_steps: Denoising steps at inference time.
            Set to ``1`` for the full single-step SnapFlow speedup.
        checkpoint_prefix: Prefix applied to every ``ModelCheckpoint`` filename
            template once SnapFlow activates, so phase-1 and phase-2
            checkpoints are distinguishable on disk. Set to ``None`` to leave
            filenames untouched.
        restore_best_teacher: If ``True`` (default), restore the best-``val/loss``
            checkpoint from a monitored ``ModelCheckpoint`` before enabling
            SnapFlow. If ``False``, distill from the live in-memory model.
        best_teacher_monitor: Disambiguates which monitored ``ModelCheckpoint``
            to restore from when more than one is configured. Required (and
            only used) when ``restore_best_teacher`` is ``True`` and multiple
            monitored checkpoints are present.
        scope_best_to_phase: If ``True`` (default), reset every monitored
            ``ModelCheckpoint``'s best-tracking state at the boundary, so
            phase-2's best checkpoint is ranked only against phase-2 scores.

    Example:
        >>> from physicalai.train.callbacks import SnapFlowPhaseCallback
        >>> cb = SnapFlowPhaseCallback(start_step=50_000)
        >>> trainer = Trainer(max_steps=80_000, callbacks=[cb])

        >>> # Epoch-budgeted run: 10 epochs of flow matching, then 5 of distillation.
        >>> cb = SnapFlowPhaseCallback(start_epoch=10)
        >>> trainer = Trainer(max_epochs=15, callbacks=[cb])
    """

    def __init__(
        self,
        start_step: int | None = None,
        alpha: float = 0.5,
        lambda_: float = 0.1,
        num_inference_steps: int = 1,
        start_epoch: int | None = None,
        checkpoint_prefix: str | None = "snapflow-",
        *,
        restore_best_teacher: bool = True,
        best_teacher_monitor: str | None = None,
        scope_best_to_phase: bool = True,
    ) -> None:
        """Store phase-transition hyperparameters.

        Args:
            start_step: Optimizer step at which SnapFlow distillation begins.
            alpha: FM-loss weight.  Paper default: ``0.5``.
            lambda_: Shortcut-loss scale.  Paper default: ``0.1``.
            num_inference_steps: Inference denoising steps.  Use ``1`` for
                1-NFE SnapFlow.
            start_epoch: Epoch at which SnapFlow distillation begins.
            checkpoint_prefix: Prefix for ``ModelCheckpoint`` filename templates
                from the boundary onwards, or ``None`` to leave them alone.
            restore_best_teacher: Restore the best-``val/loss`` checkpoint
                before activating SnapFlow. Default: ``True``.
            best_teacher_monitor: Monitor name to disambiguate between multiple
                monitored ``ModelCheckpoint`` callbacks.
            scope_best_to_phase: Reset best-checkpoint tracking at the boundary
                so phase-2 checkpoints are ranked only against phase-2 scores.
                Default: ``True``.

        Raises:
            ValueError: If neither or both of ``start_step`` and ``start_epoch``
                are given, or if the given boundary is negative.
        """
        super().__init__()
        if (start_step is None) == (start_epoch is None):
            msg = (
                "SnapFlowPhaseCallback requires exactly one of start_step or start_epoch, "
                f"got start_step={start_step}, start_epoch={start_epoch}."
            )
            raise ValueError(msg)
        boundary = start_step if start_step is not None else start_epoch
        if boundary is not None and boundary < 0:
            msg = f"SnapFlow phase boundary must be >= 0, got {boundary}."
            raise ValueError(msg)

        self.start_step = start_step
        self.start_epoch = start_epoch
        self.alpha = alpha
        self.lambda_ = lambda_
        self.num_inference_steps = num_inference_steps
        self.checkpoint_prefix = checkpoint_prefix
        self.restore_best_teacher = restore_best_teacher
        self.best_teacher_monitor = best_teacher_monitor
        self.scope_best_to_phase = scope_best_to_phase
        self._activated = False
        self._activated_at_step: int | None = None
        self._restored_teacher_path: str | None = None

    def state_dict(self) -> dict[str, object]:
        """Persist the activation state so a resume does not re-trigger phase 2.

        Returns:
            Callback state to embed in the checkpoint.
        """
        return {
            "activated": self._activated,
            "activated_at_step": self._activated_at_step,
            "restored_teacher_path": self._restored_teacher_path,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore the activation state from a checkpoint.

        Args:
            state_dict: State previously produced by :meth:`state_dict`.
        """
        self._activated = bool(state_dict.get("activated"))
        activated_at_step = state_dict.get("activated_at_step")
        self._activated_at_step = int(activated_at_step) if isinstance(activated_at_step, int) else None
        restored_teacher_path = state_dict.get("restored_teacher_path")
        self._restored_teacher_path = restored_teacher_path if isinstance(restored_teacher_path, str) else None

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,  # noqa: ARG002
        pl_module: L.LightningModule,  # noqa: ARG002
        checkpoint: dict[str, Any],
    ) -> None:
        """Stamp the SnapFlow phase into every checkpoint this run writes.

        Makes the phase machine-readable rather than relying on the filename
        convention alone, so downstream tooling can tell a distilled checkpoint
        from a flow-matching one without parsing names.

        Args:
            trainer: The active Lightning trainer (unused).
            pl_module: The policy being trained (unused).
            checkpoint: Checkpoint dict to augment in place.
        """
        checkpoint[SNAPFLOW_CHECKPOINT_KEY] = {
            "enabled": self._activated,
            "alpha": self.alpha,
            "lambda_": self.lambda_,
            "num_inference_steps": self.num_inference_steps,
            "activated_at_step": self._activated_at_step,
        }

    def _prefix_checkpoint_filenames(self, trainer: L.Trainer) -> list[str]:
        """Prefix every ``ModelCheckpoint`` filename template with the phase marker.

        ``ModelCheckpoint.format_checkpoint_name`` reads ``self.filename`` at
        save time, so rewriting the template mid-run only affects checkpoints
        written from here on. Templates already carrying the prefix are skipped,
        which keeps the rewrite idempotent when a post-activation checkpoint is
        resumed.

        Args:
            trainer: The active Lightning trainer.

        Returns:
            The rewritten filename templates, for reporting in the banner.
        """
        if not self.checkpoint_prefix:
            return []

        renamed: list[str] = []
        for callback in trainer.checkpoint_callbacks:
            if not isinstance(callback, ModelCheckpoint):
                continue
            # Mirror Lightning's implicit default so an unset template keeps its
            # stock "{epoch}-{step}" shape instead of silently changing.
            template = callback.filename or f"{{epoch}}{callback.CHECKPOINT_JOIN_CHAR}{{step}}"
            if template.startswith(self.checkpoint_prefix):
                continue
            callback.filename = f"{self.checkpoint_prefix}{template}"
            renamed.append(callback.filename)
        return renamed

    def _resolve_monitored_checkpoint(self, trainer: L.Trainer) -> ModelCheckpoint | None:
        """Find the monitored ``ModelCheckpoint`` to restore the best teacher from.

        Unmonitored ``ModelCheckpoint`` callbacks (``monitor=None``) are
        deliberately excluded: Lightning still populates their
        ``best_model_path`` with the most recently saved file, which means
        "latest", not "best", and using it here would silently reintroduce the
        exact bug this feature fixes.

        Args:
            trainer: The active Lightning trainer.

        Returns:
            The monitored ``ModelCheckpoint`` to use, or ``None`` if none is
            configured.

        Raises:
            ValueError: If more than one monitored ``ModelCheckpoint`` is
                configured and ``best_teacher_monitor`` does not disambiguate
                between them.
        """
        monitored = [
            callback
            for callback in trainer.checkpoint_callbacks
            if isinstance(callback, ModelCheckpoint) and callback.monitor is not None
        ]
        if self.best_teacher_monitor is not None:
            monitored = [callback for callback in monitored if callback.monitor == self.best_teacher_monitor]
        if not monitored:
            return None
        if len(monitored) > 1:
            monitors = [callback.monitor for callback in monitored]
            msg = (
                "SnapFlowPhaseCallback found multiple monitored ModelCheckpoint callbacks "
                f"({monitors}); set best_teacher_monitor to disambiguate which one to restore "
                "the best teacher from."
            )
            raise ValueError(msg)
        return monitored[0]

    def _restore_best_teacher(self, trainer: L.Trainer, pl_module: L.LightningModule) -> str | None:
        """Load the best-``val/loss`` checkpoint into ``pl_module`` before distillation.

        Args:
            trainer: The active Lightning trainer.
            pl_module: The policy being trained; its weights are replaced in place.

        Returns:
            The restored checkpoint path, or ``None`` if nothing was restored
            (no monitored checkpoint configured, or it has not saved yet).

        Raises:
            IsADirectoryError: If the resolved ``best_model_path`` is a
                directory (a sharded FSDP/DeepSpeed checkpoint), which this
                path cannot load.
        """
        ckpt_cb = self._resolve_monitored_checkpoint(trainer)
        if ckpt_cb is None:
            rank_zero_warn(
                "SnapFlowPhaseCallback: restore_best_teacher=True but no monitored ModelCheckpoint "
                "(ModelCheckpoint(monitor=..., mode=...)) was found among trainer.callbacks; "
                "distilling from the live in-memory model instead. Add a monitored ModelCheckpoint "
                "(e.g. monitor='val/loss', mode='min') to warm-start phase 2 from the best checkpoint, "
                "or pass restore_best_teacher=False to silence this warning.",
            )
            return None
        best_path = ckpt_cb.best_model_path
        if not best_path:
            rank_zero_warn(
                f"SnapFlowPhaseCallback: monitored ModelCheckpoint(monitor={ckpt_cb.monitor!r}) has not "
                "saved a best checkpoint yet (best_model_path is empty); distilling from the live "
                "in-memory model instead.",
            )
            return None
        if Path(best_path).is_dir():
            msg = (
                f"SnapFlowPhaseCallback cannot restore best_model_path={best_path!r}: it is a directory "
                "(a sharded FSDP/DeepSpeed checkpoint), which this feature does not support. Pass "
                "restore_best_teacher=False and handle the warm-start manually."
            )
            raise IsADirectoryError(msg)

        trainer.strategy.barrier()
        # weights_only=False: Lightning checkpoints bundle hparams/callback state beyond
        # tensors, which the restricted unpickler rejects; this loads a checkpoint written
        # by this same trainer run, not untrusted input.
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)  # nosec B614
        pl_module.load_state_dict(checkpoint["state_dict"], strict=True)
        return best_path

    @staticmethod
    def _reset_best_tracking(trainer: L.Trainer) -> list[str]:
        """Reset every monitored ``ModelCheckpoint``'s best-tracking state.

        ``val/loss`` is not comparable across the phase boundary (different
        ``num_inference_steps``), so phase-1 scores must not keep phase-2
        checkpoints from ever being recognized as "best". ``last_model_path``
        is left untouched: ``last.ckpt`` stays the stable, phase-agnostic
        resume point. Already-saved phase-1 checkpoint files are not deleted.

        Args:
            trainer: The active Lightning trainer.

        Returns:
            The monitor names of the ``ModelCheckpoint`` callbacks that were reset.
        """
        reset: list[str] = []
        for callback in trainer.checkpoint_callbacks:
            if not isinstance(callback, ModelCheckpoint) or callback.monitor is None:
                continue
            torch_inf = torch.tensor(torch.inf)
            callback.kth_value = torch_inf if callback.mode == "min" else -torch_inf
            callback.current_score = None
            callback.best_k_models = {}
            callback.kth_best_model_path = ""
            callback.best_model_score = None
            callback.best_model_path = ""
            reset.append(callback.monitor)
        return reset

    def _banner(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        renamed: list[str],
        restored_teacher_path: str | None,
        reset_monitors: list[str],
    ) -> str:
        """Build the human-facing phase-transition banner.

        Args:
            trainer: The active Lightning trainer.
            pl_module: The policy being trained.
            renamed: Checkpoint filename templates rewritten at the boundary.
            restored_teacher_path: The checkpoint restored as the distillation
                teacher, or ``None`` if the live in-memory model was used.
            reset_monitors: Monitor names whose best-tracking state was reset.

        Returns:
            Multi-line banner text.
        """
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in pl_module.parameters())

        rule = "=" * 78
        lines = [
            rule,
            f"SnapFlow distillation ENABLED at step {trainer.global_step} (epoch {trainer.current_epoch})",
            f"  alpha={self.alpha:.2f}  lambda_={self.lambda_:.2f}  num_inference_steps={self.num_inference_steps}",
            (
                f"  trainable params: {_format_param_count(trainable)} / {_format_param_count(total)} "
                f"({100 * trainable / max(1, total):.1f}%) - VLM backbone is now frozen"
            ),
            "  Optimizer and LR scheduler rebuilt; phase 2 restarts the warmup.",
            "  Expect slower steps: the consistency branch runs 3 velocity passes per sample.",
        ]
        if restored_teacher_path is not None:
            lines.append(f"  Distillation teacher: restored best checkpoint '{restored_teacher_path}'.")
        elif self.restore_best_teacher:
            lines.append("  Distillation teacher: live in-memory model (no monitored checkpoint found).")
        else:
            lines.append("  Distillation teacher: live in-memory model (restore_best_teacher=False).")
        if reset_monitors:
            lines.append(
                f"  Best-checkpoint tracking reset for monitor(s) {reset_monitors}: "
                "val/loss is not comparable across num_inference_steps.",
            )
        if getattr(getattr(pl_module, "config", None), "compile_model", False):
            lines.append(
                "  compile_model is on: the first phase-2 step pays a one-time torch.compile "
                "recompile (minutes, not a hang).",
            )
        lines.extend(
            f"  checkpoints from here on: '{template}.ckpt' ('last.ckpt' is unchanged)" for template in renamed
        )
        lines.append(rule)
        return "\n".join(lines)

    def _activate(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Flip the policy into SnapFlow mode and rebuild the optimizer.

        Args:
            trainer: The active Lightning trainer.
            pl_module: The policy being trained.

        Raises:
            TypeError: If ``pl_module`` does not expose an ``enable_snapflow``
                method (i.e. is not a SmolVLA or Pi05 policy).
        """
        if not hasattr(pl_module, "enable_snapflow"):
            msg = (
                f"{type(pl_module).__name__} does not implement enable_snapflow(); "
                "SnapFlowPhaseCallback requires a SmolVLA or Pi05 policy."
            )
            raise TypeError(msg)
        policy = cast("SnapFlowCapable", pl_module)

        restored_teacher_path = self._restore_best_teacher(trainer, pl_module) if self.restore_best_teacher else None
        self._restored_teacher_path = restored_teacher_path

        policy.enable_snapflow(
            alpha=self.alpha,
            lambda_=self.lambda_,
            num_inference_steps=self.num_inference_steps,
        )
        # Reconfigure optimizers so frozen VLM params are excluded and phase 2
        # starts with a fresh optimizer state (no stale momentum from phase 1).
        trainer.strategy.setup_optimizers(trainer)
        self._activated = True
        self._activated_at_step = trainer.global_step

        reset_monitors = self._reset_best_tracking(trainer) if self.scope_best_to_phase else []
        renamed = self._prefix_checkpoint_filenames(trainer)
        _print_banner(trainer, self._banner(trainer, pl_module, renamed, restored_teacher_path, reset_monitors))

    @staticmethod
    def _log_phase(pl_module: L.LightningModule) -> None:
        """Surface the active SnapFlow phase in the progress bar.

        Logged only while phase 2 is running, so the key's presence in the bar
        is the indicator and phase 1 stays uncluttered.

        Args:
            pl_module: The policy being trained.
        """
        pl_module.log(SNAPFLOW_PROGRESS_BAR_KEY, 1.0, prog_bar=True, on_step=True, on_epoch=False)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Activate SnapFlow at an epoch boundary when configured with ``start_epoch``.

        Args:
            trainer: The active Lightning trainer.
            pl_module: The policy being trained.
        """
        if self._activated or self.start_epoch is None or trainer.current_epoch < self.start_epoch:
            return
        self._activate(trainer, pl_module)

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: object,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Activate SnapFlow at a step boundary and keep the phase visible thereafter.

        Args:
            trainer: The active Lightning trainer.
            pl_module: The policy being trained.
            batch: Current batch (unused).
            batch_idx: Current batch index within the epoch (unused).
        """
        if not self._activated and self.start_step is not None and trainer.global_step >= self.start_step:
            self._activate(trainer, pl_module)
        if self._activated:
            self._log_phase(pl_module)
