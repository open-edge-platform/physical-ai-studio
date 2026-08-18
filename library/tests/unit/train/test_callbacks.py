# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for training callbacks."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from physicalai.train.callbacks import (
    SNAPFLOW_PROGRESS_BAR_KEY,
    IterationTimer,
    ProgressReportingCallback,
    SnapFlowPhaseCallback,
)


def _loss(value: float) -> MagicMock:
    """Build a fake loss tensor whose ``.detach().cpu().item()`` is ``value``."""
    tensor = MagicMock()
    tensor.detach.return_value.cpu.return_value.item.return_value = value
    return tensor


def _trainer(
    *, global_step: int, max_steps: int, estimated_steps: int | None = None, epoch: int = 0,
) -> MagicMock:
    trainer = MagicMock(spec=L.Trainer)
    trainer.global_step = global_step
    trainer.max_steps = max_steps
    trainer.estimated_stepping_batches = max_steps if estimated_steps is None else estimated_steps
    trainer.current_epoch = epoch
    trainer.should_stop = False
    trainer.callback_metrics = {}
    # Real Trainers expose these as lists; MagicMock(spec=...) would hand back a
    # non-iterable mock and break the SnapFlow checkpoint-prefixing walk.
    trainer.checkpoint_callbacks = []
    return trainer


class TestProgressReportingCallback:
    """Tests for the shared progress/telemetry callback."""

    def _callback(self, *, should_stop: bool = False) -> tuple[ProgressReportingCallback, MagicMock]:
        report = MagicMock()
        cb = ProgressReportingCallback(report=report, should_stop=lambda: should_stop)
        return cb, report

    def test_train_batch_reports_loss_and_cadence_fields(self) -> None:
        cb, report = self._callback()
        trainer = _trainer(global_step=1, max_steps=1000, epoch=0)
        cb.on_fit_start(trainer, MagicMock())

        cb.on_train_batch_end(trainer, MagicMock(), {"loss": _loss(0.5)}, None, 0)

        progress, message, extra = report.call_args[0]
        assert progress == 0  # 1/1000 -> 0%
        assert message is None
        assert extra == {"train/loss_step": 0.5, "global_step": 1, "max_steps": 1000, "epoch": 0}

    def test_train_batch_off_cadence_reports_only_loss(self) -> None:
        cb, report = self._callback()
        trainer = _trainer(global_step=3, max_steps=10)
        cb.on_fit_start(trainer, MagicMock())  # cadence -> every 1 step for small budgets

        # Force a coarse cadence so step 3 is off-cadence.
        cb._every_n_steps = 5
        cb.on_train_batch_end(trainer, MagicMock(), {"loss": _loss(0.2)}, None, 0)

        _, _, extra = report.call_args[0]
        assert extra == {"train/loss_step": 0.2}

    def test_validation_start_emits_event(self) -> None:
        cb, report = self._callback()
        trainer = _trainer(global_step=500, max_steps=1000)

        cb.on_validation_start(trainer, MagicMock())

        _, _, extra = report.call_args[0]
        assert extra == {"val_event": "start", "global_step": 500, "max_steps": 1000}

    def test_validation_batch_throttled(self) -> None:
        cb, report = self._callback()
        trainer = _trainer(global_step=500, max_steps=1000)
        cb._every_n_steps = 3

        cb.on_validation_batch_end(trainer, MagicMock(), {"loss": _loss(0.4)}, None, 0)  # batch 1 -> emits
        cb.on_validation_batch_end(trainer, MagicMock(), {"loss": _loss(0.4)}, None, 1)  # batch 2 -> off cadence

        assert report.call_count == 1
        _, _, extra = report.call_args[0]
        assert extra == {"val_event": "batch", "val_batch": 1, "val/loss_step": 0.4}

    def test_validation_end_emits_summary(self) -> None:
        cb, report = self._callback()
        trainer = _trainer(global_step=500, max_steps=1000)
        trainer.callback_metrics = {"val/loss": MagicMock(**{"item.return_value": 0.15})}
        cb.on_validation_start(trainer, MagicMock())

        cb.on_validation_epoch_end(trainer, MagicMock())

        _, _, extra = report.call_args[0]
        assert extra["val_event"] == "end"
        assert extra["global_step"] == 500
        assert extra["val/loss"] == 0.15
        assert isinstance(extra["val_elapsed_s"], float)

    def test_validation_end_handles_scalar_val_loss(self) -> None:
        # callback_metrics may hold a plain Python scalar without ``.item()``.
        cb, report = self._callback()
        trainer = _trainer(global_step=500, max_steps=1000)
        trainer.callback_metrics = {"val/loss": 0.25}
        cb.on_validation_start(trainer, MagicMock())

        cb.on_validation_epoch_end(trainer, MagicMock())

        _, _, extra = report.call_args[0]
        assert extra["val/loss"] == 0.25

    def test_progress_floors_and_never_rounds_up_before_completion(self) -> None:
        # 995/1000 must not report 100% just because it rounds up.
        cb, report = self._callback()
        trainer = _trainer(global_step=995, max_steps=1000)
        cb.on_fit_start(trainer, MagicMock())

        cb.on_train_batch_end(trainer, MagicMock(), {"loss": _loss(0.1)}, None, 0)

        progress = report.call_args[0][0]
        assert progress == 99

    def test_progress_reports_100_only_when_complete(self) -> None:
        cb, report = self._callback()
        trainer = _trainer(global_step=1000, max_steps=1000)
        cb.on_fit_start(trainer, MagicMock())

        cb.on_train_batch_end(trainer, MagicMock(), {"loss": _loss(0.1)}, None, 0)

        progress = report.call_args[0][0]
        assert progress == 100

    def test_max_epochs_uses_estimated_step_budget(self) -> None:
        cb, report = self._callback()
        trainer = _trainer(global_step=1000, max_steps=-1, estimated_steps=2000)
        cb.on_fit_start(trainer, MagicMock())

        cb.on_train_batch_end(trainer, MagicMock(), {"loss": _loss(0.1)}, None, 0)

        assert cb._every_n_steps == 2
        progress, _, extra = report.call_args[0]
        assert progress == 50
        assert extra["max_steps"] == 2000

    def test_unset_max_steps_emits_none_sentinel(self) -> None:
        # Lightning uses -1 for an unbounded step budget; surface it as None.
        cb, report = self._callback()
        trainer = _trainer(global_step=1, max_steps=-1, estimated_steps=-1)
        cb.on_fit_start(trainer, MagicMock())

        cb.on_train_batch_end(trainer, MagicMock(), {"loss": _loss(0.5)}, None, 0)

        progress, _, extra = report.call_args[0]
        assert progress == 0
        assert extra["max_steps"] is None

    def test_honors_should_stop(self) -> None:
        cb, _ = self._callback(should_stop=True)
        trainer = _trainer(global_step=1, max_steps=10)
        cb.on_fit_start(trainer, MagicMock())

        cb.on_train_batch_end(trainer, MagicMock(), {"loss": _loss(0.5)}, None, 0)

        assert trainer.should_stop is True

    def test_fit_start_honors_pending_cancel(self) -> None:
        # A cancel requested before training starts must stop before the first batch.
        cb, _ = self._callback(should_stop=True)
        trainer = _trainer(global_step=0, max_steps=10)

        cb.on_fit_start(trainer, MagicMock())

        assert trainer.should_stop is True


class TestIterationTimer:
    """Tests for the IterationTimer callback."""

    def test_logs_iter_time_in_seconds(self):
        """Verify that iter time is logged in seconds."""
        callback = IterationTimer()
        trainer = MagicMock(spec=L.Trainer)
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_batch_start(trainer, pl_module, None, 0)
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        pl_module.log.assert_called_once()
        args, kwargs = pl_module.log.call_args
        assert args[0] == "train/iter_time_s"
        assert isinstance(args[1], float)
        assert args[1] >= 0
        assert kwargs["prog_bar"] is True

    def test_iter_time_reflects_elapsed_duration(self):
        """Verify that logged time reflects actual elapsed duration."""
        import time

        callback = IterationTimer()
        trainer = MagicMock(spec=L.Trainer)
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_batch_start(trainer, pl_module, None, 0)
        time.sleep(0.05)
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        logged_time = pl_module.log.call_args[0][1]
        assert logged_time >= 0.04  # allow small timing tolerance
        assert logged_time < 1.0  # sanity upper bound


class TestSnapFlowPhaseCallback:
    """Tests for the SnapFlow phase-transition callback."""

    @staticmethod
    def _policy(*, compile_model: bool = False) -> MagicMock:
        """Build a stand-in policy exposing the enable_snapflow contract."""
        policy = MagicMock(spec=["enable_snapflow", "parameters", "log", "config"])
        policy.parameters.return_value = []
        policy.config = SimpleNamespace(compile_model=compile_model)
        return policy

    @staticmethod
    def _checkpoint_callback(filename: str | None = "epoch{epoch:03d}") -> ModelCheckpoint:
        """Build a real ModelCheckpoint so filename rewriting is exercised for real."""
        return ModelCheckpoint(filename=filename)

    def test_requires_exactly_one_boundary(self) -> None:
        with pytest.raises(ValueError, match="exactly one of start_step or start_epoch"):
            SnapFlowPhaseCallback()
        with pytest.raises(ValueError, match="exactly one of start_step or start_epoch"):
            SnapFlowPhaseCallback(start_step=10, start_epoch=2)

    def test_rejects_negative_boundary(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            SnapFlowPhaseCallback(start_step=-1)
        with pytest.raises(ValueError, match="must be >= 0"):
            SnapFlowPhaseCallback(start_epoch=-1)

    def test_step_boundary_activates_once_at_start_step(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100, alpha=0.4, lambda_=0.2, num_inference_steps=2)
        policy = self._policy()

        cb.on_train_batch_start(_trainer(global_step=99, max_steps=200), policy, None, 0)
        policy.enable_snapflow.assert_not_called()

        trainer = _trainer(global_step=100, max_steps=200)
        cb.on_train_batch_start(trainer, policy, None, 0)
        policy.enable_snapflow.assert_called_once_with(alpha=0.4, lambda_=0.2, num_inference_steps=2)
        trainer.strategy.setup_optimizers.assert_called_once_with(trainer)

        # Subsequent batches must not re-activate.
        cb.on_train_batch_start(_trainer(global_step=101, max_steps=200), policy, None, 0)
        policy.enable_snapflow.assert_called_once()

    def test_epoch_boundary_activates_at_start_epoch(self) -> None:
        cb = SnapFlowPhaseCallback(start_epoch=10)
        policy = self._policy()

        cb.on_train_epoch_start(_trainer(global_step=0, max_steps=-1, epoch=9), policy)
        policy.enable_snapflow.assert_not_called()

        trainer = _trainer(global_step=0, max_steps=-1, epoch=10)
        cb.on_train_epoch_start(trainer, policy)
        policy.enable_snapflow.assert_called_once_with(alpha=0.5, lambda_=0.1, num_inference_steps=1)
        trainer.strategy.setup_optimizers.assert_called_once_with(trainer)

    def test_boundaries_do_not_cross_trigger(self) -> None:
        """A step-configured callback ignores epochs, and vice versa."""
        step_cb = SnapFlowPhaseCallback(start_step=100)
        policy = self._policy()
        step_cb.on_train_epoch_start(_trainer(global_step=0, max_steps=200, epoch=999), policy)
        policy.enable_snapflow.assert_not_called()

        epoch_cb = SnapFlowPhaseCallback(start_epoch=10)
        epoch_cb.on_train_batch_start(_trainer(global_step=10_000, max_steps=-1, epoch=0), policy, None, 0)
        policy.enable_snapflow.assert_not_called()

    def test_rejects_policy_without_enable_snapflow(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=0)
        policy = MagicMock(spec=["parameters"])

        with pytest.raises(TypeError, match="does not implement enable_snapflow"):
            cb.on_train_batch_start(_trainer(global_step=0, max_steps=10), policy, None, 0)

    def test_activation_state_survives_checkpoint_round_trip(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100)
        policy = self._policy()
        cb.on_train_batch_start(_trainer(global_step=100, max_steps=200), policy, None, 0)
        assert cb.state_dict() == {
            "activated": True,
            "activated_at_step": 100,
            "restored_teacher_path": None,
        }

        resumed = SnapFlowPhaseCallback(start_step=100)
        resumed.load_state_dict(cb.state_dict())
        resumed_policy = self._policy()
        resumed.on_train_batch_start(_trainer(global_step=150, max_steps=200), resumed_policy, None, 0)
        resumed_policy.enable_snapflow.assert_not_called()

    def test_fresh_callback_reports_not_activated(self) -> None:
        cb = SnapFlowPhaseCallback(start_epoch=5)
        assert cb.state_dict() == {
            "activated": False,
            "activated_at_step": None,
            "restored_teacher_path": None,
        }
        cb.load_state_dict({})
        assert cb.state_dict() == {
            "activated": False,
            "activated_at_step": None,
            "restored_teacher_path": None,
        }

    def test_phase_metric_absent_before_activation_and_present_after(self) -> None:
        """The metric's presence in the progress bar is itself the phase indicator."""
        cb = SnapFlowPhaseCallback(start_step=100)
        policy = self._policy()

        cb.on_train_batch_start(_trainer(global_step=99, max_steps=200), policy, None, 0)
        policy.log.assert_not_called()

        cb.on_train_batch_start(_trainer(global_step=100, max_steps=200), policy, None, 0)
        name, value = policy.log.call_args[0]
        assert name == SNAPFLOW_PROGRESS_BAR_KEY
        assert value == 1.0
        assert policy.log.call_args[1]["prog_bar"] is True

    def test_phase_metric_logged_after_epoch_boundary_activation(self) -> None:
        cb = SnapFlowPhaseCallback(start_epoch=10)
        policy = self._policy()

        cb.on_train_epoch_start(_trainer(global_step=0, max_steps=-1, epoch=10), policy)
        cb.on_train_batch_start(_trainer(global_step=1, max_steps=-1, epoch=10), policy, None, 0)

        policy.log.assert_called_once()
        assert policy.log.call_args[0][0] == SNAPFLOW_PROGRESS_BAR_KEY

    def test_checkpoint_filenames_get_phase_prefix(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100)
        ckpt = self._checkpoint_callback()
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt]

        cb.on_train_batch_start(trainer, self._policy(), None, 0)

        assert ckpt.filename == "snapflow-epoch{epoch:03d}"

    def test_unset_checkpoint_template_keeps_lightning_default_shape(self) -> None:
        """An unset filename must not silently change shape when prefixed."""
        cb = SnapFlowPhaseCallback(start_step=0)
        ckpt = self._checkpoint_callback(filename=None)
        trainer = _trainer(global_step=0, max_steps=10)
        trainer.checkpoint_callbacks = [ckpt]

        cb.on_train_batch_start(trainer, self._policy(), None, 0)

        assert ckpt.filename == f"snapflow-{{epoch}}{ModelCheckpoint.CHECKPOINT_JOIN_CHAR}{{step}}"

    def test_checkpoint_prefixing_is_idempotent_across_resume(self) -> None:
        """Resuming a phase-2 checkpoint must not stack a second prefix."""
        ckpt = self._checkpoint_callback(filename="snapflow-epoch{epoch:03d}")
        resumed = SnapFlowPhaseCallback(start_step=100)
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt]

        resumed.on_train_batch_start(trainer, self._policy(), None, 0)

        assert ckpt.filename == "snapflow-epoch{epoch:03d}"

    def test_checkpoint_prefix_can_be_disabled(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100, checkpoint_prefix=None)
        ckpt = self._checkpoint_callback()
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt]

        cb.on_train_batch_start(trainer, self._policy(), None, 0)

        assert ckpt.filename == "epoch{epoch:03d}"

    def test_checkpoint_is_stamped_with_phase_metadata(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100, alpha=0.4, lambda_=0.2, num_inference_steps=2)
        policy = self._policy()
        trainer = _trainer(global_step=100, max_steps=200)

        before: dict[str, object] = {}
        cb.on_save_checkpoint(trainer, policy, before)
        assert before["snapflow"] == {
            "enabled": False,
            "alpha": 0.4,
            "lambda_": 0.2,
            "num_inference_steps": 2,
            "activated_at_step": None,
        }

        cb.on_train_batch_start(trainer, policy, None, 0)
        after: dict[str, object] = {}
        cb.on_save_checkpoint(trainer, policy, after)
        assert after["snapflow"] == {
            "enabled": True,
            "alpha": 0.4,
            "lambda_": 0.2,
            "num_inference_steps": 2,
            "activated_at_step": 100,
        }

    def test_banner_is_printed_through_the_progress_bar(self) -> None:
        """The banner must go through the bar's print so tqdm output is not garbled."""
        cb = SnapFlowPhaseCallback(start_step=100)
        trainer = _trainer(global_step=100, max_steps=200)

        cb.on_train_batch_start(trainer, self._policy(), None, 0)

        banner = trainer.progress_bar_callback.print.call_args[0][0]
        assert "SnapFlow distillation ENABLED at step 100" in banner
        assert "compile_model" not in banner

    def test_banner_warns_about_the_compile_recompile_stall(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100)
        trainer = _trainer(global_step=100, max_steps=200)

        cb.on_train_batch_start(trainer, self._policy(compile_model=True), None, 0)

        banner = trainer.progress_bar_callback.print.call_args[0][0]
        assert "compile_model is on" in banner


class TestSnapFlowPhaseCallbackBestCheckpoint:
    """Tests for restore-best-teacher and phase-scoped best-tracking behavior."""

    @staticmethod
    def _policy(*, compile_model: bool = False) -> MagicMock:
        policy = MagicMock(spec=["enable_snapflow", "parameters", "log", "config", "load_state_dict"])
        policy.parameters.return_value = []
        policy.config = SimpleNamespace(compile_model=compile_model)
        return policy

    @staticmethod
    def _monitored_checkpoint(
        *,
        monitor: str = "val/loss",
        mode: str = "min",
        best_model_path: str = "/ckpt/phase1-best.ckpt",
    ) -> MagicMock:
        """A stand-in monitored ModelCheckpoint with a populated best_model_path."""
        ckpt = MagicMock(spec=ModelCheckpoint)
        ckpt.monitor = monitor
        ckpt.mode = mode
        ckpt.best_model_path = best_model_path
        ckpt.last_model_path = "/ckpt/last.ckpt"
        ckpt.filename = "epoch{epoch:03d}"
        return ckpt

    def test_restores_best_teacher_before_enabling_snapflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loaded = {}

        def fake_load(path, map_location=None, weights_only=None):  # ruff: ignore[missing-type-function-argument, unused-function-argument]
            loaded["path"] = path
            return {"state_dict": {"fake": "weights"}}

        monkeypatch.setattr("physicalai.train.callbacks.torch.load", fake_load)

        cb = SnapFlowPhaseCallback(start_step=100)
        ckpt_cb = self._monitored_checkpoint()
        policy = self._policy()
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt_cb]

        call_order: list[str] = []
        policy.load_state_dict.side_effect = lambda *a, **k: call_order.append("load_state_dict")  # ruff: ignore[unused-lambda-argument]
        policy.enable_snapflow.side_effect = lambda *a, **k: call_order.append("enable_snapflow")  # ruff: ignore[unused-lambda-argument]

        cb.on_train_batch_start(trainer, policy, None, 0)

        assert loaded["path"] == "/ckpt/phase1-best.ckpt"
        policy.load_state_dict.assert_called_once_with({"fake": "weights"}, strict=True)
        assert call_order == ["load_state_dict", "enable_snapflow"]
        assert cb.state_dict()["restored_teacher_path"] == "/ckpt/phase1-best.ckpt"

    def test_warns_and_continues_when_no_monitored_checkpoint(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100)
        policy = self._policy()
        # Only an unmonitored ModelCheckpoint present.
        unmonitored = ModelCheckpoint()
        unmonitored.best_model_path = "/ckpt/most-recent.ckpt"  # populated but meaningless: monitor=None
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [unmonitored]

        with pytest.warns(UserWarning, match="no monitored ModelCheckpoint"):
            cb.on_train_batch_start(trainer, policy, None, 0)

        policy.load_state_dict.assert_not_called()
        policy.enable_snapflow.assert_called_once()
        assert cb.state_dict()["restored_teacher_path"] is None

    def test_warns_when_monitored_checkpoint_has_not_saved_yet(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100)
        policy = self._policy()
        ckpt_cb = self._monitored_checkpoint(best_model_path="")
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt_cb]

        with pytest.warns(UserWarning, match="has not saved a best checkpoint"):
            cb.on_train_batch_start(trainer, policy, None, 0)

        policy.load_state_dict.assert_not_called()

    def test_restore_best_teacher_can_be_disabled(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100, restore_best_teacher=False)
        policy = self._policy()
        ckpt_cb = self._monitored_checkpoint()
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt_cb]

        cb.on_train_batch_start(trainer, policy, None, 0)

        policy.load_state_dict.assert_not_called()
        assert cb.state_dict()["restored_teacher_path"] is None

    def test_multiple_monitored_checkpoints_require_disambiguation(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100)
        policy = self._policy()
        ckpt_a = self._monitored_checkpoint(monitor="val/loss", best_model_path="/ckpt/a.ckpt")
        ckpt_b = self._monitored_checkpoint(monitor="val/other", best_model_path="/ckpt/b.ckpt")
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt_a, ckpt_b]

        with pytest.raises(ValueError, match="multiple monitored ModelCheckpoint"):
            cb.on_train_batch_start(trainer, policy, None, 0)

    def test_best_teacher_monitor_disambiguates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "physicalai.train.callbacks.torch.load",
            lambda *a, **k: {"state_dict": {}},  # ruff: ignore[unused-lambda-argument]
        )
        cb = SnapFlowPhaseCallback(start_step=100, best_teacher_monitor="val/other")
        policy = self._policy()
        ckpt_a = self._monitored_checkpoint(monitor="val/loss", best_model_path="/ckpt/a.ckpt")
        ckpt_b = self._monitored_checkpoint(monitor="val/other", best_model_path="/ckpt/b.ckpt")
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt_a, ckpt_b]

        cb.on_train_batch_start(trainer, policy, None, 0)

        assert cb.state_dict()["restored_teacher_path"] == "/ckpt/b.ckpt"

    def test_directory_best_model_path_raises(self, tmp_path) -> None:  # ruff: ignore[missing-type-function-argument]
        cb = SnapFlowPhaseCallback(start_step=100)
        policy = self._policy()
        ckpt_cb = self._monitored_checkpoint(best_model_path=str(tmp_path))
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt_cb]

        with pytest.raises(IsADirectoryError, match="sharded FSDP/DeepSpeed"):
            cb.on_train_batch_start(trainer, policy, None, 0)

    def test_reset_clears_best_tracking_but_preserves_last(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100, restore_best_teacher=False)
        policy = self._policy()
        ckpt_cb = ModelCheckpoint(monitor="val/loss", mode="min")
        ckpt_cb.best_model_score = torch.tensor(0.1)
        ckpt_cb.best_model_path = "/ckpt/phase1-best.ckpt"
        ckpt_cb.best_k_models = {"/ckpt/phase1-best.ckpt": torch.tensor(0.1)}
        ckpt_cb.kth_best_model_path = "/ckpt/phase1-best.ckpt"
        ckpt_cb.kth_value = torch.tensor(0.1)
        ckpt_cb.current_score = torch.tensor(0.1)
        ckpt_cb.last_model_path = "/ckpt/last.ckpt"
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt_cb]

        cb.on_train_batch_start(trainer, policy, None, 0)

        assert ckpt_cb.best_model_path == ""
        assert ckpt_cb.best_model_score is None
        assert ckpt_cb.best_k_models == {}
        assert ckpt_cb.kth_best_model_path == ""
        assert ckpt_cb.current_score is None
        assert ckpt_cb.kth_value == torch.tensor(torch.inf)
        # last.ckpt is the stable phase-agnostic resume point; must be untouched.
        assert ckpt_cb.last_model_path == "/ckpt/last.ckpt"

    def test_scope_best_to_phase_can_be_disabled(self) -> None:
        cb = SnapFlowPhaseCallback(start_step=100, restore_best_teacher=False, scope_best_to_phase=False)
        policy = self._policy()
        ckpt_cb = ModelCheckpoint(monitor="val/loss", mode="min")
        ckpt_cb.best_model_path = "/ckpt/phase1-best.ckpt"
        ckpt_cb.best_model_score = torch.tensor(0.1)
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt_cb]

        cb.on_train_batch_start(trainer, policy, None, 0)

        assert ckpt_cb.best_model_path == "/ckpt/phase1-best.ckpt"
        assert ckpt_cb.best_model_score == torch.tensor(0.1)

    def test_restore_happens_before_reset_so_phase1_best_is_still_visible(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The restore must read best_model_path before the reset clears it."""
        monkeypatch.setattr(
            "physicalai.train.callbacks.torch.load",
            lambda *a, **k: {"state_dict": {}},  # ruff: ignore[unused-lambda-argument]
        )
        cb = SnapFlowPhaseCallback(start_step=100)
        policy = self._policy()
        ckpt_cb = ModelCheckpoint(monitor="val/loss", mode="min")
        ckpt_cb.best_model_path = "/ckpt/phase1-best.ckpt"
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt_cb]

        cb.on_train_batch_start(trainer, policy, None, 0)

        assert cb.state_dict()["restored_teacher_path"] == "/ckpt/phase1-best.ckpt"
        assert ckpt_cb.best_model_path == ""  # reset afterward

    def test_banner_reports_restored_teacher_and_reset_monitors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "physicalai.train.callbacks.torch.load",
            lambda *a, **k: {"state_dict": {}},  # ruff: ignore[unused-lambda-argument]
        )
        cb = SnapFlowPhaseCallback(start_step=100)
        policy = self._policy()
        ckpt_cb = self._monitored_checkpoint()
        trainer = _trainer(global_step=100, max_steps=200)
        trainer.checkpoint_callbacks = [ckpt_cb]

        cb.on_train_batch_start(trainer, policy, None, 0)

        banner = trainer.progress_bar_callback.print.call_args[0][0]
        assert "restored best checkpoint '/ckpt/phase1-best.ckpt'" in banner
        assert "Best-checkpoint tracking reset for monitor(s)" in banner
