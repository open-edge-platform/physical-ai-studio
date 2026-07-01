# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for training callbacks."""

from unittest.mock import MagicMock

import lightning as L

from physicalai.train.callbacks import IterationTimer, ProgressReportingCallback


def _loss(value: float) -> MagicMock:
    """Build a fake loss tensor whose ``.detach().cpu().item()`` is ``value``."""
    tensor = MagicMock()
    tensor.detach.return_value.cpu.return_value.item.return_value = value
    return tensor


def _trainer(*, global_step: int, max_steps: int, epoch: int = 0) -> MagicMock:
    trainer = MagicMock(spec=L.Trainer)
    trainer.global_step = global_step
    trainer.max_steps = max_steps
    trainer.current_epoch = epoch
    trainer.should_stop = False
    trainer.callback_metrics = {}
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

    def test_unset_max_steps_emits_none_sentinel(self) -> None:
        # Lightning uses -1 for an unbounded step budget; surface it as None.
        cb, report = self._callback()
        trainer = _trainer(global_step=1, max_steps=-1)
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
