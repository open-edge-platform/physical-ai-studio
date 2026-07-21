# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared training-progress log formatting and remote mirroring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.training_backends._log_format import (
    format_training_progress,
    format_validation_progress,
    format_validation_start,
    format_validation_summary,
    render_progress_log,
)
from services.training_backends.remote import RemoteTrainingBackend

REMOTE = "services.training_backends.remote"


class TestFormatTrainingProgress:
    def test_renders_step_progress_and_loss(self) -> None:
        line = format_training_progress(global_step=250, max_steps=1000, loss=0.125)
        assert line == "Training progress: step=250/1000 (25%), train/loss_step=0.125"

    def test_none_loss_renders_literally(self) -> None:
        line = format_training_progress(global_step=1, max_steps=100, loss=None)
        assert line == "Training progress: step=1/100 (1%), train/loss_step=None"

    def test_zero_max_steps_floors_to_one(self) -> None:
        # Avoids divide-by-zero; progress clamps to 100.
        line = format_training_progress(global_step=5, max_steps=0, loss=0.5)
        assert line == "Training progress: step=5/1 (100%), train/loss_step=0.5"


class TestFormatValidation:
    def test_start_renders_step(self) -> None:
        assert format_validation_start(global_step=300, max_steps=1000) == "Validation started at step=300/1000"

    def test_start_floors_zero_max_steps(self) -> None:
        assert format_validation_start(global_step=0, max_steps=0) == "Validation started at step=0/1"

    def test_progress_renders_batch_and_loss(self) -> None:
        assert format_validation_progress(batch=3, loss=0.4) == "Validation progress: batch=3, val/loss_step=0.4"

    def test_progress_none_loss_renders_literally(self) -> None:
        assert format_validation_progress(batch=1, loss=None) == "Validation progress: batch=1, val/loss_step=None"

    def test_summary_renders_loss_and_elapsed(self) -> None:
        line = format_validation_summary(global_step=300, val_loss=0.2, elapsed_s=12.34)
        assert line == "Validation finished at step=300, val/loss=0.2, elapsed=12.3s"

    def test_summary_none_loss_renders_literally(self) -> None:
        line = format_validation_summary(global_step=300, val_loss=None, elapsed_s=0.0)
        assert line == "Validation finished at step=300, val/loss=None, elapsed=0.0s"


class TestRenderTrainingProgress:
    def test_logs_when_detailed_fields_present(self) -> None:
        extra_info = {
            "train/loss_step": 0.2,
            "global_step": 300,
            "max_steps": 1000,
            "epoch": 2,
        }
        assert render_progress_log(extra_info) == "Training progress: step=300/1000 (30%), train/loss_step=0.2"

    def test_skips_when_global_step_absent(self) -> None:
        # Non-cadence states carry only the loss; nothing is logged.
        assert render_progress_log({"train/loss_step": 0.2}) is None

    def test_skips_malformed_step_fields(self) -> None:
        assert render_progress_log({"global_step": "oops", "max_steps": 1000, "train/loss_step": 0.2}) is None

    def test_non_numeric_loss_becomes_none(self) -> None:
        extra_info = {"global_step": 10, "max_steps": 100, "train/loss_step": "nan"}
        assert render_progress_log(extra_info) == "Training progress: step=10/100 (10%), train/loss_step=None"


class TestRenderValidationProgress:
    def test_logs_validation_start(self) -> None:
        extra_info = {"val_event": "start", "global_step": 300, "max_steps": 1000}
        assert render_progress_log(extra_info) == "Validation started at step=300/1000"

    def test_logs_validation_batch(self) -> None:
        extra_info = {"val_event": "batch", "val_batch": 2, "val/loss_step": 0.4}
        assert render_progress_log(extra_info) == "Validation progress: batch=2, val/loss_step=0.4"

    def test_logs_validation_summary(self) -> None:
        extra_info = {"val_event": "end", "global_step": 300, "val/loss": 0.2, "val_elapsed_s": 5.0}
        assert render_progress_log(extra_info) == "Validation finished at step=300, val/loss=0.2, elapsed=5.0s"

    def test_non_numeric_val_loss_becomes_none(self) -> None:
        extra_info = {"val_event": "end", "global_step": 300, "val/loss": "nan", "val_elapsed_s": "oops"}
        assert render_progress_log(extra_info) == "Validation finished at step=300, val/loss=None, elapsed=0.0s"

    def test_skips_malformed_start(self) -> None:
        assert render_progress_log({"val_event": "start", "global_step": "oops"}) is None

    def test_skips_unknown_event(self) -> None:
        assert render_progress_log({"val_event": "other"}) is None


def test_apply_state_mirrors_detailed_fields_to_job_log() -> None:
    """A running state with cadence fields produces a job-log line."""
    context = MagicMock()
    state = {
        "status": "running",
        "progress": 50,
        "message": "Training",
        "extra_info": {"train/loss_step": 0.1, "global_step": 500, "max_steps": 1000, "epoch": 1},
    }
    backend = RemoteTrainingBackend.__new__(RemoteTrainingBackend)
    with patch(f"{REMOTE}.logger") as logger:
        completed = backend._apply_state(context, state)
    assert completed is False
    logger.info.assert_called_once_with("Training progress: step=500/1000 (50%), train/loss_step=0.1")


def test_apply_state_mirrors_validation_fields_to_job_log() -> None:
    """A running state carrying a val_event produces a validation job-log line."""
    context = MagicMock()
    state = {
        "status": "running",
        "progress": 50,
        "message": "Validating",
        "extra_info": {"val_event": "batch", "val_batch": 1, "val/loss_step": 0.3},
    }
    backend = RemoteTrainingBackend.__new__(RemoteTrainingBackend)
    with patch(f"{REMOTE}.logger") as logger:
        completed = backend._apply_state(context, state)
    assert completed is False
    logger.info.assert_called_once_with("Validation progress: batch=1, val/loss_step=0.3")


def test_apply_state_suppresses_duplicate_progress_lines() -> None:
    """Re-emitted identical progress (e.g. during export) is logged only once."""
    context = MagicMock()
    state = {
        "status": "running",
        "progress": 100,
        "message": "Training",
        "extra_info": {"train/loss_step": 4.48, "global_step": 100, "max_steps": 100, "epoch": 1},
    }
    backend = RemoteTrainingBackend.__new__(RemoteTrainingBackend)
    with patch(f"{REMOTE}.logger") as logger:
        backend._apply_state(context, state)
        backend._apply_state(context, dict(state))
        backend._apply_state(context, dict(state))
    logger.info.assert_called_once_with("Training progress: step=100/100 (100%), train/loss_step=4.48")
