# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared training-progress log formatting and remote mirroring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.training_backends._log_format import format_training_progress
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


class TestRemoteLogTrainingProgress:
    def test_logs_when_detailed_fields_present(self) -> None:
        extra_info = {
            "train/loss_step": 0.2,
            "global_step": 300,
            "max_steps": 1000,
            "epoch": 2,
        }
        with patch(f"{REMOTE}.logger") as logger:
            RemoteTrainingBackend._log_training_progress(extra_info)
        logger.info.assert_called_once_with("Training progress: step=300/1000 (30%), train/loss_step=0.2")

    def test_skips_when_global_step_absent(self) -> None:
        # Non-cadence states carry only the loss; nothing is logged.
        with patch(f"{REMOTE}.logger") as logger:
            RemoteTrainingBackend._log_training_progress({"train/loss_step": 0.2})
        logger.info.assert_not_called()

    def test_skips_malformed_step_fields(self) -> None:
        extra_info = {"global_step": "oops", "max_steps": 1000, "train/loss_step": 0.2}
        with patch(f"{REMOTE}.logger") as logger:
            RemoteTrainingBackend._log_training_progress(extra_info)
        logger.info.assert_not_called()

    def test_non_numeric_loss_becomes_none(self) -> None:
        extra_info = {"global_step": 10, "max_steps": 100, "train/loss_step": "nan"}
        with patch(f"{REMOTE}.logger") as logger:
            RemoteTrainingBackend._log_training_progress(extra_info)
        logger.info.assert_called_once_with("Training progress: step=10/100 (10%), train/loss_step=None")


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
