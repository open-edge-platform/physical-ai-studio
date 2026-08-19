# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the phase-windowed progress helper shared by remote backends."""

from itertools import pairwise
from unittest.mock import MagicMock

import pytest

from services.training_backends.phase import (
    PHASE_TABLE_VERSION,
    SSH_PHASE_WINDOWS,
    PhaseKey,
    PhaseState,
    PhaseWindow,
    report_phase,
)


def test_ssh_phase_windows_are_contiguous_and_span_0_to_100() -> None:
    """Each phase must own a slice of the bar with no gaps or overlaps."""
    assert SSH_PHASE_WINDOWS[0].start == 0
    assert SSH_PHASE_WINDOWS[-1].end == 100
    for previous, current in pairwise(SSH_PHASE_WINDOWS):
        assert previous.end == current.start


def test_ssh_phase_windows_verify_before_pull_before_trainer_start() -> None:
    """Windows follow the real provisioning order: verify an image's signature
    before ever pulling it, then launch the container."""
    keys = [window.key for window in SSH_PHASE_WINDOWS]
    assert keys == [
        PhaseKey.CONNECT,
        PhaseKey.IMAGE_VERIFY,
        PhaseKey.IMAGE_PULL,
        PhaseKey.TRAINER_START,
        PhaseKey.UPLOAD,
        PhaseKey.TRAIN,
        PhaseKey.DOWNLOAD,
    ]


def test_phase_window_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="Invalid phase window"):
        PhaseWindow(PhaseKey.CONNECT, "Connect", start=10, end=5)


def test_report_phase_maps_sub_progress_into_the_window() -> None:
    reporter = MagicMock()
    windows = (PhaseWindow(PhaseKey.UPLOAD, "Dataset upload", 9, 17),)

    report_phase(reporter, windows, PhaseKey.UPLOAD, sub_progress=50.0, message="halfway")

    reporter.assert_called_once()
    args, kwargs = reporter.call_args
    assert args[0] == 13  # 9 + round(50/100 * 8)
    assert kwargs["message"] == "halfway"
    phase = kwargs["extra_info"]["phase"]
    assert phase == {
        "version": PHASE_TABLE_VERSION,
        "key": "upload",
        "label": "Dataset upload",
        "index": 0,
        "total": 1,
        "state": PhaseState.ACTIVE.value,
        "indeterminate": False,
    }


def test_report_phase_indeterminate_pins_to_window_start() -> None:
    reporter = MagicMock()
    windows = (PhaseWindow(PhaseKey.IMAGE_PULL, "Image pull", 2, 5),)

    report_phase(reporter, windows, PhaseKey.IMAGE_PULL, sub_progress=None, message="Pulling image")

    args, kwargs = reporter.call_args
    assert args[0] == 2
    assert kwargs["extra_info"]["phase"]["indeterminate"] is True


def test_report_phase_merges_caller_extra_info() -> None:
    reporter = MagicMock()
    windows = (PhaseWindow(PhaseKey.TRAIN, "Training", 17, 96),)

    report_phase(reporter, windows, PhaseKey.TRAIN, sub_progress=0.0, extra_info={"loss": 0.5})

    _, kwargs = reporter.call_args
    assert kwargs["extra_info"]["loss"] == 0.5
    assert "phase" in kwargs["extra_info"]


def test_report_phase_rejects_a_key_not_in_the_table() -> None:
    reporter = MagicMock()
    windows = (PhaseWindow(PhaseKey.CONNECT, "Connect", 0, 2),)

    with pytest.raises(ValueError, match="not part of"):
        report_phase(reporter, windows, PhaseKey.TRAIN)


def test_local_and_direct_url_phase_windows_are_untouched() -> None:
    """This module must not retune the shared local/direct-URL progress constants."""
    from services.training_backends.remote import SNAPSHOT_UPLOAD_PROGRESS, TRAINING_PROGRESS_END

    assert SNAPSHOT_UPLOAD_PROGRESS == 10
    assert TRAINING_PROGRESS_END == 95
