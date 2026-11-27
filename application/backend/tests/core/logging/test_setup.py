# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the worker log sinks that setup_logging adds."""

from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from core.logging.log_config import LogConfig

logging_setup = import_module("core.logging.setup")


class _RecordingLogger:
    """Stands in for Loguru's logger and records the options of each sink."""

    def __init__(self) -> None:
        self.sinks: dict[str, dict[str, Any]] = {}

    def remove(self) -> None:
        pass

    def add(self, sink: object, **options: Any) -> int:
        self.sinks[Path(str(sink)).name] = options
        return len(self.sinks)

    def warning(self, message: str) -> None:
        raise AssertionError(message)

    error = warning


def test_only_listed_workers_turn_off_diagnose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(logging_setup, "logger", recorder)
    monkeypatch.setattr(logging_setup, "global_log_config", logging_setup.global_log_config)
    monkeypatch.setattr(logging_setup, "_quiet_noisy_loggers", lambda: None)

    logging_setup.setup_logging(LogConfig(log_folder=tmp_path))

    for worker_name, log_file in LogConfig.worker_log_info.items():
        options = recorder.sinks[Path(log_file).name]
        if worker_name in LogConfig.workers_without_locals:
            assert options["diagnose"] is False
        else:
            # Left unset, so Loguru's own default (LOGURU_DIAGNOSE) applies as before.
            assert "diagnose" not in options
    assert recorder.sinks["inference.log"]["diagnose"] is False
