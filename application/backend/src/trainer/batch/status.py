# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``status.json`` document and the throttled writer that publishes it."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

from trainer.schemas import TrainerJobStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from trainer.batch.storage import JobStorage


class BatchJobStatus(BaseModel):
    """Contents of ``status.json``; Studio polls this while the job runs."""

    status: TrainerJobStatus
    progress: int = Field(default=0, ge=0, le=100)
    message: str | None = None
    extra_info: dict[str, Any] | None = None
    updated_at: datetime


class StatusReporter:
    """Throttle progress writes to ``status.json``; terminal states always flush."""

    def __init__(self, storage: JobStorage, interval_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._storage = storage
        self._interval_s = interval_s
        self._clock = clock
        self._last_write: float | None = None
        self._lock = threading.Lock()

    def report(self, progress: int, message: str | None, extra_info: dict[str, Any] | None) -> None:
        """``ProgressFn``-compatible callback for ``run_training_job``."""
        self._write(TrainerJobStatus.RUNNING, progress, message, extra_info, force=False)

    def finish(self, status: TrainerJobStatus, *, progress: int, message: str | None = None) -> None:
        self._write(status, progress, message, None, force=True)

    def _write(
        self,
        status: TrainerJobStatus,
        progress: int,
        message: str | None,
        extra_info: dict[str, Any] | None,
        *,
        force: bool,
    ) -> None:
        with self._lock:
            now = self._clock()
            if not force and self._last_write is not None and now - self._last_write < self._interval_s:
                return
            self._last_write = now
        payload = BatchJobStatus(
            status=status, progress=progress, message=message, extra_info=extra_info, updated_at=datetime.now(tz=UTC)
        )
        try:
            self._storage.put_status(payload)
        except Exception:
            logger.opt(exception=True).warning("Failed to write status.json")
