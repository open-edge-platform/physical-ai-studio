"""Fake workers for the worker base tests.

Module-level and configured through plain arguments, because a process worker
is pickled into a spawned child that imports this module. They report what
happened through marker files, which is the one channel that works the same
for a thread and a process.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from workers.base import ProcessWorker, ThreadWorker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

Mode = Literal["ok", "fail_acquire", "fail_run", "ignore_stop", "exit_hard", "slow_ready"]


class _FakeBehaviour:
    """Acquire a pretend device, write markers, and misbehave on request."""

    marker_dir: Path
    mode: Mode
    name: str

    def _configure(self, marker_dir: Path, mode: Mode) -> None:
        self.marker_dir = marker_dir
        self.mode = mode

    def _mark(self, marker: str, content: str = "") -> None:
        (self.marker_dir / marker).write_text(content)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[str]:
        self._mark("acquired")
        try:
            if self.mode == "fail_acquire":
                raise RuntimeError("no device")
            if self.mode == "exit_hard":
                os._exit(3)
            if self.mode == "slow_ready":
                await asyncio.sleep(1.0)
            yield f"device-of-{self.name}"
        finally:
            self._mark("released")

    async def run_loop(self, context: str) -> None:
        self._mark("context", context)
        while not self.should_stop():  # type: ignore[attr-defined]
            if self.mode == "fail_run":
                raise ValueError("lost device")
            if self.mode == "ignore_stop":
                # Blocks the loop, like a driver call that never returns.
                time.sleep(2.0)  # noqa: ASYNC251
                return
            await asyncio.sleep(0.01)


class FakeThreadWorker(_FakeBehaviour, ThreadWorker[str]):
    def __init__(self, marker_dir: Path, mode: Mode = "ok") -> None:
        super().__init__(name=f"fake-thread-{mode}")
        self._configure(marker_dir, mode)


class FakeProcessWorker(_FakeBehaviour, ProcessWorker[str]):
    def __init__(self, marker_dir: Path, mode: Mode = "ok") -> None:
        super().__init__(name=f"fake-process-{mode}")
        self._configure(marker_dir, mode)
