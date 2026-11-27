"""The fake model worker under the real worker's class name, for the log file tests.

Records are routed to a log file by the worker's class name, so this one logs
where a real ``ModelWorker`` does. Module-level so the spawned child can import it.
"""

from __future__ import annotations

from .fake_model import FakeModelWorker, Oracle

SENTINEL = "sentinel-local-7f3a9c"
"""A local on the failing call's line, which Loguru's ``diagnose`` would print in the traceback."""


class ModelWorker(FakeModelWorker):
    def open_model(self) -> Oracle:
        sentinel = SENTINEL
        return self._open_with(sentinel)

    def _open_with(self, _sentinel: str) -> Oracle:
        return super().open_model()
