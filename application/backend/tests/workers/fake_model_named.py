"""The fake model worker under the real worker's class name, for the log file tests.

Records are routed to a log file by the worker's class name, so this one logs
where a real ``ModelWorker`` does. Module-level so the spawned child can import it.
"""

from __future__ import annotations

from .fake_model import FakeModelWorker


class ModelWorker(FakeModelWorker):
    pass
