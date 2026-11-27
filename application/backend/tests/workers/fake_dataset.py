"""A dataset worker that records into a JSON log instead of lerobot, for the recording tests.

Module-level so the spawned child can import it. The child writes the log; the
test reads it back.
"""

from __future__ import annotations

import json
from pathlib import Path

from internal_datasets.mutations.recording_mutation import (
    RecordingCleanupError,
    RecordingNotSavedError,
    UnusedCacheCleanupError,
)
from workers.dataset_worker import DatasetWorker


class FakeMutation:
    def __init__(self, log: Path) -> None:
        self.log = log
        self.frames = 0
        self.saved = False

    def _write(self, event: str, **fields: object) -> None:
        with self.log.open("a") as log:
            log.write(json.dumps({"event": event, **fields}) + "\n")

    def _fails(self, what: str) -> bool:
        """A test makes an operation fail by touching ``fail_<what>`` next to the log."""
        return (self.log.parent / f"fail_{what}").exists()

    def add_frame(self, obs: dict, act: dict, task: str) -> None:
        if self._fails("frame"):
            raise ValueError("a frame of the wrong shape")
        if self.frames == 0:
            cameras = {key: list(value.shape) for key, value in obs.items() if hasattr(value, "shape")}
            joints = {key: value for key, value in obs.items() if not hasattr(value, "shape")}
            self._write("first_frame", task=task, joints=joints, action=act, cameras=cameras)
        self.frames += 1

    def save_episode(self) -> None:
        if self._fails("save"):
            raise OSError("the disk is full")
        self._write("save", frames=self.frames)
        self.frames = 0
        self.saved = True

    def discard_buffer(self) -> None:
        if self._fails("discard"):
            raise OSError("the episode's files are gone")
        self._write("discard", frames=self.frames)
        self.frames = 0

    def teardown(self) -> None:
        if self._fails("teardown"):  # as RecordingMutation reports a failed finalize, before any copy
            raise RecordingNotSavedError(
                "Finalizing failed (No space left on device). What was recorded is left in /cache/x"
            )
        if self._fails("worker"):  # something else breaks: nobody can say whether the copy landed
            raise ValueError("the worker broke")
        self._write("teardown")
        if self._fails("cleanup"):  # as RecordingMutation: whether anything was copied before
            if self.saved:
                raise RecordingCleanupError("its cache /cache/x could not be removed (Permission denied)")
            raise UnusedCacheCleanupError("the cache /cache/x could not be removed (Permission denied)")


class FakeDatasetWorker(DatasetWorker):
    def open_mutation(self) -> FakeMutation:  # type: ignore[override]
        return FakeMutation(Path(self.dataset.path) / "recording.jsonl")

    def free_bytes(self) -> int:
        """What a test wrote to ``free_bytes`` next to the log, or plenty."""
        free = Path(self.dataset.path) / "free_bytes"
        return int(free.read_text()) if free.exists() else 100 * 1024**3


def recording_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class DatasetThatWillNotReopenWorker(DatasetWorker):
    """The real worker and mutation through lerobot; once every file is copied back, the dataset will not open."""

    def open_mutation(self):  # type: ignore[override]
        mutation = super().open_mutation()

        def unreadable() -> None:
            raise OSError("meta/info.json is unreadable")

        mutation.source_dataset.load_dataset = unreadable  # type: ignore[method-assign]
        return mutation
