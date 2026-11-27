"""Record the session's store into a dataset, in the dataset worker's own process.

The worker attaches to the store and opens the dataset's recording mutation,
which works on a copy of the dataset in the cache. While an episode is open it
samples the store at the dataset's fps and appends a frame. It owns its
commands and its state: it serves ``save_episode`` and ``discard_episode`` on
the bus (see ``command_key``), and ``START`` for the session, which serves
``start_episode`` once it has checked the environment. It
replies with its new status, and publishes every change on ``evt/dataset`` for
the session to show, with a reason when the user must know an episode was
lost.

Its own process, because lerobot writes parquet and encodes video in Python,
which would otherwise compete with the API process for the GIL.

On exit an open episode is discarded, then the cache is copied back over the
dataset. That copy can take a while, and killing the worker in the middle of it
loses the dataset, so the parent waits long for it (``stop_timeout_s``).
"""

from __future__ import annotations

import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel

from exceptions import RuntimeSessionCommandError
from runtime_session.bus import DEADLINE, Bus, command_key, status_key
from runtime_session.store import ACTION, STATE, FeatureSpec, Store
from schemas.runtime_session import DiscardEpisodeCommand, SaveEpisodeCommand, StartEpisodeCommand, StatusMessage
from workers.base import ProcessWorker
from workers.rate import Rate

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable

    from internal_datasets.mutations.recording_mutation import RecordingMutation
    from schemas.dataset import Dataset

STATUS = "rpc/dataset/status"
START = "rpc/dataset/start_episode"
"""Served for the session, which serves ``start_episode`` to clients after its own checks."""
EVENTS = "evt/dataset"
MIN_EPISODE_FRAMES = 2
"""lerobot's streaming encoder gives a 1-frame episode no video stats; it saves, then closing the dataset fails."""
MIN_FREE_BYTES_TO_START = 2 * 1024**3
"""An episode starts only with this much free on the disks the recording writes to, measured when it starts."""


def _gib(size: int) -> str:
    return f"{size / 1024**3:.1f} GB"


class Empty(BaseModel):
    pass


class RecordingStatus(BaseModel):
    is_recording: bool
    episodes_recorded: int
    """Episodes saved by this worker, not yet copied back into the dataset."""
    error: str | None = None
    """Why an episode was automatically discarded, on the event announcing that discard."""


@dataclass
class Recorder:
    """The worker's recording state; only touched on the worker's event loop."""

    mutation: RecordingMutation
    store: Store
    publish: Callable[[RecordingStatus], None]
    """Announces every change on ``evt/dataset``."""
    task: str | None = None
    episodes_recorded: int = 0
    skipped: int = 0
    frames: int = 0
    """In the open episode."""
    free_bytes: Callable[[], int] | None = None
    """Free space on the disks the recording writes to; ``None`` skips the check at Start."""
    _open: bool = field(default=False, repr=False)
    _uncleared: str | None = field(default=None, repr=False)
    """Why clearing a lost episode's frames failed, until a later clear works: no episode starts meanwhile, or it
    would carry those frames."""

    def status(self, _request: Empty | None = None) -> RecordingStatus:
        return RecordingStatus(is_recording=self._open, episodes_recorded=self.episodes_recorded)

    def start(self, request: StartEpisodeCommand) -> RecordingStatus:
        if self._open:
            raise RuntimeSessionCommandError("An episode is already being recorded.", error_code="episode_open")
        if self._uncleared is not None and not self._clear():
            raise RuntimeSessionCommandError(
                f"The frames of the last lost episode could not be cleared ({self._uncleared}), so a new episode "
                "would include them. Finish the recording and start it again.",
                error_code="buffer_not_cleared",
            )
        if not request.task:
            raise RuntimeSessionCommandError("A task is required to record an episode.", error_code="task_required")
        if (free := self._free()) is not None and free < MIN_FREE_BYTES_TO_START:
            raise RuntimeSessionCommandError(
                f"Only {_gib(free)} is free on the disk the recording is written to. "
                f"Free at least {_gib(MIN_FREE_BYTES_TO_START)} before recording.",
                error_code="low_disk",
            )
        if (deadline := DEADLINE.get()) is not None and time.monotonic() >= deadline:
            # The client has stopped waiting and was told the start timed out: do not open an episode behind it.
            raise RuntimeSessionCommandError("start_episode did not finish in time", error_code="command_timeout")
        self.task, self._open, self.frames = request.task, True, 0
        return self._changed()

    def _free(self) -> int | None:
        """Free bytes on the recording's disks, or ``None`` when not checked or not known (then Start goes ahead)."""
        if self.free_bytes is None:
            return None
        try:
            return self.free_bytes()
        except Exception:
            logger.opt(exception=True).warning("Could not check the free disk space; starting without the check")
            return None

    def save(self, _request: SaveEpisodeCommand) -> RecordingStatus:
        if not self._open:
            raise RuntimeSessionCommandError("No episode is being recorded.", error_code="no_episode")
        self._open = False
        if self.frames < MIN_EPISODE_FRAMES:
            self.discard()
            raise RuntimeSessionCommandError(
                f"The episode has {self.frames} frame(s), too short to save; it was discarded.",
                error_code="episode_too_short",
            )
        try:
            self.mutation.save_episode()
        except Exception:
            # lerobot consumes its buffer as it saves: a failed save can't be retried, and the next
            # episode would append to the remains. Clear it, so only this episode is lost.
            self._clear()
            self._changed()  # not recording any more, whether or not the save landed
            raise
        self.episodes_recorded += 1
        return self._changed()

    def discard(self, _request: DiscardEpisodeCommand | None = None, *, error: str | None = None) -> RecordingStatus:
        # Does not require an open episode: also how a too-short one is thrown away.
        self._open = False
        try:
            self.mutation.discard_buffer()
        except Exception as failure:
            self._uncleared = str(failure) or type(failure).__name__
            raise
        self._uncleared = None
        return self._changed(error)

    def _changed(self, error: str | None = None) -> RecordingStatus:
        status = self.status()
        status.error = error
        self.publish(status)
        return status

    def camera_status(self, status: StatusMessage) -> None:
        """Discard on camera loss even if the feed recovers between recording ticks."""
        key = status.component.removeprefix("environment/")
        if key in self.store.spec.observations and status.state in {"starting", "failed"}:
            self._camera_lost(key)

    def _camera_lost(self, key: str) -> None:
        if self._open:
            error = (
                f'Episode discarded because camera "{key}" stopped sending frames. '
                "Start a new episode after it reconnects."
            )
            logger.warning(error)
            self.discard(error=error)

    def record_frame(self) -> None:
        """Append the store's current values, if an episode is open and every feature has one."""
        if not self._open or self.task is None:
            return
        state, action = self.store.read(STATE), self.store.read(ACTION)
        frames = {key: self.store.read_observation(key) for key in self.store.spec.observations}
        for key, frame in frames.items():
            if frame is None:
                self._camera_lost(key)
                return
        if state is None or action is None:
            # A row is all or nothing: a default where a measurement belongs would be indistinguishable from data.
            self.skipped += 1
            return
        observation: dict[str, Any] = state.named_values()
        observation.update({key: frame.value for key, frame in frames.items() if frame is not None})
        try:
            self.mutation.add_frame(observation, action.named_values(), self.task)
        except Exception as error:
            # Lose this episode, not the worker: a crash would end the recording without finalizing it.
            logger.exception("Recording a frame failed; discarding the episode")
            self._open = False
            message = f"Episode discarded because recording a frame failed: {error}."
            if self._clear():
                message += " Start a new episode."
            else:
                message += (
                    f" Clearing its frames failed too ({self._uncleared}); no new episode starts until they are "
                    "cleared. Finish the recording and start it again."
                )
            self._changed(message)
            return
        self.frames += 1

    def _clear(self) -> bool:
        """Clear the episode buffer; whether that worked. A failure is kept in ``_uncleared``."""
        try:
            self.mutation.discard_buffer()
        except Exception as failure:
            logger.exception("Clearing the episode buffer failed")
            self._uncleared = str(failure) or type(failure).__name__
            return False
        self._uncleared = None
        return True

    def finish(self) -> None:
        """Discard an open episode, then copy the recording back over the dataset."""
        if self._open:
            logger.warning("Discarding the episode that was still open when recording stopped")
            try:
                self.discard()
            except Exception:
                # Finalizing still saves every episode before it; only the open one is lost.
                logger.exception("Discarding the open episode failed")
        if self.skipped:
            logger.info("Recording skipped {} ticks without a complete row", self.skipped)
        self.mutation.teardown()


class DatasetWorker(ProcessWorker[Recorder]):
    ready_timeout_s = 300.0
    """Opening copies the dataset into the cache, which takes as long as the dataset is big."""
    stop_timeout_s = 300.0
    """Closing copies the cache back; killing the worker during that copy loses the dataset."""

    def __init__(
        self, dataset: Dataset, *, session_id: str, store_name: str, spec: FeatureSpec, fps: int, robot_type: str
    ) -> None:
        super().__init__(name=f"DatasetWorker-{dataset.id}")
        self.dataset = dataset
        self.session_id = session_id
        self.store_name = store_name
        self.spec = spec
        self.fps = fps
        self.robot_type = robot_type

    def free_bytes(self) -> int:
        """Runs in the child: free space on the cache disk the episodes go to and the disk they are copied back to."""
        from settings import get_settings

        return lowest_free_bytes((get_settings().cache_dir, Path(self.dataset.path)))

    def open_mutation(self) -> RecordingMutation:
        """Runs in the child. Tests replace this to record without lerobot."""
        from internal_datasets.utils import get_internal_recording_dataset
        from runtime_session.dataset import lerobot_features

        return get_internal_recording_dataset(self.dataset).start_recording_mutation(
            fps=self.fps, features=lerobot_features(self.spec), robot_type=self.robot_type
        )

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[Recorder]:
        store = Store.attach(self.spec, self.store_name)
        try:
            with Bus.join(self.session_id) as bus:
                recorder = Recorder(
                    self.open_mutation(),
                    store,
                    publish=lambda status: bus.publish(EVENTS, status),
                    free_bytes=self.free_bytes,
                )
                try:
                    bus.listen(status_key("environment/*"), StatusMessage, recorder.camera_status)
                    bus.serve(STATUS, Empty, recorder.status)
                    bus.serve(START, StartEpisodeCommand, recorder.start)
                    bus.serve(command_key("save_episode"), SaveEpisodeCommand, recorder.save)
                    bus.serve(command_key("discard_episode"), DiscardEpisodeCommand, recorder.discard)
                    yield recorder
                finally:
                    recorder.finish()
        finally:
            store.close()

    async def run_loop(self, context: Recorder) -> None:
        rate = Rate(self.fps, name=self.name)
        while not self.should_stop():
            context.record_frame()
            await rate.sleep()


def lowest_free_bytes(paths: Iterable[Path]) -> int:
    """The least free space among the disks of ``paths``. Each is probed on its own, so one that cannot be read
    does not hide a full one.

    Raises:
        OSError: None of them could be read.
    """
    free: list[int] = []
    failures: list[OSError] = []
    for path in paths:
        try:
            free.append(shutil.disk_usage(_existing(path)).free)
        except OSError as error:
            logger.warning("Could not check the free space for {}: {}", path, error)
            failures.append(error)
    if not free:
        raise failures[0] if failures else OSError("no path to check")
    return min(free)


def _existing(path: Path) -> Path:
    """``path``, or its nearest ancestor that exists: where it will be created."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return path
