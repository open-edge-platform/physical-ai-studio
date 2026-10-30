"""Record the session's store into a dataset, in the dataset worker's own process.

The worker attaches to the store and opens the dataset's recording mutation,
which works on a copy of the dataset in the cache. While an episode is open it
samples the store at the dataset's fps and appends a frame. It owns its
commands and its state: it serves ``start_episode``, ``save_episode`` and
``discard_episode`` on the bus (see ``command_key``), replies with its new
status, and publishes every change on ``evt/dataset`` for the session to show.

Its own process, because lerobot writes parquet and encodes video in Python,
which would otherwise compete with the API process for the GIL.

On exit an open episode is discarded, then the cache is copied back over the
dataset. That copy can take a while, and killing the worker in the middle of it
loses the dataset, so the parent waits long for it (``stop_timeout_s``).
"""

from __future__ import annotations

from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel

from exceptions import RuntimeSessionCommandError
from runtime_session.bus import Bus, command_key
from runtime_session.store import ACTION, STATE, FeatureSpec, Store
from schemas.runtime_session import DiscardEpisodeCommand, SaveEpisodeCommand, StartEpisodeCommand
from workers.base import ProcessWorker
from workers.rate import Rate

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from internal_datasets.mutations.recording_mutation import RecordingMutation
    from schemas.dataset import Dataset

STATUS = "rpc/dataset/status"
EVENTS = "evt/dataset"
MIN_EPISODE_FRAMES = 2
"""lerobot's streaming encoder gives a 1-frame episode no video stats; it saves, then closing the dataset fails."""


class Empty(BaseModel):
    pass


class RecordingStatus(BaseModel):
    is_recording: bool
    episodes_recorded: int
    """Episodes saved by this worker, not yet copied back into the dataset."""


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
    _open: bool = field(default=False, repr=False)

    def status(self, _request: Empty | None = None) -> RecordingStatus:
        return RecordingStatus(is_recording=self._open, episodes_recorded=self.episodes_recorded)

    def start(self, request: StartEpisodeCommand) -> RecordingStatus:
        if self._open:
            raise RuntimeSessionCommandError("An episode is already being recorded.", error_code="episode_open")
        if not request.task:
            raise RuntimeSessionCommandError("A task is required to record an episode.", error_code="task_required")
        self.task, self._open, self.frames = request.task, True, 0
        return self._changed()

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
            with suppress(Exception):
                self.mutation.discard_buffer()
            self._changed()  # not recording any more, whether or not the save landed
            raise
        self.episodes_recorded += 1
        return self._changed()

    def discard(self, _request: DiscardEpisodeCommand | None = None) -> RecordingStatus:
        # Does not require an open episode: also how a too-short one is thrown away.
        self._open = False
        self.mutation.discard_buffer()
        return self._changed()

    def _changed(self) -> RecordingStatus:
        status = self.status()
        self.publish(status)
        return status

    def record_frame(self) -> None:
        """Append the store's current values, if an episode is open and every feature has one."""
        if not self._open or self.task is None:
            return
        state, action = self.store.read(STATE), self.store.read(ACTION)
        frames = {key: self.store.read_observation(key) for key in self.store.spec.observations}
        if state is None or action is None or any(frame is None for frame in frames.values()):
            # A row is all or nothing: a default where a measurement belongs would be indistinguishable from data.
            self.skipped += 1
            return
        observation: dict[str, Any] = dict(zip(state.names, map(float, state.value), strict=True))
        observation.update({key: frame.value for key, frame in frames.items() if frame is not None})
        try:
            self.mutation.add_frame(
                observation, dict(zip(action.names, map(float, action.value), strict=True)), self.task
            )
        except Exception:
            # Lose this episode, not the worker: a crash would end the recording without finalizing it.
            logger.exception("Recording a frame failed; discarding the episode")
            with suppress(Exception):
                self.discard()
            self._open = False
            self._changed()
            return
        self.frames += 1

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
                recorder = Recorder(self.open_mutation(), store, publish=lambda status: bus.publish(EVENTS, status))
                try:
                    bus.serve(STATUS, Empty, recorder.status)
                    bus.serve(command_key("start_episode"), StartEpisodeCommand, recorder.start)
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
