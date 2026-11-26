"""Autopilot recording: record a simulation's demonstrator into the loaded dataset.

The simulation knows the task: it lays out each episode, runs its demonstrator and decides how the
episode ended. The session owns the arm and the dataset. They meet on the plugin SDK's autopilot
protocol (``physicalai_studio_plugin.autopilot``). In leader mode the simulation never moves the arm:
its demonstrator publishes targets on the simulation's leader, and only the session's control writes
them to the follower. So the session's control slot alone decides who drives the arm.

Two parts, both in the API process:

- :class:`AutopilotControl`, in the session's control slot. Each tick it reads the leader that the robot
  type's ``AutopilotSpec.leader_for`` builds and writes its targets as the follower's action. Those reads
  are also the simulation's sign of life: when they stop for ``LEADER_TIMEOUT_S``, the simulation aborts
  its episode by itself. The control notes when the leader, the follower and every camera last sent a
  new sample (:class:`Freshness`), and whether the leader's sample is of the episode being recorded.
- :class:`AutopilotRecording`, an asyncio loop the session owns. For each episode it asks the simulation
  for one, waits until it is ready and every camera is fresh, starts the dataset episode with the
  episode's task, begins it, polls until it ended, then saves or discards it. It never holds the
  session's lock.

Only successes with at least :data:`MIN_FRAMES` frames are kept, a success after a regrasp included. A
leader or follower without a new sample for :data:`STALE_AFTER_S` halts the follower and fails the
recording; a stale camera discards the episode.

Stopping (the user, a control change, an environment or dataset change, a simulation restart, teardown)
halts the follower at once, aborts the simulation's episode (best effort, within :data:`ABORT_TIMEOUT_S`),
discards the open dataset episode and reports. A save already in progress finishes first.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from loguru import logger
from physicalai_studio_plugin.autopilot import (
    AutopilotClient,
    AutopilotRefusedError,
    AutopilotUnreachableError,
    EpisodeOptions,
)

from exceptions import RuntimeSessionCommandError
from robots.features import feature_names
from runtime_session.bus import BusRequestError, command_key
from runtime_session.controls.base import Control
from runtime_session.store import ACTION, STATE
from schemas.runtime_session import (
    AutopilotControlConfig,
    AutopilotEpisodeSummary,
    AutopilotRecordingState,
    DiscardEpisodeCommand,
    SaveEpisodeCommand,
    StartEpisodeCommand,
)
from workers.camera_worker import CameraWorker
from workers.dataset_worker import STATUS, Empty, RecordingStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from uuid import UUID

    import httpx
    from physicalai.robot import Robot
    from physicalai_studio_plugin.autopilot import AutopilotSpec, Episode
    from pydantic import BaseModel

    from runtime_session.bus import Bus
    from runtime_session.store import Store
    from schemas.runtime_session import AutopilotRecordingOptions, DiscardReason

STALE_AFTER_S = CameraWorker.stale_after_s
"""The most the leader, the follower or a camera may go without a new sample (0.5 s), as the cameras' own limit."""
FIRST_SAMPLE_S = 2.0
"""How long the leader and the follower may take to send their first sample once the control starts."""
MIN_FRAMES = 10
"""Fewer frames than this and a success is discarded (``too_short``)."""
POLL_S = 0.1
READY_TIMEOUT_S = 30.0
"""How long the simulation may take to lay out an episode."""
FRESH_TIMEOUT_S = 5.0
"""How long a ready episode waits for every camera to be fresh before it is discarded."""
MAX_STALE_STARTS = 3
"""Ready episodes in a row whose cameras are not fresh, after which the recording fails."""
ABORT_TIMEOUT_S = 1.0
HTTP_TIMEOUT_S = 2.0
UNANSWERED_S = 5.0
"""How long the simulation may leave polls of a running episode unanswered before the recording fails."""
DATASET_TIMEOUT_S = 10.0
SAVE_TIMEOUT_S = 300.0
"""Saving encodes the episode's video in the dataset worker."""

LEADER = "leader"
FOLLOWER = "follower"
_MOTION = (LEADER, FOLLOWER)
_NO_EPISODE_ID = object()
"""A leader driver that does not say which episode its sample is of: nothing to check."""
_DISCARDS: dict[str | None, DiscardReason] = {"failure": "failure", "stalled": "stalled"}
"""An outcome that is not a success, as the discard reason it counts under; any other is ``aborted``."""

Phase = Literal["done", "stopped", "failed"]


# ----------------------------------------------------------------------
# The control
# ----------------------------------------------------------------------


class Freshness:
    """When each stream last sent a new sample: the leader, the follower's state and every camera.

    A sample is new when its timestamp changes; when is measured on this process's clock, so it holds for
    any device's clock. The first timestamp seen is only a baseline: the store may still hold a sample of
    the devices before (a swap or reload keeps the frames). Cameras are kept apart from the leader and the
    follower, so a camera named ``leader`` is its own stream. The control's thread writes, the loop reads:
    each write replaces one dict entry.
    """

    def __init__(self, cameras: Iterable[str], clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.cameras = tuple(cameras)
        self.started = clock()
        self._stamps: dict[tuple[bool, str], float] = {}
        self._new_at: dict[tuple[bool, str], float] = {}

    def restart(self) -> None:
        self.started = self._clock()

    def observe(self, stream: str, stamp: float | None, *, camera: bool = False) -> None:
        key = (camera, stream)
        if stamp is None or stamp == (seen := self._stamps.get(key)):
            return
        self._stamps[key] = stamp
        if seen is not None:
            self._new_at[key] = self._clock()

    def stale(self, streams: Iterable[str], *, camera: bool = False, grace_s: float = 0.0) -> list[str]:
        """The streams without a new sample for longer than :data:`STALE_AFTER_S`, each with its gap.

        A stream never seen counts once *grace_s* passed since the control started.
        """
        now = self._clock()
        stale = []
        for stream in streams:
            new_at = self._new_at.get((camera, stream))
            if new_at is None:
                if now - self.started > grace_s:
                    stale.append(f"{stream} (never)")
            elif now - new_at > STALE_AFTER_S:
                stale.append(f"{stream} ({now - new_at:.2f} s)")
        return stale


def _no_samples(stale: list[str]) -> str:
    return "No new samples from " + ", ".join(stale)


class AutopilotControl(Control):
    """Command the follower whatever the simulation's demonstrator publishes on its leader.

    Each tick reads the leader, notes every stream's newest sample, and writes the leader's target. A
    leader or follower that went stale holds the follower and says why (:attr:`fault`). While the loop
    records an episode (:meth:`watch`), a leader sample of another episode is noted (:attr:`mismatch`).
    :meth:`halt` holds the follower at once, from any thread, and no target is written after it. A
    leader that fails to read ends the control. The leader is built for this control: it disconnects it.
    """

    def __init__(self, store: Store, leader: Robot, config: AutopilotControlConfig) -> None:
        leader_joints = tuple(feature_names(list(leader.joint_names)))
        if leader_joints != store.spec.joint_names:
            raise ValueError(
                f"The demonstrator's joints {leader_joints} do not match the follower's {store.spec.joint_names}"
            )
        super().__init__(store, config, name="AutopilotControl", hz=config.hz)
        self.leader = leader
        self.freshness = Freshness(store.spec.observations)
        self.fault: str | None = None
        self.mismatch: tuple[str, str] | None = None
        """The episode being recorded, and the episode a leader sample was of instead."""
        self._episode_id: str | None = None
        self._halted = threading.Event()
        self._write = threading.Lock()
        """Taken for every action write, so a hold is never overwritten by a target read before it."""

    def watch(self, episode_id: str | None) -> None:
        """Check from now on that every leader sample is of *episode_id*; ``None`` checks nothing."""
        self.mismatch = None
        self._episode_id = episode_id

    def halt(self) -> None:
        """Hold the follower where it is, and write no target after; safe from any thread.

        Where it is: its last state, as the session's hold, since a follower behind its last target would
        go on moving to it. The last action written when it has no state yet.
        """
        with self._write:
            if self._halted.is_set():
                return
            self._halted.set()
            if (held := self.store.read(STATE) or self.store.read(ACTION)) is not None:
                self.store.write(ACTION, held.value, timestamp=time.monotonic())

    @property
    def halted(self) -> bool:
        return self._halted.is_set()

    def on_start(self) -> None:
        # Like teleop's: a connection idle since connecting may fail its first read, then reconnects.
        try:
            self.leader.get_observation()
        except ConnectionError as error:
            logger.info("The demonstrator's first reading failed ({}); reading it again", error)
            self.leader.get_observation()
        self.freshness.restart()

    def on_stop(self) -> None:
        with contextlib.suppress(Exception):
            self.leader.disconnect()

    def on_tick(self) -> None:
        if self._halted.is_set():
            return
        observation = self.leader.get_observation()
        freshness = self.freshness
        freshness.observe(LEADER, observation.timestamp)
        freshness.observe(FOLLOWER, self.store.stamp(STATE))
        for camera in freshness.cameras:
            # The stamp only: frames are big.
            freshness.observe(camera, self.store.observation_stamp(camera), camera=True)
        expected = self._episode_id
        sample_episode = getattr(self.leader, "episode_id", _NO_EPISODE_ID)  # the sample's ``episode_id``
        if expected is not None and sample_episode is not _NO_EPISODE_ID and sample_episode != expected:
            self.mismatch = (expected, str(sample_episode) if sample_episode is not None else "")
        if stale := freshness.stale(_MOTION, grace_s=FIRST_SAMPLE_S):
            self.fault = _no_samples(stale)
            self.halt()
            return
        with self._write:
            if self._halted.is_set():  # halted while reading: never write after a hold
                return
            self.store.write(ACTION, observation.joint_positions, timestamp=observation.timestamp + 1 / self.hz)


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------


class _Ended(Exception):
    """The recording cannot go on."""

    def __init__(self, message: str, phase: Phase = "failed") -> None:
        super().__init__(message)
        self.message = message
        self.phase = phase


@dataclass
class _Open:
    """The episode the loop works on, as the simulation last reported it."""

    episode: Episode
    dataset_episode: str | None = None
    """The dataset episode's id while the dataset worker has it open."""


class AutopilotRecording:
    """One autopilot recording in a session; see the module docstring.

    :meth:`open` checks the simulation and builds the control, :meth:`start` runs the loop in the
    background, :meth:`stop` ends it. ``on_end`` is called once, from the loop's task, when it ends by
    itself.
    """

    def __init__(  # noqa: PLR0913 - what one recording needs
        self,
        *,
        bus: Bus,
        store: Store,
        dataset_id: UUID,
        http_url: str,
        spec: AutopilotSpec[Any],
        payload: Any,
        options: AutopilotRecordingOptions,
        emit: Callable[[], None],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.bus = bus
        self.store = store
        self.dataset_id = dataset_id
        self.http_url = http_url
        self.spec = spec
        self.payload = payload
        self.options = options
        self._emit = emit
        self.client = AutopilotClient(http_url, spec.path, timeout_s=HTTP_TIMEOUT_S, async_transport=transport)
        self.state = AutopilotRecordingState(phase="starting", max_episodes=options.max_episodes)
        self.control: AutopilotControl | None = None
        self._open: _Open | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping: tuple[str, bool] | None = None
        self._finishing = False
        self._on_end: Callable[[AutopilotRecording], None] | None = None

    @property
    def running(self) -> bool:
        """Until the loop finished, its stop included."""
        return self.state.phase not in {"done", "stopped", "failed"}

    def _episode_options(self) -> EpisodeOptions:
        return EpisodeOptions(max_attempts=self.options.max_attempts, recovery_fraction=self.options.recovery_fraction)

    # -- opening ---------------------------------------------------------

    async def open(self) -> AutopilotControl:
        """Check that the simulation's scene has an autopilot, and connect its leader.

        Returns:
            The control for the session to run in its control slot.

        Raises:
            RuntimeSessionCommandError: Why not; nothing is left open.
        """
        try:
            try:
                info = await self.client.ainfo()
            except (AutopilotUnreachableError, AutopilotRefusedError) as error:
                raise _refused(f"The simulation does not answer for its autopilot: {error}") from error
            if not info.supported:
                raise _refused(info.reason or f"Scene {info.scene!r} has no autopilot.")
            if unsupported := self._episode_options().set_fields() - set(info.options):
                raise _refused(f"Scene {info.scene!r} does not take {', '.join(sorted(unsupported))}.")
            self.state = self.state.model_copy(update={"scene": info.scene})
            leader = self.spec.leader_for(self.payload, self.http_url)
            try:
                await asyncio.to_thread(leader.connect)
                self.control = AutopilotControl(self.store, leader, AutopilotControlConfig())
            except Exception as error:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(leader.disconnect)
                raise _refused(f"The simulation's leader cannot lead the follower: {error}") from error
        except BaseException:
            await self.client.aclose()
            raise
        return self.control

    async def abandon(self) -> None:
        """Undo :meth:`open` when the control did not start."""
        if self.control is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.control.leader.disconnect)
        await self.client.aclose()
        self.state = self.state.model_copy(update={"phase": "failed", "message": "Its control did not start."})

    # -- running ---------------------------------------------------------

    def start(self, on_end: Callable[[AutopilotRecording], None]) -> None:
        """Run the episodes in the background, until :meth:`stop` or the recording ends by itself."""
        self._on_end = on_end
        self._task = asyncio.create_task(self._run(), name="autopilot-recording")

    async def stop(self, message: str, *, failed: bool = False) -> None:
        """Halt the follower at once, then end the loop and wait until it reported (see the module docstring)."""
        self._halt()
        task = self._task
        if task is None:
            return
        if self._stopping is None:
            self._stopping = (message, failed)
            if not task.done() and not self._finishing:
                self._update(phase="stopping", message=message)
                task.cancel()
        await asyncio.wait({task})
        if not self._finishing:  # cancelled before its first step: ``_run`` never ran, nor its cleanup
            self._finishing = True
            await self._finish("failed" if failed else "stopped", message)

    async def _run(self) -> None:
        phase: Phase = "done"
        message = ""
        try:
            await self._episodes()
            message = f"Saved {self.state.saved} episode(s)."
        except _Ended as ended:
            phase, message = ended.phase, ended.message
        except asyncio.CancelledError:
            message, failed = self._stopping or ("Stopped.", False)
            phase = "failed" if failed else "stopped"
        except Exception as error:
            logger.opt(exception=error).error("Autopilot recording failed")
            phase, message = "failed", _message(error)
        # In a task of its own: a stop may cancel this one again, and the cleanup must finish.
        self._finishing = True
        finishing = asyncio.ensure_future(self._finish(phase, message))
        while not finishing.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(finishing)
        if self._on_end is not None and self._stopping is None:
            self._on_end(self)

    async def _episodes(self) -> None:
        stale_starts = 0
        while self.options.max_episodes == 0 or self.state.saved < self.options.max_episodes:
            self._update(phase="preparing", task=None, message="")
            episode = await self._call(lambda: self.client.astart_episode(self._episode_options()))
            self._open = current = _Open(episode)
            episode = await self._until_ready(current)
            if episode.phase == "ended":  # ended before it began
                self._end(current, discard=_DISCARDS.get(episode.outcome, "aborted"), reason=episode.reason)
                continue
            self._update(task=episode.task)
            if stale := await self._stale_cameras():
                reason = _no_samples(stale)
                await self._abort(current)
                self._end(current, discard="stale_stream", reason=reason)
                stale_starts += 1
                if stale_starts >= MAX_STALE_STARTS:
                    raise _Ended(f"{reason}, before {stale_starts} episodes in a row.")
                continue
            stale_starts = 0
            await self._record(current)

    async def _until_ready(self, current: _Open) -> Episode:
        deadline = time.monotonic() + READY_TIMEOUT_S
        episode_id = current.episode.id
        while current.episode.phase == "preparing":
            if time.monotonic() > deadline:
                raise _Ended(f"The simulation did not lay out the episode within {READY_TIMEOUT_S:.0f} s.")
            await self._sleep()
            current.episode = await self._call(lambda: self.client.aepisode(episode_id))
        return current.episode

    async def _stale_cameras(self) -> list[str]:
        """Wait until every camera sent a new frame lately; the stale ones if they do not within the bound."""
        deadline = time.monotonic() + FRESH_TIMEOUT_S
        freshness = self._control().freshness
        while (stale := freshness.stale(freshness.cameras, camera=True)) and time.monotonic() < deadline:
            await self._sleep()
        return stale

    async def _record(self, current: _Open) -> None:
        """Record one ready episode: the dataset episode first, then begin, watch, and keep or discard it."""
        episode_id = current.episode.id
        dataset_episode = uuid4().hex
        task = current.episode.task or self.state.scene or "autopilot"
        # Before the start goes out: a stop while it does discards it (``_discard_dataset`` tells a start
        # that never opened it).
        current.dataset_episode = dataset_episode
        await self._dataset(
            command_key("start_episode"),
            StartEpisodeCommand(
                request_id=uuid4().hex, task=task, dataset_id=self.dataset_id, episode_id=dataset_episode
            ),
        )
        current.episode = await self._call(lambda: self.client.abegin(episode_id))
        control = self._control()
        control.watch(episode_id)
        self._update(phase="recording")
        unanswered_since: float | None = None
        while current.episode.phase != "ended":
            await self._sleep()
            if await self._discarded(current, control, dataset_episode):
                return
            try:
                current.episode = await self.client.aepisode(episode_id)
                unanswered_since = None
            except AutopilotUnreachableError as error:
                unanswered_since = unanswered_since or time.monotonic()
                if time.monotonic() - unanswered_since > UNANSWERED_S:
                    raise _Ended(f"Lost the simulation: {error}") from error
            except AutopilotRefusedError as error:
                raise _Ended(f"The simulation refused ({error.code}): {error.message}") from error
        # The samples since the last check went into the episode too.
        if await self._discarded(current, control, dataset_episode):
            return
        control.watch(None)
        await self._conclude(current)

    async def _discarded(self, current: _Open, control: AutopilotControl, dataset_episode: str) -> bool:
        """Discard the episode if a camera went stale, the leader led another one, or the dataset dropped it.

        Returns:
            Whether it was discarded.
        """
        episode_id = current.episode.id
        if stale := control.freshness.stale(control.freshness.cameras, camera=True):
            await self._discard(current, "stale_stream", _no_samples(stale))
            return True
        if (mismatch := control.mismatch) is not None and mismatch[0] == episode_id:
            published = f"episode {mismatch[1]!r}" if mismatch[1] else "no episode"
            reason = f"The simulation's leader published {published}, not episode {episode_id!r}."
            await self._discard(current, "aborted", reason)
            return True
        status = await self._dataset_status()
        if not status.is_recording or status.episode_id != dataset_episode:  # a camera was lost, say
            current.dataset_episode = None
            await self._discard(current, "dataset", status.error or "The dataset discarded the episode.")
            return True
        return False

    async def _conclude(self, current: _Open) -> None:
        """Keep an ended episode, or discard it."""
        episode = current.episode
        status = await self._dataset_status()
        if not status.is_recording or status.episode_id != current.dataset_episode:
            current.dataset_episode = None
            await self._discard(current, "dataset", status.error or "The dataset discarded the episode.")
            return
        frames = status.frames
        if episode.outcome != "success":
            reason = episode.reason or f"The episode ended as {episode.outcome}."
            await self._discard(current, _DISCARDS.get(episode.outcome, "aborted"), reason, frames=frames)
            return
        if frames < MIN_FRAMES:
            reason = f"Only {frames} frame(s) were recorded; at least {MIN_FRAMES} are needed."
            await self._discard(current, "too_short", reason, frames=frames)
            return
        error = await self._save(current, frames)
        if error is not None:
            raise _Ended(f"Saving the episode failed: {error}")

    async def _save(self, current: _Open, frames: int) -> str | None:
        """Save the dataset episode and count it. A stop meanwhile waits for the save, then goes on.

        Returns:
            Why saving failed, or ``None``.
        """
        self._update(phase="saving")
        command = SaveEpisodeCommand(
            request_id=uuid4().hex, dataset_id=self.dataset_id, episode_id=current.dataset_episode
        )
        current.dataset_episode = None  # the worker closes the episode as it starts saving
        request = asyncio.ensure_future(
            self.bus.request(command_key("save_episode"), command, RecordingStatus, timeout=SAVE_TIMEOUT_S)
        )
        cancelled = False
        while not request.done():
            try:
                await asyncio.shield(request)
            except asyncio.CancelledError:
                cancelled = True  # stopped: the save goes on, and is counted
            except Exception:  # noqa: S110  # nosec B110 - read below
                pass
        failure = request.exception()
        error = None if failure is None else _message(failure)
        if error is None:
            self._end(current, frames=frames, reason=current.episode.reason)
        else:
            self._end(current, discard="save_failed", reason=error, frames=frames)
        if cancelled:
            raise asyncio.CancelledError
        return error

    async def _discard(self, current: _Open, discard: DiscardReason, reason: str, *, frames: int | None = None) -> None:
        """Discard the episode in the dataset, abort it on the simulation if it still runs, and count it."""
        await self._discard_dataset(current)
        await self._abort(current)
        self._end(current, discard=discard, reason=reason, frames=frames)

    async def _abort(self, current: _Open) -> None:
        """Abort the episode on the simulation, best effort: a new episode replaces it anyway."""
        if current.episode.phase == "ended":
            return
        try:
            current.episode = await asyncio.wait_for(self.client.aabort(current.episode.id), ABORT_TIMEOUT_S)
        except (AutopilotUnreachableError, AutopilotRefusedError, TimeoutError) as error:
            logger.info("Autopilot recording could not abort episode {}: {}", current.episode.id, _message(error))

    def _end(
        self,
        current: _Open,
        *,
        discard: DiscardReason | None = None,
        reason: str | None = None,
        frames: int | None = None,
        outcome: str | None = None,
    ) -> None:
        """Count the episode: kept without *discard*."""
        episode = current.episode
        kept = discard is None
        summary = AutopilotEpisodeSummary(
            id=episode.id,
            task=episode.task,
            outcome=outcome or episode.outcome,
            clean=episode.clean,
            kept=kept,
            reason=reason,
            discard_reason=discard,
            frames=frames,
        )
        discards = dict(self.state.discards)
        if discard is not None:
            discards[discard] = discards.get(discard, 0) + 1
        if self._open is current:
            self._open = None
        self._update(
            saved=self.state.saved + kept, discarded=self.state.discarded + (not kept), discards=discards, last=summary
        )
        logger.info(
            "Autopilot episode {}: {} ({}, clean {}), {} frames; {}",
            episode.id,
            "saved" if kept else f"discarded ({discard})",
            summary.outcome,
            episode.clean,
            frames,
            reason or "",
        )

    async def _finish(self, phase: Phase, message: str) -> None:
        """Hold, abort the open episode, discard its dataset episode, report; each step best effort."""
        self._halt()
        if (current := self._open) is not None:
            await self._abort(current)
            if current.dataset_episode is not None:
                try:
                    await self._discard_dataset(current)
                except Exception as error:
                    logger.warning("Autopilot recording could not discard its open episode: {}", _message(error))
                self._end(current, discard="aborted", reason=message, outcome="aborted")
            self._open = None
        with contextlib.suppress(Exception):
            await self.client.aclose()
        self._update(phase=phase, message=message, task=None)
        logger.info(
            "Autopilot recording {}: {} saved, {} discarded {}. {}",
            phase,
            self.state.saved,
            self.state.discarded,
            self.state.discards,
            message,
        )

    # -- helpers ---------------------------------------------------------

    def _control(self) -> AutopilotControl:
        if self.control is None:
            raise RuntimeError("Autopilot recording was not opened")
        return self.control

    def _halt(self) -> None:
        if self.control is not None:
            self.control.halt()

    async def _sleep(self) -> None:
        """Wait one poll; end the recording, holding the follower, once the leader or the follower went stale."""
        await asyncio.sleep(POLL_S)
        control = self._control()
        stale = control.freshness.stale(_MOTION, grace_s=FIRST_SAMPLE_S)
        if (fault := control.fault or (_no_samples(stale) if stale else None)) is not None:
            control.halt()
            raise _Ended(f"{fault}: the follower holds where it is.")

    async def _call(self, call: Callable[[], Awaitable[Episode]]) -> Episode:
        try:
            return await call()
        except AutopilotUnreachableError as error:
            raise _Ended(f"Lost the simulation: {error}") from error
        except AutopilotRefusedError as error:
            raise _Ended(f"The simulation refused ({error.code}): {error.message}") from error

    async def _dataset(self, key: str, message: BaseModel) -> RecordingStatus:
        try:
            return await self.bus.request(key, message, RecordingStatus, timeout=DATASET_TIMEOUT_S)
        except BusRequestError as error:
            raise _Ended(f"The dataset refused ({error.error_code}): {error.message}") from error
        except TimeoutError as error:
            raise _Ended("The dataset did not answer.") from error

    async def _dataset_status(self) -> RecordingStatus:
        return await self._dataset(STATUS, Empty())

    async def _discard_dataset(self, current: _Open) -> None:
        dataset_episode, current.dataset_episode = current.dataset_episode, None
        if dataset_episode is None:
            return
        command = DiscardEpisodeCommand(request_id=uuid4().hex, dataset_id=self.dataset_id, episode_id=dataset_episode)
        try:
            await self._dataset(command_key("discard_episode"), command)
        except _Ended:
            status = await self._dataset_status()
            if status.is_recording and status.episode_id == dataset_episode:
                raise  # still open: the worker refused for another reason
            # gone already: the worker discarded it itself

    def _update(self, **changes: Any) -> None:
        self.state = self.state.model_copy(update=changes)
        self._emit()


def _refused(message: str) -> RuntimeSessionCommandError:
    return RuntimeSessionCommandError(message, error_code="autopilot_refused")


def _message(error: BaseException) -> str:
    message = getattr(error, "message", None)
    return message if isinstance(message, str) and message else str(error) or type(error).__name__
