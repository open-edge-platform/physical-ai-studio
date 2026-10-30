"""A runtime session: what one websocket has loaded, and the workers running it.

The session lives in the API process and belongs to the websocket that opened
it: the socket holds it with ``async with``, and leaving that block releases
everything it loaded, in reverse. Like the workers, it is a part on the bus:
while it runs it serves its own commands (loading, unloading and switching
control, see ``command_key``) and publishes its state and errors on
``evt/session``. The socket only forwards between the two.

Its optional parts are loaded and unloaded independently, each in its own
:class:`Slot`: the environment (see ``runtime_session/environment.py``), the
control driving the follower, and the dataset being recorded (see
``runtime_session/dataset.py``), plus the store they share. The session adds
what is about the session itself: its claims on the devices, and deciding what
a part's failure means for the others. The parts report their status on
``evt/status/<part>`` themselves, and the session listens (see ``_listen``);
it reports a control's for it, as a control is a thread in this process; a
control serves its own commands on the session's bus while it runs. The socket
forwards all of them.

See the "RuntimeSession API" section of docs/feature_plans/runtime-inhousing-plan.md.
"""

from __future__ import annotations

import asyncio
import datetime
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast
from uuid import uuid4

import anyio
from loguru import logger

from exceptions import BaseException as AppBaseException
from exceptions import IncompatiblePartsError, RuntimeSessionCommandError
from internal_datasets.utils import get_internal_read_dataset
from runtime_session.bus import EVENTS, Bus, BusRequestError, command_key, publish_status, status_key
from runtime_session.controls import build_control
from runtime_session.controls.base import CHANGED as CONTROL_CHANGED
from runtime_session.dataset import COMPONENT as DATASET
from runtime_session.dataset import Recording, dataset_spec
from runtime_session.environment import COMPONENT as ENVIRONMENT
from runtime_session.environment import RELOAD, Environment, describe_environment
from runtime_session.model import COMPONENT as MODEL
from runtime_session.model import ProcessModel, model_inputs
from runtime_session.slot import Slot
from runtime_session.store import ACTION, STATE, FeatureSpecMismatchError, Store
from schemas.runtime_session import (
    CommandDone,
    DiscardEpisodeCommand,
    ErrorMessage,
    LoadDatasetCommand,
    LoadEnvironmentCommand,
    LoadModelCommand,
    ModelControlConfig,
    ReloadEnvironment,
    ReloadEnvironmentCommand,
    ReplayControlConfig,
    RuntimeSessionInfo,
    SessionEnvironment,
    SessionState,
    SetControlCommand,
    StateMessage,
    StatusMessage,
    UnloadDatasetCommand,
    UnloadEnvironmentCommand,
    UnloadModelCommand,
)
from services.camera_claims import CameraClaim, settings_from_camera
from workers import dataset_worker as dataset_rpc
from workers.base import WorkerStartupError
from workers.dataset_worker import RecordingStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
    from uuid import UUID

    from physicalai.robot import Robot
    from pydantic import BaseModel

    from robots.robot_client_factory import RobotClientFactory
    from runtime_session.controls import Control
    from runtime_session.environment import EnvironmentDescription
    from runtime_session.registry import RuntimeSessionRegistry
    from runtime_session.store import FeatureSpec, Mismatch
    from schemas.dataset import Dataset, Episode
    from schemas.environment import EnvironmentWithRelations
    from schemas.model import Model
    from schemas.runtime_session import ControlConfig, EnvironmentDraft
    from services.camera_claims import CameraClaimRegistry
    from workers.base import ThreadWorker

_RELOAD_S = 120.0
"""A backstop: each device has its own ready timeout."""

_RELEASE_S = 10.0
"""How long a load waits for a closing session to disconnect the devices it held."""

_CONTROL = "control"
"""What the session reports a control as: a control is a thread in the session, not on the bus itself."""


@dataclass(frozen=True, slots=True)
class Resolvers:
    """Look up what a command names, in the session's project; supplied by whoever opens the session."""

    environment: Callable[[EnvironmentDraft], Awaitable[EnvironmentWithRelations]]
    dataset: Callable[[UUID], Awaitable[Dataset]]
    model: Callable[[UUID], Awaitable[Model]]


@dataclass(frozen=True, slots=True)
class LoadedEnvironment:
    environment: Environment
    state: SessionEnvironment
    robot_type: str
    """The follower's type, which a new dataset records."""


class RuntimeSession:
    def __init__(
        self,
        *,
        project_id: UUID,
        project_name: str,
        registry: RuntimeSessionRegistry,
        camera_claims: CameraClaimRegistry,
        robot_factory: RobotClientFactory,
        resolve: Resolvers,
    ) -> None:
        self.id = uuid4().hex
        self.project_id = project_id
        self.project_name = project_name
        self.started_at = datetime.datetime.now(tz=datetime.UTC)
        self.closed = asyncio.Event()
        self.finished = asyncio.Event()
        """Set once a closed session has let go of everything it held."""
        self._resolve = resolve
        self._registry = registry
        self._camera_claims = camera_claims
        self._robot_factory = robot_factory
        self._lock = asyncio.Lock()
        self._environment: Slot[LoadedEnvironment] = Slot("environment")
        self._control: Slot[Control] = Slot("control")
        """Uses the environment's store and leader, so it is unloaded before the environment."""
        self._dataset: Slot[Recording] = Slot("dataset")
        """Records the store, so it is unloaded before the store is replaced."""
        self._model: Slot[ProcessModel] = Slot("model")
        """Reads the store's frames, so it is unloaded before the store is replaced; a model control borrows it."""
        self._store: Store | None = None
        """Joints and frames, for the environment's devices and the dataset. Outlives a reloaded
        environment while the spec stays the same, so the dataset recording from it stays loaded."""
        self._bus: Bus | None = None
        """Hosted for the session's lifetime; its parts join it."""

    async def __aenter__(self) -> Self:
        self._bus = Bus.host(self.id)
        self._serve_commands(self._bus)
        self._listen(self._bus)
        self._registry.add(self)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self._teardown("socket closed", notify=False)

    async def stop(self) -> None:
        """Close the session from outside its socket; the socket gets ``session_closed`` and closes."""
        await self._teardown("stopped", notify=True)

    async def _teardown(self, reason: str, *, notify: bool) -> None:
        """Unload everything and unregister. Runs once, whichever of the socket and a stop gets here first."""
        if self.closed.is_set():
            return
        self.closed.set()
        with anyio.CancelScope(shield=True):
            try:
                async with self._lock:
                    await self._control.unload()
                    await self._model.unload()
                    await self._dataset.unload()
                    await self._environment.unload()
                    self._drop_store()
            finally:
                self._registry.remove(self)
                if notify:
                    self._emit(
                        ErrorMessage(
                            component="session",
                            message="The runtime session was closed.",
                            error_code="session_closed",
                            session_alive=False,
                        )
                    )
                if self._bus is not None:
                    self._bus.close()
                logger.info("Session {} closed ({})", self.id, reason)
                self.finished.set()

    def _serve_commands(self, bus: Bus) -> None:
        """Serve the commands that change what the session holds; workers serve their own."""
        commands: dict[str, tuple[type[BaseModel], Callable[[Any], Awaitable[None]]]] = {
            "load_environment": (
                LoadEnvironmentCommand,
                lambda command: self._load_draft(command.environment, unload_incompatible=command.unload_incompatible),
            ),
            "unload_environment": (UnloadEnvironmentCommand, lambda _command: self.unload_environment()),
            "reload_environment": (ReloadEnvironmentCommand, lambda _command: self.reload_environment()),
            "set_control": (SetControlCommand, lambda command: self.set_control(command.control)),
            "load_dataset": (
                LoadDatasetCommand,
                lambda command: self._load_dataset_by_id(command.dataset_id, command.fps),
            ),
            "unload_dataset": (UnloadDatasetCommand, lambda _command: self.unload_dataset()),
            "load_model": (LoadModelCommand, self._load_model_command),
            "unload_model": (UnloadModelCommand, lambda _command: self.unload_model()),
        }
        for name, (model, run) in commands.items():
            bus.serve(command_key(name), model, _as_command(name, run))

    def _listen(self, bus: Bus) -> None:
        """Deal with what the parts report about themselves on the bus."""
        bus.listen(status_key(ENVIRONMENT), StatusMessage, self._on_environment)
        bus.listen(status_key(_CONTROL), StatusMessage, self._on_control)
        # A control changed what it does through one of its commands; it already holds the new config.
        bus.listen(CONTROL_CHANGED, dict, lambda _config: self._emit(StateMessage(data=self.state())))
        bus.listen(status_key(DATASET), StatusMessage, self._on_dataset)
        bus.listen(status_key(MODEL), StatusMessage, self._on_model)
        bus.listen(dataset_rpc.EVENTS, RecordingStatus, self._on_recording)

    async def _load_draft(self, draft: EnvironmentDraft, *, unload_incompatible: bool) -> None:
        await self.load_environment(await self._resolve.environment(draft), unload_incompatible=unload_incompatible)

    async def _load_dataset_by_id(self, dataset_id: UUID, fps: int | None) -> None:
        await self.load_dataset(await self._resolve.dataset(dataset_id), fps)

    async def _load_model_command(self, command: LoadModelCommand) -> None:
        await self.load_model(await self._resolve.model(command.model_id), command.backend, command.device)

    @property
    def label(self) -> str:
        """How errors name this session to a user."""
        if (environment := self.environment) is not None:
            return f"the runtime session for {environment.name!r} (project {self.project_name!r})"
        return f"a runtime session in project {self.project_name!r}"

    @property
    def bus(self) -> Bus:
        """The session's bus: commands go onto it, and its events come off it."""
        if self._bus is None:
            raise RuntimeSessionCommandError("The session is not open.", error_code="session_closed")
        return self._bus

    @property
    def environment(self) -> SessionEnvironment | None:
        loaded = self._environment.value
        return loaded.state if loaded is not None else None

    @property
    def store(self) -> Store | None:
        return self._store

    @property
    def leader(self) -> Robot | None:
        """The environment's leader, connected, for controls to borrow."""
        loaded = self._environment.value
        return loaded.environment.leader if loaded is not None else None

    def state(self) -> SessionState:
        control, dataset, model = self._control.value, self._dataset.value, self._model.value
        return SessionState(
            session_id=self.id,
            environment=self.environment,
            control=control.config if control else None,
            dataset=dataset.state() if dataset else None,
            model=model.state() if model else None,
        )

    def info(self) -> RuntimeSessionInfo:
        environment = self.environment
        return RuntimeSessionInfo(
            session_id=self.id,
            project_id=self.project_id,
            project_name=self.project_name,
            started_at=self.started_at,
            environment_name=environment.name if environment else None,
            robot_ids=[robot for robot in (environment.robot_id, environment.leader_id) if robot]
            if environment
            else [],
            camera_ids=[camera.camera_id for camera in environment.cameras] if environment else [],
        )

    async def load_environment(
        self, environment: EnvironmentWithRelations, *, unload_incompatible: bool = False
    ) -> None:
        """Connect the environment's robot, leader and cameras, replacing a loaded one.

        An environment with the same spec reuses the store, and the dataset stays loaded. With
        another spec the store is replaced, and a loaded dataset is loaded again on it when it
        still fits. If a device fails to connect, the environment is loaded anyway, failed: its
        status says why, and it can be reloaded.

        Raises:
            RuntimeSessionCommandError: The session is closed.
            IncompatiblePartsError: A loaded dataset does not fit the new environment and
                ``unload_incompatible`` is not set; nothing changed.
            DeviceInUseError: Another session holds one of its robots.
            CameraSettingsConflictError: Another session uses a camera at other settings.
        """
        async with self._lock:
            if self.closed.is_set():
                raise RuntimeSessionCommandError("The session is closed.", error_code="session_closed")
            # Described before anything is unloaded: building drivers touches no device, and a
            # refusal must leave the session as it was.
            description = await describe_environment(environment, self._robot_factory)
            reuse_store = self._store is not None and self._store.spec == description.spec
            dataset = self._dataset.value
            # A new store needs the dataset loaded again on it.
            keep = dataset.dataset if dataset is not None and not reuse_store else None
            keep_fps = dataset.fps if dataset is not None else None
            model = self._model.value
            # Like the dataset, a model reads the store, so it is loaded again on a new one.
            keep_model = model if model is not None and not reuse_store else None
            incompatible: list[dict[str, object]] = []
            if keep is not None and (mismatches := self._dataset_mismatches(keep, dataset, description.spec)):
                incompatible.append({"part": "dataset", "name": keep.name, "reasons": [str(m) for m in mismatches]})
                keep = None
            if keep_model is not None and (reason := _model_mismatch(keep_model, description.spec)):
                incompatible.append({"part": "model", "name": keep_model.model.name, "reasons": [reason]})
                keep_model = None
            if incompatible and not unload_incompatible:
                raise IncompatiblePartsError(incompatible)
            await self._control.unload()
            await self._environment.unload()
            if not reuse_store:
                await self._model.unload()
                await self._dataset.unload()
                self._drop_store()  # first: the new one takes the same name
                self._store = Store.create(description.spec, self.id)
            try:
                await self._environment.load(self._loaded_environment(environment, description))
                logger.info("Session {} loaded environment {!r}", self.id, environment.name)
            finally:
                # Also when the load was refused: the dataset records from the store, which is there either way.
                if keep is not None:
                    await self._reload_dataset(keep, environment.robots[0].robot.type, keep_fps)
                if keep_model is not None:
                    await self._reload_model(keep_model)
                self._emit(StateMessage(data=self.state()))

    def _drop_store(self) -> None:
        """Hold the lock, with nothing loaded that uses the store."""
        if self._store is not None:
            self._store.unlink()
            self._store = None

    def _dataset_mismatches(self, dataset: Dataset, loaded: Recording | None, spec: FeatureSpec) -> list[Mismatch]:
        """What ``spec`` lacks that the dataset needs, counting what this session recorded but has not copied back."""
        recorded = dataset_spec(dataset)
        if recorded is None and loaded is not None and loaded.status.episodes_recorded and self.store is not None:
            # A new dataset gets its shape from the episodes this session recorded into it.
            recorded = self.store.spec
        return spec.check_compatible(recorded) if recorded is not None else []

    async def _reload_dataset(self, dataset: Dataset, robot_type: str, fps: int | None) -> None:
        """Hold the lock. Load ``dataset`` again on a new store at its rate; a failure is reported, not raised."""
        try:
            await self._dataset.load(self._recording(dataset, robot_type, fps))
        except Exception as error:
            logger.opt(exception=error).error("Session {} could not load dataset {!r} again", self.id, dataset.name)
            self._emit(
                ErrorMessage(
                    component="dataset",
                    message=f"Dataset {dataset.name!r} could not be loaded again: {error}",
                    error_code="dataset_reload_failed",
                    session_alive=True,
                )
            )

    async def _reload_model(self, model: ProcessModel) -> None:
        """Hold the lock. Load ``model`` again on a new store; a failure is reported, not raised."""
        try:
            await self._model.load(self._process_model(model.model, model.backend, model.device))
        except Exception as error:
            logger.opt(exception=error).error("Session {} could not load model {!r} again", self.id, model.model.name)
            self._emit(
                ErrorMessage(
                    component="model",
                    message=f"Model {model.model.name!r} could not be loaded again: {error}",
                    error_code="model_reload_failed",
                    session_alive=True,
                )
            )

    @asynccontextmanager
    async def _loaded_environment(
        self, environment: EnvironmentWithRelations, description: EnvironmentDescription
    ) -> AsyncIterator[LoadedEnvironment]:
        """Claim and connect the environment for as long as it is loaded."""
        store = cast("Store", self._store)  # created before the environment is loaded
        await self._registry.until_closed(within=_RELEASE_S)
        with self._claim(description):
            async with Environment(description, self.id, store) as held:
                yield LoadedEnvironment(
                    environment=held,
                    state=description.session_environment(),
                    robot_type=environment.robots[0].robot.type,
                )

    @contextmanager
    def _claim(self, description: EnvironmentDescription) -> Iterator[None]:
        """Claim the robots and pin the camera settings for this session, so no form or other session takes them.

        Raises:
            DeviceInUseError: Another session holds one of the robots.
            CameraSettingsConflictError: Another session uses a camera at other settings.
        """
        claims = []
        for camera in description.cameras.values():
            if camera.fingerprint is None:
                raise ValueError(f"Camera {camera.name!r} must be reselected")
            claims.append(
                CameraClaim(
                    fingerprint=camera.fingerprint,
                    settings=settings_from_camera(camera),
                    holder=self.id,
                    project_id=self.project_id,
                    project_name=self.project_name,
                )
            )
        self._registry.claim_robots(self, dict(description.robot_names))
        try:
            generation = self._camera_claims.claim(claims)
            try:
                yield
            finally:
                self._camera_claims.release(self.id, generation=generation)
        finally:
            self._registry.release_robots(self)

    async def set_control(self, config: ControlConfig | None) -> None:
        """Stop the running control, hold the follower where it is, then start ``config``.

        Raises:
            RuntimeSessionCommandError: No environment is loaded.
            RuntimeSessionCommandError: A replay without a loaded dataset.
            ValueError: The environment cannot run ``config`` (no leader, unknown joints, no such episode).
            WorkerStartupError: The control failed to start; the follower holds.
        """
        async with self._lock:
            loaded = self._environment.require("Load an environment first.")
            try:
                await self._switch_control(loaded.environment, config)
            finally:
                self._emit(StateMessage(data=self.state()))

    async def _switch_control(self, environment: Environment, config: ControlConfig | None) -> None:
        """Hold the lock. One control at a time, by construction: the old one is joined before the next starts."""
        await self._control.unload()
        # Hold: the follower stays where it is now, not where the old control last pointed it.
        if (state := environment.store.read(STATE)) is not None:
            environment.store.write(ACTION, state.value, timestamp=time.monotonic())
        if config is not None:
            await self._control.load(self._watched_control(config, environment))
            logger.info("Session {} switched to {} control", self.id, config.type)

    @asynccontextmanager
    async def _watched_control(self, config: ControlConfig, environment: Environment) -> AsyncIterator[Control]:
        episode = await self._episode(config.episode_index) if isinstance(config, ReplayControlConfig) else None
        control = build_control(config, environment.store, environment.leader, episode, self._model.value, self.bus)
        async with self._reported(_CONTROL, control):
            yield control

    async def _episode(self, index: int) -> Episode | None:
        """An episode of the loaded dataset as it is on disk, without what this session has recorded into it."""
        dataset = self._dataset.require("Load a dataset to replay first.").dataset
        return await asyncio.to_thread(lambda: get_internal_read_dataset(dataset).find_episode(index))

    async def reload_environment(self) -> None:
        """Ask the environment to connect its devices again; the store and the dataset stay.

        Raises:
            RuntimeSessionCommandError: No environment is loaded, or a device failed to connect;
                the environment stays loaded then, failed, to try again.
        """
        async with self._lock:
            self._environment.require("Load an environment first.")
            # The control borrows the leader, and the episode records the devices being replaced.
            await self._control.unload()
            await self._stop_recording()
            try:
                await self.bus.request(RELOAD, ReloadEnvironment(), CommandDone, timeout=_RELOAD_S)
            except BusRequestError as error:
                raise RuntimeSessionCommandError(error.message, error_code=error.error_code) from error
            finally:
                self._emit(StateMessage(data=self.state()))

    async def unload_environment(self) -> None:
        """Stop the control and disconnect the environment. A no-op when none is loaded.

        The store and the dataset stay, until an environment with another spec needs a new store.
        """
        async with self._lock:
            if (environment := self.environment) is None:
                return
            await self._control.unload()
            await self._environment.unload()
            logger.info("Session {} unloaded environment {!r}", self.id, environment.name)
        self._emit(StateMessage(data=self.state()))

    async def load_dataset(self, dataset: Dataset, fps: int | None = None) -> None:
        """Start recording into ``dataset`` from the environment's store, at ``fps`` if it has nothing recorded yet.

        Raises:
            RuntimeSessionCommandError: No environment is loaded.
            FeatureSpecMismatchError: The dataset was recorded with joints or cameras this environment lacks.
            WorkerStartupError: The dataset could not be opened for recording.
        """
        async with self._lock:
            environment = self._environment.require("Load an environment first.")
            # Swapping: the loaded one is finished (copied back) first.
            await self._dataset.unload()
            await self._dataset.load(self._recording(dataset, environment.robot_type, fps))
            logger.info("Session {} records into dataset {!r}", self.id, dataset.name)
        self._emit(StateMessage(data=self.state()))

    def _recording(self, dataset: Dataset, robot_type: str, fps: int | None) -> Recording:
        # The store is there with any environment, even one that failed to connect.
        return Recording(dataset, self.id, cast("Store", self._store), robot_type, fps)

    async def load_model(self, model: Model, backend: str, device: str) -> None:
        """Load one of ``model``'s exports in its own process, replacing a loaded model.

        Raises:
            RuntimeSessionCommandError: No environment is loaded.
            ValueError: The model has no such export, or does not fit the environment.
            WorkerStartupError: The model could not be loaded.
        """
        async with self._lock:
            environment = self._environment.require("Load an environment first.")
            await self._unload_model(environment.environment)
            await self._model.load(self._process_model(model, backend, device))
            logger.info("Session {} loaded model {!r} ({}, {})", self.id, model.name, backend, device)
        self._emit(StateMessage(data=self.state()))

    def _process_model(self, model: Model, backend: str, device: str) -> ProcessModel:
        # The store is there with any environment, even one that failed to connect.
        return ProcessModel(model, self.id, cast("Store", self._store), backend=backend, device=device)

    async def unload_model(self) -> None:
        """Stop a model control, then the model. A no-op when none is loaded."""
        async with self._lock:
            if (loaded := self._environment.value) is not None:
                await self._unload_model(loaded.environment)
            else:
                await self._model.unload()
        self._emit(StateMessage(data=self.state()))

    async def _unload_model(self, environment: Environment) -> None:
        """Hold the lock. A control driving the follower with the model holds it first."""
        if (control := self._control.value) is not None and isinstance(control.config, ModelControlConfig):
            await self._switch_control(environment, None)
        await self._model.unload()

    async def unload_dataset(self) -> None:
        """Stop recording and copy what was recorded back into the dataset. A no-op when none is loaded."""
        async with self._lock:
            await self._dataset.unload()
        self._emit(StateMessage(data=self.state()))

    async def _stop_recording(self) -> None:
        """Hold the lock. Discard the episode being recorded: its last frames came from failing devices."""
        dataset = self._dataset.value
        if dataset is None or not dataset.status.is_recording:
            return
        try:
            await self.bus.request(
                command_key("discard_episode"), DiscardEpisodeCommand(request_id=""), RecordingStatus, timeout=5.0
            )
        except Exception:  # the dataset worker reports its own failure
            logger.exception("Session {} could not discard the open episode", self.id)

    @asynccontextmanager
    async def _reported(self, component: str, worker: ThreadWorker[Any]) -> AsyncIterator[None]:
        """Run a control, reporting its status for it, as the other parts report their own."""
        publish_status(self.bus, component, "starting")
        try:
            await worker.__aenter__()
        except Exception as error:
            publish_status(self.bus, component, "failed", str(error))
            raise
        publish_status(self.bus, component, "running")
        watch = asyncio.create_task(self._report_death(component, worker))
        try:
            yield
        finally:
            watch.cancel()
            await worker.__aexit__(None, None, None)
            if not worker.failure:
                publish_status(self.bus, component, "stopped")

    async def _report_death(self, component: str, worker: ThreadWorker[Any]) -> None:
        await worker.wait_stopped()
        publish_status(self.bus, component, "failed", worker.failure or "stopped unexpectedly")

    # What the session does when its parts report; each part says why on the bus itself.

    async def _on_environment(self, status: StatusMessage) -> None:
        """A failed environment stays loaded, to be reloaded; nothing drives or records it meanwhile."""
        if status.state == "failed":
            async with self._lock:
                await self._control.unload()
                await self._stop_recording()
            self._emit(StateMessage(data=self.state()))

    async def _on_control(self, status: StatusMessage) -> None:
        """A failed control is dropped, and the follower holds; the environment stays loaded."""
        if status.state == "failed":
            async with self._lock:
                if (loaded := self._environment.value) is not None and self._control.value is not None:
                    await self._switch_control(loaded.environment, None)
            self._emit(StateMessage(data=self.state()))

    async def _on_model(self, status: StatusMessage) -> None:
        """A failed model is unloaded, and the follower holds if it was driving it."""
        if status.state == "failed":
            await self.unload_model()

    async def _on_dataset(self, status: StatusMessage) -> None:
        """A failed dataset is unloaded."""
        if status.state == "failed":
            await self.unload_dataset()

    def _on_recording(self, status: RecordingStatus) -> None:
        """The dataset worker owns its recording state; show whatever it last announced."""
        if (recording := self._dataset.value) is not None:
            recording.status = status
            self._emit(StateMessage(data=self.state()))

    def _emit(self, event: StateMessage | ErrorMessage) -> None:
        if self._bus is not None:
            self._bus.publish(EVENTS, event)
        if event.type == "state":
            self._registry.changed()


def _model_mismatch(model: ProcessModel, spec: FeatureSpec) -> str | None:
    """Why the loaded model would not fit a store with ``spec``, or ``None``."""
    try:
        model_inputs(Path(model.model.path) / "exports" / model.backend, spec)
    except ValueError as error:
        return str(error)
    return None


def _as_command(name: str, run: Callable[[Any], Awaitable[None]]) -> Callable[[Any], Awaitable[CommandDone]]:
    """A bus handler for a session command, failing with an error code its requester can show."""

    async def handle(command: Any) -> CommandDone:
        try:
            await run(command)
        except Exception as error:
            refused = _command_error(error)
            if refused.error_code != "internal_error":  # an internal error is logged with its traceback
                logger.warning("Session command {} refused ({}): {}", name, refused.error_code, refused.message)
            raise refused from error
        return CommandDone()

    return handle


_ERROR_CODES: tuple[tuple[type[Exception], str], ...] = (
    (FeatureSpecMismatchError, "feature_mismatch"),  # before ValueError, which it is
    (TimeoutError, "command_timeout"),
    (ValueError, "invalid_session_command"),
)


def _command_error(error: Exception) -> AppBaseException:
    if isinstance(error, AppBaseException):
        return error
    if isinstance(error, WorkerStartupError):
        return RuntimeSessionCommandError(f"{error.worker_name}: {error.reason}", error_code="device_connection_failed")
    code = next((code for kind, code in _ERROR_CODES if isinstance(error, kind)), None)
    if code is None:
        logger.opt(exception=error).error("Runtime session command failed")
        code = "internal_error"
    return RuntimeSessionCommandError(str(error) or type(error).__name__, error_code=code)
