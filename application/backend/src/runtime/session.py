from __future__ import annotations

import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Protocol, cast

from loguru import logger
from physicalai.config import Config
from physicalai.robot import RobotError
from physicalai.runtime import RobotRuntime

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from robots.shared_robot_errors import translate_robot_error
from runtime.action_source import StudioActionSource
from runtime.callbacks.recording import RecordingCallback, RecordingState
from runtime.callbacks.stream import StreamCallback
from runtime.command_thread import CommandWorker
from runtime.contract import (
    DiscardEpisodeCommand,
    DisconnectCommand,
    ErrorEvent,
    InMemoryCommandMailbox,
    LoadDatasetCommand,
    SaveEpisodeCommand,
    StateEvent,
)
from runtime.dataset_features import build_lerobot_dataset_features
from settings import get_settings

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.capture import Camera
    from physicalai.robot.interface import Robot
    from physicalai.runtime import StopSignal

    from runtime.contract import Command, EventSink

# Bounds a ``copytree`` of a dataset that may be gigabytes. The alternative
# to waiting is deleting the cache without copying it back.
RECORDING_TEARDOWN_TIMEOUT_S = 60.0
_DEVICE_READY_TIMEOUT_S = 30.0


class _Connectable(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...


def _is_connected(device: object) -> bool:
    """Return connection state for a robot (method) or camera (property)."""
    connected = getattr(device, "is_connected", False)
    return bool(connected() if callable(connected) else connected)


class RuntimeSession:
    """Own devices for one Studio control session and run them through RobotRuntime."""

    def __init__(
        self,
        document: dict[str, Any],
        *,
        event_sink: EventSink,
        follower_name: str | None = None,
        leader_name: str | None = None,
        datasets_dir: Path | None = None,
    ) -> None:
        self._document = document
        self._event_sink = event_sink
        self._follower_name = follower_name
        self._leader_name = leader_name
        self._datasets_dir = datasets_dir
        self._mailbox = InMemoryCommandMailbox()
        self.ready = threading.Event()
        self._follower: Robot | None = None
        self._leader: Robot | None = None
        self._cameras: dict[str, Camera] = {}
        self._action_source: StudioActionSource | None = None
        self._stream_callback: StreamCallback | None = None
        self._recording = RecordingState()
        self._recording_callback: RecordingCallback | None = None
        self._command_worker = CommandWorker()
        self._runtime: RobotRuntime | None = None
        self._disconnect_requested = False
        self._lifecycle_lock = threading.Lock()

    async def setup(self) -> None:
        init_args = self._document["init_args"]
        self._follower = cast("Robot", Config.from_dict(init_args["robot"]).instantiate())
        action_source = init_args.get("action_source")
        if isinstance(action_source, dict):
            source_args = action_source.get("init_args", {})
            leader_config = source_args.get("leader") if isinstance(source_args, dict) else None
            if isinstance(leader_config, dict):
                self._leader = cast("Robot", Config.from_dict(leader_config).instantiate())

        self._cameras = {
            key: cast("Camera", Config.from_dict(config).instantiate())
            for key, config in init_args.get("cameras", {}).items()
        }

        self._action_source = StudioActionSource(
            follower=self._follower,
            leader=self._leader,
            mailbox=self._mailbox,
            event_sink=self._event_sink,
            fps=float(init_args["fps"]),
            camera_keys=tuple(self._cameras),
        )
        self._action_source.bind_recording(self._recording)
        action_source = self._action_source
        self._stream_callback = StreamCallback(
            event_sink=self._event_sink,
            follower_source=lambda: action_source.follower_source,
            state_data=action_source.state_data,
            ready=self.ready,
            start_allowed=lambda: not self._disconnect_requested,
            lifecycle_lock=self._lifecycle_lock,
        )
        self._recording_callback = RecordingCallback(
            recording=self._recording,
            follower_source=lambda: action_source.follower_source,
        )

    def build_runtime(self) -> RobotRuntime:
        if (
            self._follower is None
            or self._action_source is None
            or self._stream_callback is None
            or self._recording_callback is None
        ):
            raise RuntimeError("Runtime session has not been set up")
        self._runtime = RobotRuntime(
            robot=self._follower,
            action_source=self._action_source,
            cameras=self._cameras,
            fps=float(self._document["init_args"]["fps"]),
            callbacks=[self._stream_callback, self._recording_callback],
        )
        if self._disconnect_requested:
            self._runtime.stop()
        return self._runtime

    def apply(self, command: Command) -> None:
        if isinstance(command, DisconnectCommand):
            with self._lifecycle_lock:
                self._disconnect_requested = True
                if self._runtime is not None:
                    self._runtime.stop()
            return
        if isinstance(command, LoadDatasetCommand):
            self._command_worker.submit("load_dataset", lambda: self._load_dataset(command))
            return
        self._mailbox.apply(command)

    def handle_request(self, command: Command) -> None:
        """Run an acked command on the command worker and raise if it failed."""
        if isinstance(command, SaveEpisodeCommand):
            self._command_worker.submit(
                "save_episode",
                self._save_episode,
                request_id=command.request_id,
            )
        elif isinstance(command, DiscardEpisodeCommand):
            self._command_worker.submit(
                "discard_episode",
                self._discard_episode,
                request_id=command.request_id,
            )
        else:
            raise ValueError(f"{command.command} is not supported by this runtime session")
        ack = self._command_worker.wait(command.request_id, timeout=RECORDING_TEARDOWN_TIMEOUT_S)
        if not ack.ok:
            raise RuntimeError(ack.error or f"{command.command} failed")

    def run(self, stop_signal: StopSignal) -> None:
        if self._disconnect_requested or stop_signal.is_set():
            return
        runtime = self.build_runtime()
        try:
            if self._disconnect_requested or stop_signal.is_set():
                return
            self._preconnect_devices()
            if self._disconnect_requested or stop_signal.is_set():
                return
            # Do not use ``with runtime``: the session alone owns device teardown.
            runtime.connect()
            if self._disconnect_requested or stop_signal.is_set():
                return
            runtime.run(stop_event=stop_signal)
        except RobotError as exc:
            follower_connected = self._follower is not None and self._follower.is_connected()
            leader_setup_failed = follower_connected and not self.ready.is_set()
            robot_name = self._leader_name if leader_setup_failed else self._follower_name
            raise translate_robot_error(exc, robot_name=robot_name) from exc

    async def teardown(self) -> None:
        # Finalize first: a dataset that will not copy back must not prevent
        # the arm from being released, but skipping it loses the recording.
        self._command_worker.shutdown(timeout=RECORDING_TEARDOWN_TIMEOUT_S)
        self._finalize_recording()
        if self._recording_callback is not None:
            self._recording_callback.close()
        # Device lifetime belongs to the session, not to a disposable runtime view.
        # Stop the policy worker before dropping devices it may still be reading.
        if self._action_source is not None:
            self._action_source.shutdown_policy()
        # Cameras first so a wedged publisher cannot strand the arm connected.
        for key, camera in self._cameras.items():
            try:
                camera.disconnect()
            except Exception as exc:
                logger.warning("Camera {} disconnect failed: {}", key, exc)
        if self._leader is not None:
            try:
                self._leader.disconnect()
            except Exception as exc:
                logger.warning("Leader disconnect failed: {}", exc)
        if self._follower is not None:
            try:
                self._follower.disconnect()
            except Exception as exc:
                logger.warning("Follower disconnect failed: {}", exc)

    def finalize_recording(self) -> None:
        """Queue the cache copy so an abandoned session does not hide its episodes.

        Queued rather than run here: ``_save_episode`` writes parquet and encodes
        video outside the recording lock, so finalizing straight off the watcher
        thread could stop the image writer mid-save. The command worker already
        exists to serialize exactly that, and ``teardown`` drains it, so process
        exit cannot kill a ``copytree`` that has already deleted its destination.
        """
        if not self._recording.dataset_loaded:
            return
        try:
            self._command_worker.submit("finalize_recording", self._finalize_recording)
        except RuntimeError:
            logger.debug("Command worker is closed; teardown finalizes the recording")

    def _finalize_recording(self) -> None:
        """Copy the recording cache back to the dataset and detach the mutation.

        Idempotent: ``take_mutation`` returns ``None`` once the mutation is gone,
        so a teardown following an abandonment is a no-op. It deliberately does
        not close ``RecordingState`` -- a client reattaching within the idle
        window records into a fresh mutation over the updated dataset.
        """
        mutation = self._recording.take_mutation()
        if mutation is None:
            return
        try:
            mutation.teardown()
        except Exception:
            logger.exception("Recording mutation teardown failed; the cache may not have been copied back")
        # The cached state is what a returning client recovers, so it must not
        # keep advertising a dataset this session no longer holds.
        self._emit_state()

    def _load_dataset(self, command: LoadDatasetCommand) -> None:
        try:
            self._wait_until_devices_ready()
            previous = self._recording.take_mutation()
            if previous is not None:
                previous.teardown()
            dataset_path = self._dataset_path(command)
            dataset = InternalLeRobotDataset(dataset_path, access_mode=DatasetAccessMode.RECORDING_MUTATION)
            if self._follower is None:
                raise RuntimeError("Follower robot is not set up")
            mutation = dataset.start_recording_mutation(
                fps=int(self._document["init_args"]["fps"]),
                features=build_lerobot_dataset_features(
                    joint_names=list(self._follower.joint_names),
                    camera_specs=self._camera_specs_from_frames(),
                ),
                robot_type=self._follower_name or "unknown",
            )
            self._recording.attach_mutation(mutation)
            self._emit_state()
        except Exception as exc:
            logger.exception("Failed to load dataset {}", command.dataset_id)
            self._event_sink.emit(
                ErrorEvent(
                    message=str(exc) or "Failed to load the dataset.",
                    error_code="dataset_load_failed",
                )
            )
            raise

    def _save_episode(self) -> None:
        mutation = self._recording.stop_episode()
        mutation.save_episode()
        self._recording.mark_saved()
        self._emit_state()

    def _discard_episode(self) -> None:
        mutation = self._recording.stop_episode()
        mutation.discard_buffer()
        self._recording.mark_discarded()
        self._emit_state()

    def _emit_state(self) -> None:
        if self._action_source is None:
            return
        self._event_sink.emit(StateEvent(data=self._action_source.state_data()))

    def _dataset_path(self, command: LoadDatasetCommand) -> Path:
        root = self._datasets_dir if self._datasets_dir is not None else get_settings().datasets_dir
        return root / str(command.dataset_id)

    def _camera_specs_from_frames(self) -> dict[str, tuple[int, int, int]]:
        specs: dict[str, tuple[int, int, int]] = {}
        for key, camera in self._cameras.items():
            frame = camera.read_latest()
            height, width, channels = frame.data.shape
            specs[key] = (int(height), int(width), int(channels))
        return specs

    def _wait_until_devices_ready(self) -> None:
        deadline = time.monotonic() + _DEVICE_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if (
                self._follower is not None
                and _is_connected(self._follower)
                and all(_is_connected(camera) for camera in self._cameras.values())
            ):
                return
            time.sleep(0.05)
        raise TimeoutError("Devices were not connected in time to start recording")

    def _preconnect_devices(self) -> None:
        """Connect robots and cameras in parallel to reduce session startup time."""
        if self._follower is None:
            raise RuntimeError("Follower robot is not set up")
        devices: list[_Connectable] = [self._follower]
        if self._leader is not None:
            devices.append(self._leader)
        devices.extend(self._cameras.values())
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = {executor.submit(device.connect): device for device in devices}
            try:
                for future in as_completed(futures):
                    future.result()
            except Exception as exc:
                logger.error("Device parallel connect failed: {}", exc)
                for future in futures:
                    future.cancel()
                for device in devices:
                    if _is_connected(device):
                        logger.error("Disconnecting device {} after connect failure", device)
                        with contextlib.suppress(Exception):
                            device.disconnect()
                raise
