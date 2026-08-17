from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Protocol, cast

from loguru import logger
from physicalai.config import Config
from physicalai.robot import RobotError
from physicalai.runtime import RobotRuntime

from robots.shared_robot_errors import translate_robot_error
from runtime.action_source import StudioActionSource
from runtime.callbacks.stream import StreamCallback
from runtime.contract import DisconnectCommand, InMemoryCommandMailbox

if TYPE_CHECKING:
    from physicalai.capture import Camera
    from physicalai.robot.interface import Robot
    from physicalai.runtime import StopSignal

    from runtime.contract import Command, EventSink


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
    ) -> None:
        self._document = document
        self._event_sink = event_sink
        self._follower_name = follower_name
        self._leader_name = leader_name
        self._mailbox = InMemoryCommandMailbox()
        self.ready = threading.Event()
        self._follower: Robot | None = None
        self._leader: Robot | None = None
        self._cameras: dict[str, Camera] = {}
        self._action_source: StudioActionSource | None = None
        self._stream_callback: StreamCallback | None = None
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
        )
        self._stream_callback = StreamCallback(
            event_sink=self._event_sink,
            follower_source=lambda: self._action_source.follower_source,
            ready=self.ready,
            start_allowed=lambda: not self._disconnect_requested,
            lifecycle_lock=self._lifecycle_lock,
        )

    def build_runtime(self) -> RobotRuntime:
        if self._follower is None or self._action_source is None or self._stream_callback is None:
            raise RuntimeError("Runtime session has not been set up")
        self._runtime = RobotRuntime(
            robot=self._follower,
            action_source=self._action_source,
            cameras=self._cameras,
            fps=float(self._document["init_args"]["fps"]),
            callbacks=[self._stream_callback],
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
        self._mailbox.apply(command)

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
        # Device lifetime belongs to the session, not to a disposable runtime view.
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
