from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from physicalai.config import Config
from physicalai.robot import RobotError
from physicalai.runtime import RobotRuntime

from robots.shared_robot_errors import translate_robot_error
from runtime.action_source import StudioActionSource
from runtime.callbacks.stream import StreamCallback
from runtime.contract import DisconnectCommand, InMemoryCommandMailbox

if TYPE_CHECKING:
    from physicalai.robot.interface import Robot
    from physicalai.runtime import StopSignal

    from runtime.contract import Command, EventSink


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
        self._action_source: StudioActionSource | None = None
        self._stream_callback: StreamCallback | None = None
        self._runtime: RobotRuntime | None = None

    async def setup(self) -> None:
        init_args = self._document["init_args"]
        self._follower = cast("Robot", Config.from_dict(init_args["robot"]).instantiate())
        action_source = init_args.get("action_source")
        if isinstance(action_source, dict):
            source_args = action_source.get("init_args", {})
            leader_config = source_args.get("leader") if isinstance(source_args, dict) else None
            if isinstance(leader_config, dict):
                self._leader = cast("Robot", Config.from_dict(leader_config).instantiate())

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
        )

    def build_runtime(self) -> RobotRuntime:
        if self._follower is None or self._action_source is None or self._stream_callback is None:
            raise RuntimeError("Runtime session has not been set up")
        self._runtime = RobotRuntime(
            robot=self._follower,
            action_source=self._action_source,
            cameras={},
            fps=float(self._document["init_args"]["fps"]),
            callbacks=[self._stream_callback],
        )
        return self._runtime

    def apply(self, command: Command) -> None:
        if isinstance(command, DisconnectCommand):
            if self._runtime is not None:
                self._runtime.stop()
            return
        self._mailbox.apply(command)

    def run(self, stop_signal: StopSignal) -> None:
        runtime = self.build_runtime()
        try:
            self._preconnect_robots()
            # Do not use ``with runtime``: the session alone owns device teardown.
            runtime.connect()
            runtime.run(stop_event=stop_signal)
        except RobotError as exc:
            follower_connected = self._follower is not None and self._follower.is_connected()
            leader_setup_failed = follower_connected and not self.ready.is_set()
            robot_name = self._leader_name if leader_setup_failed else self._follower_name
            raise translate_robot_error(exc, robot_name=robot_name) from exc

    async def teardown(self) -> None:
        # Device lifetime belongs to the session, not to a disposable runtime view.
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

    def _preconnect_robots(self) -> None:
        """Connect robots in parallel to reduce session startup time."""
        if self._follower is None:
            raise RuntimeError("Follower robot is not set up")
        robots = [self._follower]
        if self._leader is not None:
            robots.append(self._leader)
        with ThreadPoolExecutor(max_workers=len(robots)) as executor:
            futures = {executor.submit(robot.connect): robot for robot in robots}
            try:
                for future in as_completed(futures):
                    future.result()
            except Exception as exc:
                logger.error("Robot parallel connect failed: {}", exc)
                for future in futures:
                    future.cancel()
                for robot in robots:
                    if robot.is_connected():
                        logger.error("Disconnecting robot {} after connect failure", robot)
                        with contextlib.suppress(Exception):
                            robot.disconnect()
                raise
