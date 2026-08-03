import asyncio
import enum
import threading
from multiprocessing.synchronize import Event as EventClass

from loguru import logger

from exceptions import BaseException as AppBaseException
from robots.robot_client import RobotClient
from robots.robot_client_factory import RobotClientFactory
from schemas.robot import Robot

from .base import BaseThreadWorker, run_at_frequency


class ActionReadState(enum.IntEnum):
    NONE = 0
    TELEOPERATION = 1


class TeleoperateWorker(BaseThreadWorker):
    """Robot control and teleoperate worker

    This Worker Class builds and connects a follower robot and optional leader robot from a factory.
    The worker shares a locked observation snapshot and control mode with the API thread.

    The worker can be in 2 ActionReadState modes for the follower:
    - NONE: follower robot does not receive any actions
    - TELEOPERATION: follower robot position is set from the leader's robot position

    Example:
      >>> # Start teleoperate worker
      >>> worker = TeleoperateWorker(
      ...   robot_client_factory=factory,
      ...   follower=follower_schema,
      ...   leader=leader_schema,
      ...   frequency=fps,
      ...   stop_event=scheduler.mp_stop_event
      ... )
      >>> worker.start() # Worker is now in None mode and will only update state
      >>> worker.set_action_read_state(ActionReadState.TELEOPERATION) # Worker is now in teleoperate mode
    """

    ROLE: str = "TeleoperateWorker"

    def __init__(
        self,
        robot_client_factory: RobotClientFactory,
        follower: Robot,
        leader: Robot | None,
        frequency: float,
        stop_event: EventClass,
    ) -> None:
        super().__init__(stop_event=stop_event)

        self.robot_client_factory = robot_client_factory
        self._follower_robot = follower
        self._leader_robot = leader
        self.frequency = frequency

        # State shared with the API thread.
        self.loaded_event = threading.Event()
        self._action_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._action_read_state = ActionReadState.NONE
        self._output_state: list[float] = []
        self.follower: RobotClient | None = None
        self.leader: RobotClient | None = None
        self.features: list[str] = []
        self.setup_error: Exception | None = None

    def get_state(self) -> list[float]:
        with self._state_lock:
            return list(self._output_state)

    def _set_state(self, data: list[float]) -> None:
        with self._state_lock:
            self._output_state[:] = data

    def get_action_read_state(self) -> int:
        with self._action_lock:
            return int(self._action_read_state)

    def set_action_read_state(self, value: int | ActionReadState) -> None:
        with self._action_lock:
            self._action_read_state = ActionReadState(value)

    def _align_feature_values(
        self,
        source_state: dict[str, float],
        follower_state: dict[str, float] | None = None,
    ) -> list[float]:
        # Leader observations may not expose all follower features
        # (e.g. follower has .vel keys while leader only publishes .pos).
        # When a feature is missing from the source state, fall back to
        # the follower's current state so we always have a value for
        # every feature in the shared observation snapshot.
        values: list[float] = []
        for key in self.features:
            if key in source_state:
                values.append(source_state[key])
            elif follower_state is not None and key in follower_state:
                values.append(follower_state[key])
            else:
                values.append(0.0)
        return values

    async def wait_until_loaded(self, poll_interval: float = 0.05) -> None:
        """Wait until the worker has loaded and is ready, or raise setup_error."""
        while not self.loaded_event.is_set():
            if not self.is_alive():
                if self.setup_error is not None:
                    raise self.setup_error
                raise RuntimeError("Teleoperation worker stopped before loading")
            await asyncio.sleep(poll_interval)

    def _record_setup_failure(self, exc: Exception) -> None:
        """Store a setup failure for :meth:`wait_until_loaded` to re-raise."""
        self.setup_error = exc
        if isinstance(exc, AppBaseException):
            logger.warning("Failed to set up teleoperation worker: {} ({})", exc.message, exc.error_code)
        else:
            logger.exception("Failed to set up teleoperation worker")

    def _disconnect_robots(self) -> None:
        logger.info("Teleoperating stopped, disconnecting robots.")
        if self.leader is not None:
            self.leader.disconnect()
        if self.follower is not None:
            self.follower.disconnect()

    async def run_loop(self) -> None:
        try:
            # Setup: build and connect, then publish the first observation.
            self.follower = await self.robot_client_factory.build(self._follower_robot)
            if self._leader_robot is not None:
                self.leader = await self.robot_client_factory.build(self._leader_robot)

            if self.leader is not None:
                self.leader.connect()
            self.follower.connect()

            self.features = self.follower.features()
            state = self.follower.read_state()["state"]
            self._set_state(self._align_feature_values(state))
            self.loaded_event.set()

            # Teleoperate loop until unload is requested
            goal_time = 1 / self.frequency
            while not self.should_stop():
                async with run_at_frequency(self.frequency):
                    state = (self.follower.read_state())["state"]
                    self._set_state(self._align_feature_values(state))
                    if self.get_action_read_state() == ActionReadState.TELEOPERATION and self.leader is not None:
                        actions = (self.leader.read_state())["state"]
                        filtered = self._align_feature_values(actions, follower_state=state)
                        self.follower.set_joints_state(dict(zip(self.features, filtered)), goal_time * 2)
        except Exception as exc:
            # loaded_event marks the end of setup: before it, the API caller is
            # still waiting and needs the reason; after it, this is a runtime
            # fault and BaseThreadWorker.run logs it.
            if self.loaded_event.is_set():
                raise
            self._record_setup_failure(exc)
        finally:
            self._disconnect_robots()
