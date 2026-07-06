import asyncio
import ctypes
import enum
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from typing import Any

from loguru import logger

from robots.robot_client import RobotClient

from .base import BaseProcessWorker, run_at_frequency


class ActionReadState(enum.IntEnum):
    NONE = 0
    TELEOPERATION = 1
    FROM_ACTIONS = 2


class TeleoperateWorker(BaseProcessWorker):
    """Robot control and teleoperate worker

    This Worker Class connects to a robot and if provided a leader robot.
    Data is stored in mp.SharedMemory.

    The worker can be in 3 ActionReadState modes for the follower:
    - NONE: follower robot does not receive any actions
    - TELEOPERATION: follower robot position is set from the leader's robot position
    - FROM_ACTIONS: follower robot position is set from current _output_actions mp.SharedMemory.
    This allows control from outside this worker.

    Example:
      >>> # Start teleoperate worker
      >>> worker = TeleoperateWorker(
      ...   follower=follower_client, leader=leader_client, frequency=fps, mp_stop_event=scheduler.mp_stop_event
      ... )
      >>> worker.start() # Worker is now in None mode and will only update state shared memory
      >>> worker.set_action_read_state(ActionReadState.TELEOPERATION) # Worker is now in teleoperate mode
      >>> worker.set_action_read_state(ActionReadState.FROM_ACTIONS) # Worker now follows actions shared memory.
    """

    ROLE: str = "TeleoperateWorker"

    follower: RobotClient
    leader: RobotClient | None
    stop_event: EventClass
    _action_read_state: Any
    _output_actions: Any
    _output_state: Any

    def __init__(self, follower: RobotClient, leader: RobotClient | None, frequency: float, mp_stop_event: EventClass):
        buffer_length = len(follower.features())
        self.loaded_event = mp.Event()
        self._action_read_state = mp.Value(ctypes.c_int, ActionReadState.NONE)
        self._output_actions = mp.Array(ctypes.c_double, buffer_length)
        self._output_state = mp.Array(ctypes.c_double, buffer_length)

        super().__init__(
            stop_event=mp_stop_event,
            queues_to_cancel=[],
        )
        self.follower = follower
        self.features = self.follower.features()
        self.leader = leader
        self.frequency = frequency

    def get_state(self) -> list[float]:
        with self._output_state.get_lock():
            return list(self._output_state.get_obj())

    def _set_state(self, data: list[float]) -> None:
        with self._output_state.get_lock():
            self._output_state.get_obj()[:] = data

    def get_actions(self) -> list[float]:
        with self._output_actions.get_lock():
            return list(self._output_actions.get_obj())

    def _set_actions(self, data: list[float]) -> None:
        with self._output_actions.get_lock():
            self._output_actions.get_obj()[:] = data

    def get_action_read_state(self) -> int:
        return self._action_read_state.value

    def set_action_read_state(self, value: ActionReadState) -> None:
        with self._action_read_state.get_lock():
            self._action_read_state.value = value

    def _align_feature_values(
        self,
        source_state: dict[str, float],
        follower_state: dict[str, float] | None = None,
    ) -> list[float]:
        # Leader observations may not expose all follower features
        # (e.g. follower has .vel keys while leader only publishes .pos).
        # When a feature is missing from the source state, fall back to
        # the follower's current state so we always have a value for
        # every feature in the shared memory buffer.
        values: list[float] = []
        for key in self.features:
            if key in source_state:
                values.append(source_state[key])
            elif follower_state is not None and key in follower_state:
                values.append(follower_state[key])
            else:
                values.append(0.0)
        return values

    async def wait_for_loading_to_complete(self) -> None:
        await asyncio.to_thread(self.loaded_event.wait)

    async def setup(self) -> None:
        if self.leader is not None:
            logger.info(f"Connecting leader: {self.leader}")
            self.leader.connect()
        logger.info(f"Connecting follower: {self.follower}")
        self.follower.connect()

        # Set current actions to current follower state.
        # Features missing from follower state silently default to 0.0.
        state = (self.follower.read_state())["state"]
        self._set_actions(self._align_feature_values(state))

        self.loaded_event.set()

    async def run_loop(self) -> None:
        try:
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
                        self._set_actions(filtered)
                    elif self.get_action_read_state() == ActionReadState.FROM_ACTIONS:
                        raw_actions = self.get_actions()
                        actions = {i: raw_actions[k] for k, i in enumerate(self.features)}
                        self.follower.set_joints_state(actions, goal_time * 3)
        finally:
            logger.info("Teleoperating stopped, disconnecting robots.")
            if self.leader:
                self.leader.disconnect()
            if self.follower:
                self.follower.disconnect()

    async def teardown(self) -> None:
        await super().teardown()
