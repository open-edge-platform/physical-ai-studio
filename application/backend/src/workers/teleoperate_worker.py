import asyncio
import ctypes
import enum
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from typing import Any

from loguru import logger

from robots.robot_client import RobotClient

from .base import BaseProcessWorker, run_at_frequency


class ActionWriteState(enum.IntEnum):
    NONE = 0
    FROM_LEADER = 1
    FROM_ACTIONS = 2


class TeleoperateWorker(BaseProcessWorker):
    ROLE: str = "TeleoperateWorker"

    follower: RobotClient
    leader: RobotClient | None
    stop_event: EventClass
    _action_source: Any
    _output_actions: Any
    _output_state: Any

    def __init__(self, follower: RobotClient, leader: RobotClient | None, frequency: float, mp_stop_event: EventClass):
        buffer_length = len(follower.features())
        self.loaded_event = mp.Event()
        self._action_source = mp.Value(ctypes.c_int, ActionWriteState.NONE)
        self._output_actions = mp.Array(ctypes.c_double, buffer_length)
        self._output_state = mp.Array(ctypes.c_double, buffer_length)

        super().__init__(
            stop_event=mp_stop_event,
            queues_to_cancel=[],
        )
        self.follower = follower
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

    def get_action_source(self) -> int:
        return self._action_source.value

    def set_action_source(self, value: ActionWriteState) -> None:
        with self._action_source.get_lock():
            self._action_source.value = value

    async def wait_for_loading_to_complete(self) -> None:
        await asyncio.to_thread(self.loaded_event.wait)

    async def run_loop(self) -> None:
        try:
            features = self.follower.features()
            if self.leader is not None:
                logger.info(f"Connecting leader: {self.leader}")
                self.leader.connect()
            logger.info(f"Connecting follower: {self.follower}")
            self.follower.connect()
            self.loaded_event.set()

            # Teleoperate loop until unload is requested
            goal_time = 1 / self.frequency
            while not self.should_stop():
                async with run_at_frequency(self.frequency):
                    state = (self.follower.read_state())["state"]
                    self._set_state([state[key] for key in features])
                    if self.get_action_source() == ActionWriteState.FROM_LEADER and self.leader is not None:
                        actions = (self.leader.read_state())["state"]
                        self.follower.set_joints_state(actions, goal_time * 2)
                        self._set_actions([actions[key] for key in features])
                    elif self.get_action_source() == ActionWriteState.FROM_ACTIONS:
                        raw_actions = self.get_actions()
                        actions = {i: raw_actions[k] for k, i in enumerate(features)}
                        self.follower.set_joints_state(actions, goal_time * 3)
        finally:
            logger.info("Teleoperating stopped, disconnecting robots.")
            if self.leader:
                self.leader.disconnect()
            if self.follower:
                self.follower.disconnect()

    async def teardown(self) -> None:
        await super().teardown()


#!/usr/bin/env python3
