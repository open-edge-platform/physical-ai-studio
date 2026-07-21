from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger
from physicalai.robot.interface import Robot, RobotObservation

from robots.robot_client import RobotClient
from schemas.robot import RobotType


@dataclass(frozen=True)
class PhysicalAIRobotAdapterConfig:
    # Include ``<joint>.vel`` values from observation sensor data.
    include_velocities: bool = False
    # Multiplier for ``goal_time`` when forwarding actions.
    goal_time_scale: float = 1.0
    # Optional gain for ``set_external_efforts``; ``None`` disables force writes.
    # When ``None``, missing effort observations emit a force event with ``state=None``.
    external_effort_gain: float | None = 0.1


class PhysicalAIRobotAdapter(RobotClient):
    name = "PhysicalAIRobot"

    def __init__(
        self,
        *,
        robot: Robot,
        robot_type: RobotType,
        robot_role: Literal["follower", "leader"],
        config: PhysicalAIRobotAdapterConfig | None = None,
    ) -> None:
        resolved_config = config or PhysicalAIRobotAdapterConfig()
        self._robot = robot
        self._robot_type = robot_type
        self._robot_role = robot_role
        self._config = resolved_config
        self.is_controlled = False

    def _is_follower(self) -> bool:
        return self._robot_role == "follower"

    def _observation_to_state(self, observation: RobotObservation) -> dict[str, float]:
        state: dict[str, float] = {}
        for i, name in enumerate(self._robot.joint_names):
            raw_position = float(observation.joint_positions[i])
            state[f"{name}.pos"] = raw_position

        if self._config.include_velocities:
            sensor_data = observation.sensor_data
            if sensor_data is None or "velocities" not in sensor_data:
                msg = "Robot observation is missing velocity data"
                raise RuntimeError(msg)
            velocities = sensor_data["velocities"]
            for i, name in enumerate(self._robot.joint_names):
                state[f"{name}.vel"] = float(velocities[i])

        return state

    def _state_to_action(self, joints: dict[str, float]) -> np.ndarray:
        action = np.empty(len(self._robot.joint_names), dtype=np.float32)
        for i, name in enumerate(self._robot.joint_names):
            action[i] = float(joints[f"{name}.pos"])
        return action

    @property
    def robot_type(self) -> RobotType:
        return self._robot_type

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected()

    def connect(self) -> None:
        logger.info(f"Connecting physicalai robot type={self._robot_type}")
        try:
            self._robot.connect()
            self.is_controlled = self._is_follower()
        except TimeoutError:
            logger.error("Timeout connecting to robot")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            raise

    def disconnect(self) -> None:
        logger.info(f"Disconnecting physicalai robot type={self._robot_type}")
        try:
            self._robot.disconnect()
            logger.info("Robot disconnected")
        except TimeoutError:
            logger.warning("Timeout during robot disconnect - forcing cleanup")
        except Exception as e:
            logger.error(f"Error during robot disconnect: {e}")

    def ping(self) -> dict:
        return self._create_event("pong")

    def set_joints_state(self, joints: dict, goal_time: float) -> dict:
        action = self._state_to_action(joints)
        self._robot.send_action(action, goal_time=self._config.goal_time_scale * goal_time)
        return self._create_event("joints_state_was_set", joints=joints)

    def enable_torque(self) -> dict:
        set_torque = getattr(self._robot, "set_torque", None)
        if callable(set_torque):
            set_torque(enabled=True)
        self.is_controlled = True
        return self._create_event("torque_was_enabled")

    def disable_torque(self) -> dict:
        set_torque = getattr(self._robot, "set_torque", None)
        if callable(set_torque):
            set_torque(enabled=False)
        self.is_controlled = False
        return self._create_event("torque_was_disabled")

    def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        try:
            observation = self._robot.get_observation()
            state = self._observation_to_state(observation)

            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self.is_controlled,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

    def read_forces(self) -> dict | None:
        observation = self._robot.get_observation()

        sensor_data = observation.sensor_data
        if sensor_data is None or "efforts" not in sensor_data:
            if self._config.external_effort_gain is None:
                return self._create_event(
                    "force_was_updated",
                    state=None,
                    is_controlled=self.is_controlled,
                )
            return None

        efforts = sensor_data["efforts"]
        forces = {f"{name}.eff": float(efforts[i]) for i, name in enumerate(self._robot.joint_names)}

        return self._create_event(
            "force_was_updated",
            state=forces,
            is_controlled=self.is_controlled,
        )

    def set_forces(self, forces: dict) -> dict:
        if self._is_follower():
            logger.warning("Cannot send forces to a follower arm")
            return forces

        gain = self._config.external_effort_gain
        if gain is None:
            return forces

        set_external_efforts = getattr(self._robot, "set_external_efforts", None)
        if not callable(set_external_efforts):
            return forces

        efforts = np.zeros(len(self._robot.joint_names), dtype=np.float32)
        for i, name in enumerate(self._robot.joint_names):
            efforts[i] = float(forces.get(f"{name}.eff", 0.0))

        set_external_efforts(efforts, gain=gain)
        return forces

    def features(self) -> list[str]:
        position_features = [f"{name}.pos" for name in self._robot.joint_names]
        if not self._config.include_velocities:
            return position_features
        velocity_features = [f"{name}.vel" for name in self._robot.joint_names]
        return position_features + velocity_features
