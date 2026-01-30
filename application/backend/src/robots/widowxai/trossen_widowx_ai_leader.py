from typing import Any

import numpy as np
import trossen_arm
from loguru import logger

from robots.robot_client import RobotClient
from schemas import NetworkIpRobotConfig


class TrossenWidowXAILeader(RobotClient):
    def __init__(self, config: NetworkIpRobotConfig):
        self.driver = trossen_arm.TrossenArmDriver()
        self.driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            config.connection_string,
            True,
            timeout=30,
        )
        self.driver.set_all_modes(trossen_arm.Mode.external_effort)

        self.config: NetworkIpRobotConfig = config
        self.motor_names = {
            0: "shoulder_pan",
            1: "shoulder_lift",
            2: "elbow_flex",
            3: "wrist_flex",
            4: "wrist_roll",
            5: "wrist_yaw",
            6: "gripper",
        }
        self.name = "trossen_widowx_ai_leader"

    @property
    def feedback_features(self) -> dict:
        return {f"{motor}.eff": motor for motor in self.motor_names.values()}

    @property
    def is_connected(self) -> bool:
        return self.driver.get_is_configured()

    async def ping(self) -> dict:
        return self._create_event("pong")

    async def connect(self, calibrate: bool = False) -> None:  # noqa: ARG002
        self.driver.set_all_modes(trossen_arm.Mode.position)
        self.driver.set_all_positions(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2.0, True)
        self.driver.set_all_modes(trossen_arm.Mode.external_effort)
        self.driver.set_all_external_efforts(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0.0,
            False,
        )

    async def set_joints_state(self, joints: dict) -> dict:  # noqa: ARG002
        raise Exception("Not implemented for leaders")

    async def enable_torque(self) -> dict:
        return {}

    async def disable_torque(self) -> dict:
        return {}

    async def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        """Read current robot state. Returns state dict with timestamp."""
        try:
            observation = self.get_action()
            return self._create_event(
                "state_was_updated",
                state=observation,
                is_controlled=False,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

    async def read_forces(self) -> dict | None:
        pass

    async def set_forces(self, forces: dict) -> dict:
        force_feedback_gain = 0.1

        efforts = {key.removesuffix(".eff"): val for key, val in forces.items() if key.endswith(".eff")}

        effs = [0] * len(self.motor_names)

        # Map motor name / value pair into the right position of list
        for p, v in efforts.items():
            i = next((k for k, v in self.motor_names.items() if v == p), None)
            if i is not None:
                effs[i] = v

        self.driver.set_all_external_efforts(
            -force_feedback_gain * np.array(effs),
            0.0,
            False,
        )
        return forces

    def features(self) -> list[str]:
        pos = [f"{motor}.pos" for motor in self.motor_names.values()]
        vel = [f"{motor}.vel" for motor in self.motor_names.values()]
        return pos + vel

    def get_action(self) -> dict[str, Any]:
        positions = self.driver.get_all_positions()
        velocities = self.driver.get_all_velocities()

        action = {}
        for index, name in self.motor_names.items():
            if index >= len(positions) or index >= len(velocities):
                continue
            pos = np.rad2deg(positions[index])
            vel = velocities[index]
            action[f"{name}.pos"] = pos
            action[f"{name}.vel"] = vel

        return action

    async def disconnect(self) -> None:
        try:
            self.driver.set_all_modes(trossen_arm.Mode.position)
            self.driver.set_all_positions(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2.0, True)
        except Exception:
            logger.error("Failed to home trossen WidowX AI leader")
        finally:
            self.driver.cleanup()
