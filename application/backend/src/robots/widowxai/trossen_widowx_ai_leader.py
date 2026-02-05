import numpy as np
import trossen_arm
from loguru import logger

from robots.robot_client import RobotClient
from schemas import NetworkIpRobotConfig
from schemas.robot import RobotType


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
            4: "wrist_yaw",
            5: "wrist_roll",
            6: "gripper",
        }
        self.name = "trossen_widowx_ai_leader"

    @property
    def robot_type(self) -> RobotType:
        return RobotType.TROSSEN_WIDOWXAI_LEADER

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
        """Read current robot state. Returns state dict with timestamp.

        Example state: {
            'elbow_flex.pos': 4.535314764553813,
            'elbow_flex.vel': -0.0024420025292783976,
            'gripper.pos': -2.8371810913085938e-05,
            'gripper.vel': -0.0001923076924867928,
            'shoulder_lift.pos': 0.4917811441211757,
            'shoulder_lift.vel': -0.0024420025292783976,
            'shoulder_pan.pos': 0.03278540827405706,
            'shoulder_pan.vel': -0.0024420025292783976,
            'wrist_flex.pos': 4.273031658443915,
            'wrist_flex.vel': -0.007326007355004549,
            'wrist_roll.pos': -0.09835622482217117,
            'wrist_roll.vel': -0.007326007355004549,
            'wrist_yaw.pos': 0.1420700958508073,
            'wrist_yaw.vel': -0.007326007355004549
        }`
        """
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

    def get_action(self) -> dict:
        positions = self.driver.get_all_positions()
        velocities = self.driver.get_all_velocities()

        obs_dict = {}
        # First: all positions
        for index, name in self.motor_names.items():
            if index >= len(positions):
                continue
            obs_dict[f"{name}.pos"] = np.rad2deg(positions[index]) if "gripper" not in name else positions[index]

        # Then: all velocities
        for index, name in self.motor_names.items():
            if index >= len(velocities):
                continue
            obs_dict[f"{name}.vel"] = velocities[index]

        return obs_dict

    async def disconnect(self) -> None:
        try:
            self.driver.set_all_modes(trossen_arm.Mode.position)
            self.driver.set_all_positions(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2.0, True)
        except Exception:
            logger.error("Failed to home trossen WidowX AI leader")
        finally:
            self.driver.cleanup()
