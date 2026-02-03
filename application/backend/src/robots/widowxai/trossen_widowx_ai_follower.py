import numpy as np
import trossen_arm
from loguru import logger

from robots.robot_client import RobotClient
from schemas import NetworkIpRobotConfig
from schemas.robot import RobotType


class TrossenWidowXAIFollower(RobotClient):
    def __init__(self, config: NetworkIpRobotConfig):
        self.driver = trossen_arm.TrossenArmDriver()
        self.connection_string = config.connection_string

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
        self.name = "trossen_widowx_ai_follower"

    @property
    def robot_type(self) -> RobotType:
        return RobotType.TROSSEN_WIDOWXAI_FOLLOWER

    @property
    def _motors_ft(self):
        pos = {f"{motor}.pos": float for motor in self.motor_names.values()}
        vel = {f"{motor}.vel": float for motor in self.motor_names.values()}
        return {**pos, **vel}

    @property
    def is_connected(self) -> bool:
        return self.driver.get_is_configured()

    async def ping(self) -> dict:
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict) -> dict:
        positions = {key.removesuffix(".pos"): val for key, val in joints.items() if key.endswith(".pos")}
        velocities = {key.removesuffix(".vel"): val for key, val in joints.items() if key.endswith(".vel")}

        ps = [0] * len(self.motor_names)
        vs = [0] * len(self.motor_names)

        # Map motor name / value pair into the right position of list
        for p, v in positions.items():
            i = next((k for k, v in self.motor_names.items() if v == p), None)
            if i is not None:
                ps[i] = np.deg2rad(v) if "gripper" not in p else v

        # Map motor name / value pair into the right position of list
        for p, v in velocities.items():
            i = next((k for k, v in self.motor_names.items() if v == p), None)
            if i is not None:
                vs[i] = v

        self.driver.set_all_positions(
            ps,
            0.0,
            False,
            vs,
        )

        return joints

    async def enable_torque(self) -> dict:
        return {}

    async def disable_torque(self) -> dict:
        return {}

    async def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        """Read current robot state. Returns state dict with timestamp.

        Example state: {
            'elbow_flex.pos': 4.535314764553813,
            'elbow_flex.vel': -0.0024420025292783976,
            'gripper.pos': -1.7546117305755615e-06,
            'gripper.vel': -6.410256173694506e-05,
            'shoulder_lift.pos': 0.4917811441211757,
            'shoulder_lift.vel': -0.0024420025292783976,
            'shoulder_pan.pos': 0.03278540827405706,
            'shoulder_pan.vel': 0.0024420025292783976,
            'wrist_flex.pos': 4.4260300302863165,
            'wrist_flex.vel': -0.007326007355004549,
            'wrist_roll.pos': -0.09835622482217117,
            'wrist_roll.vel': -0.021978022530674934,
            'wrist_yaw.pos': 0.12021317034164915,
            'wrist_yaw.vel': -0.007326007355004549
        }
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
        try:
            forces = self.get_forces()
            return self._create_event(
                "state_was_updated",
                state=forces,
                is_controlled=False,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

    async def set_forces(self, forces: dict) -> dict:
        return forces

    def get_action(self) -> dict:
        positions = self.driver.get_all_positions()
        velocities = self.driver.get_all_velocities()
        # efforts = self.driver.get_all_external_efforts()

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

    def get_forces(self) -> dict:
        efforts = self.driver.get_all_external_efforts()

        obs_dict = {}
        for index, name in self.motor_names.items():
            if index >= len(efforts) or index >= len(efforts):
                continue
            eff = efforts[index]
            obs_dict[f"{name}.eff"] = eff

        return obs_dict

    def features(self) -> list[str]:
        pos = [f"{motor}.pos" for motor in self.motor_names.values()]
        vel = [f"{motor}.vel" for motor in self.motor_names.values()]
        # eff = [f"{motor}.eff" for motor in self.motor_names.values()]
        return pos + vel

    async def connect(self, calibrate: bool = False) -> None:  # noqa: ARG002
        self.driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_follower,
            self.connection_string,
            True,
            timeout=5,
        )
        self.driver.set_all_modes(trossen_arm.Mode.position)

        self.driver.set_all_modes(trossen_arm.Mode.position)
        self.driver.set_all_positions(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2.0, True)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    async def disconnect(self) -> None:
        try:
            self.driver.set_all_modes(trossen_arm.Mode.position)
            self.driver.set_all_positions(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2.0, True)
        except Exception:
            logger.error("Failed to home trossen WidowX AI follower")
        finally:
            self.driver.cleanup()
