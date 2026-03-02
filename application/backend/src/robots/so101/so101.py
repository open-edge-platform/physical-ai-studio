from schemas.calibration import Calibration
from typing import Literal

from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
from loguru import logger

from robots.robot_client import RobotClient
from schemas.robot import RobotType


def clamp_joints(current: dict, target: dict, max_distance: float) -> dict:
    """Clamp a current joints dict to target with a max value"""
    return {key: value + clamp(target[key] - value, max_distance) for key, value in current.items()}


def clamp(value: float, min_max: float) -> float:
    """Clamp value between -min_max and min_max."""
    return max(min(value, min_max), -min_max)


RobotMode = Literal["follower", "teleoperator"]


class So101(RobotClient):
    name = "So101"

    previous_target: dict[str, float] | None = None
    fps = 30
    max_speed = 270  # From feetech 12V servo spec: 60 deg / 0.222s

    id: str
    port: str
    bus: FeetechMotorsBus
    is_controlled: bool = False

    def __init__(self, port: str, id: str, mode: RobotMode, calibration: Calibration):
        norm_mode_body = MotorNormMode.RANGE_M100_100
        self.calibration = calibration
        self.bus = FeetechMotorsBus(
            port=port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self._convert_calibration_to_dict(calibration)
        )
        self.id = id
        self.port = port
        self.mode = mode

    @staticmethod
    def _convert_calibration_to_dict(calibration: Calibration) -> dict[str, MotorCalibration]:
        return {key: MotorCalibration(
            id=values.id,
            drive_mode=values.drive_mode,
            homing_offset=values.homing_offset,
            range_min=values.range_min,
            range_max=values.range_max,
        ) for key, values in calibration.values.items()}

    @property
    def robot_type(self) -> RobotType:
        """Specify the RobotType"""
        if self.mode == "follower":
            return RobotType.SO101_FOLLOWER
        return RobotType.SO101_LEADER

    @property
    async def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected

    async def connect(self) -> None:
        """Connect to the robot."""
        logger.info(f"Connecting to SO {self.mode} on port {self.port}")
        self.bus.connect()
        self.configure()

    def configure(self) -> None:
        if not self.bus.is_calibrated:
            logger.info(f"Not calibrated, writing {self.calibration}")
            self.bus.write_calibration(self.calibration)

        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def _configure_leader(self) -> None:
        logger.info("_configure_leader")
        self.is_controlled = True
        self.bus.disable_torque()

    def _configure_follower(self) -> None:
        logger.info("_configure_follower")
        self.is_controlled = False
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of the max torque limit to avoid burnout
                    self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def _move_to_target(self, joints: dict, goal_time: float) -> None:
        max_rotational_distance = self.max_speed * goal_time

        state = self._get_state()
        if self.previous_target:
            # Additional clamp to make sure that previous_target is not too far of current position
            state = clamp_joints(state, self.previous_target, max_rotational_distance * 2)

        target = {key: value + clamp(joints[key] - value, max_rotational_distance) for key, value in state.items()}
        self.previous_target = target

        goal_pos = {key.removesuffix(".pos"): val for key, val in target.items() if key.endswith(".pos")}
        self.bus.sync_write("Goal_Position", goal_pos)

    def _get_state(self) -> dict:
        obs_dict = self.bus.sync_read("Present_Position")
        return {f"{motor}.pos": val for motor, val in obs_dict.items()}

    async def disconnect(self) -> None:
        """Disconnect from the robot."""
        logger.info(f"Disconnecting to SO {self.mode} on port {self.port}")
        self.bus.disconnect(True)

    async def ping(self) -> dict:
        """Send ping command. Returns event dict with timestamp."""
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict, goal_time: float) -> dict:
        """Set joint positions. Returns event dict with timestamp."""
        self._move_to_target(joints, goal_time)
        return self._create_event(
            "joints_state_was_set",
            joints=joints,
        )

    async def enable_torque(self) -> dict:
        """Enable torque. Returns event dict with timestamp."""
        self._configure_follower()
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        """Disable torque. Returns event dict with timestamp."""
        logger.info("Disabling torque")
        self._configure_leader()
        return self._create_event("torque_was_disabled")

    async def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        """Read current robot state. Returns state dict with timestamp.

        Example state: {
            'shoulder_pan.pos': -8.705526116578355,
            'shoulder_lift.pos': -98.16753926701571,
            'elbow_flex.pos': 95.98393574297188,
            'wrist_flex.pos': 73.85993485342019,
            'wrist_roll.pos': -13.84615384615384,
            'gripper.pos': 26.885644768856448
        }
        """
        try:
            state = self._get_state()
            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self.is_controlled,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

    async def read_forces(self) -> dict | None:
        """Read current robot forces. Returns state dict with timestamp."""
        return self._create_event(
            "force_was_updated",
            state=None,
            is_controlled=self.is_controlled,
        )

    async def set_forces(self, forces: dict) -> dict:  # noqa: ARG002
        """Set current robot forces. Returns event dict with timestamp."""
        raise Exception("Not implemented for SO101")

    def features(self) -> list[str]:
        """Get Robot features. Returns list with joints."""
        return [f"{motor}.pos" for motor in self.bus.motors]
