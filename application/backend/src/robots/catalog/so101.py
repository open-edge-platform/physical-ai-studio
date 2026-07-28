from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from physicalai.robot.so101 import SO101, SO101Calibration, SO101JointCalibration
from physicalai_studio_plugin import RobotAdapterOptions, RobotAsset, RobotCatalogDefinition
from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas import SerialPortInfo
from schemas.robot_type import BaseRobot

SO101Types = Literal["SO101_Follower", "SO101_Leader"]

if TYPE_CHECKING:
    from physicalai_studio_plugin import CatalogRobot, CatalogRobotFactory, PortScanner

    from schemas.robot import Robot


class SO101CalibrationValue(BaseModel):
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


class SO101RobotPayload(BaseModel):
    """Connection configuration for SO-101 serial robots."""

    connection_string: str = Field(
        default="",
        description="Serial port path; leave empty to auto-discover via serial_number",
    )
    serial_number: str = Field(default="", description="USB serial number of the robot (when available)")
    calibration: dict[str, SO101JointCalibration] | None = Field(
        default=None,
        description="Per-joint calibration values (id, drive_mode, homing_offset, range_min, range_max)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "connection_string": "",
                "serial_number": "SO101-2024-001",
                "calibration": None,
            },
        },
    )

    @model_validator(mode="after")
    def validate_identifier(self) -> SO101RobotPayload:
        if self.connection_string == "" and self.serial_number == "":
            raise ValueError("Either serial_number or connection_string is required for SO101 robots")
        return self


class SO101Robot(BaseRobot):
    """SO-101 follower or leader robot using a serial connection."""

    type: SO101Types = Field(..., description="Type of robot configuration")
    payload: SO101RobotPayload = Field(..., description="SO-101 connection configuration")


_SO101_TO_URDF = {
    "shoulder_pan.pos": ["shoulder_pan"],
    "shoulder_lift.pos": ["shoulder_lift"],
    "elbow_flex.pos": ["elbow_flex"],
    "wrist_flex.pos": ["wrist_flex"],
    "wrist_roll.pos": ["wrist_roll"],
    "gripper.pos": ["gripper"],
}

_SO101_ASSET = RobotAsset(
    urdf_relative_path=Path("SO101/so101_new_calib.urdf"),
    packages={"SO101": Path("SO101")},
    joint_map=_SO101_TO_URDF,
)


async def _build_so101_driver(robot: CatalogRobot[SO101RobotPayload], factory: CatalogRobotFactory) -> SO101:
    if not isinstance(robot.payload, SO101RobotPayload):
        raise TypeError("Expected SO101Robot")
    port_info = SerialPortInfo(
        connection_string=robot.payload.connection_string or None,
        serial_number=robot.payload.serial_number or None,
    )
    port = await factory.find_port(port_info)
    if port is None:
        resource_key = robot.payload.serial_number or robot.payload.connection_string
        raise ValueError(f"Could not resolve a serial port for {resource_key}")

    calibration: SO101Calibration | None = None
    if robot.payload.calibration is not None:
        calibration = SO101Calibration(joints=robot.payload.calibration)

    role = "follower" if robot.type == "SO101_Follower" else "leader"
    return SO101(port=port, calibration=calibration, role=role, unit="normalized")


def serial_port_from_so101(robot: SO101Robot) -> SerialPortInfo:
    """Build a serial identity from an SO101 robot configuration."""
    connection_string = robot.payload.connection_string or None
    serial_number = robot.payload.serial_number or None
    return SerialPortInfo(connection_string=connection_string, serial_number=serial_number)


def _resolve_serial_port(discovered: list[SerialPortInfo], target: SerialPortInfo) -> str | None:
    if target.serial_number is not None:
        for serial_port in discovered:
            if serial_port.serial_number == target.serial_number:
                return serial_port.connection_string
        return None

    for serial_port in discovered:
        if serial_port.connection_string == target.connection_string:
            return serial_port.connection_string
    return None


async def find_so101_port(
    manager: PortScanner,
    serial_port: SerialPortInfo,
) -> str | None:
    """Find the current port for an SO101 robot by serial number or configured port."""
    port = _resolve_serial_port(manager.robots, serial_port)
    if port is not None:
        return port

    await manager.find_robots()
    return _resolve_serial_port(manager.robots, serial_port)


async def identify_so101_robot_visually(
    manager: PortScanner,
    robot: Robot,
    joint: str | None = None,
) -> None:
    """Identify the robot by moving the joint from current to min to max to initial position."""
    import asyncio

    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    if not isinstance(robot.payload, SO101RobotPayload):
        raise ValueError(f"Trying to identify unsupported robot: {robot.type}")

    if joint is None:
        joint = "gripper"

    connection_string = await find_so101_port(manager, serial_port_from_so101(robot))

    if connection_string is None:
        if robot.payload.serial_number:
            raise ValueError(f"Could not find the serial port for serial number {robot.payload.serial_number}")
        raise ValueError("Could not resolve a serial port from connection_string")
    connection = SOFollower(SOFollowerRobotConfig(port=connection_string))
    connection.bus.connect()

    PRESENT_POSITION_KEY = "Present_Position"
    GOAL_POSITION_KEY = "Goal_Position"

    current_position = connection.bus.sync_read(PRESENT_POSITION_KEY, normalize=False)
    gripper_calibration = connection.bus.read_calibration()[joint]
    connection.bus.write(GOAL_POSITION_KEY, joint, gripper_calibration.range_min, normalize=False)
    await asyncio.sleep(1)
    connection.bus.write(GOAL_POSITION_KEY, joint, gripper_calibration.range_max, normalize=False)
    await asyncio.sleep(1)
    connection.bus.write(GOAL_POSITION_KEY, joint, current_position[joint], normalize=False)
    await asyncio.sleep(1)
    connection.bus.disconnect()


class SO101Probe:
    """Probe for SO101 robots — serial port discovery + joint identification."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        await manager.find_robots()
        return manager.robots

    async def identify(
        self,
        payload: SO101RobotPayload,
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None:
        if manager is None:
            raise ValueError("PortScanner required for SO101 identification")

        now = datetime.now()
        robot = SO101Robot(
            id=UUID(int=0),
            name="",
            type="SO101_Follower",
            payload=payload,
            created_at=now,
            updated_at=now,
        )
        await identify_so101_robot_visually(manager, robot, joint)

    async def is_online(self, payload: SO101RobotPayload, manager: PortScanner | None = None) -> bool:
        serial_port = SerialPortInfo(
            connection_string=payload.connection_string or None,
            serial_number=payload.serial_number or None,
        )

        if manager is not None:
            return await find_so101_port(manager, serial_port) is not None

        from serial.tools import list_ports

        discovered = [
            SerialPortInfo(
                connection_string=port.device,
                serial_number=getattr(port, "serial_number", None) or None,
            )
            for port in list_ports.comports()
        ]
        return _resolve_serial_port(discovered, serial_port) is not None


_SO101_PROBE = SO101Probe()


def get_definitions() -> list[RobotCatalogDefinition]:
    """Return built-in SO101 robot catalog definitions."""
    return [
        RobotCatalogDefinition(
            type="SO101_Follower",
            display_name="SO101 Follower",
            role="follower",
            robot_builder=_build_so101_driver,
            robot_payload=SO101RobotPayload,
            asset=_SO101_ASSET,
            adapter_options=RobotAdapterOptions(goal_time_scale=1.0, external_effort_gain=None),
            probe=_SO101_PROBE,
        ),
        RobotCatalogDefinition(
            type="SO101_Leader",
            display_name="SO101 Leader",
            role="leader",
            robot_builder=_build_so101_driver,
            robot_payload=SO101RobotPayload,
            asset=_SO101_ASSET,
            adapter_options=RobotAdapterOptions(goal_time_scale=1.0, external_effort_gain=None),
            probe=_SO101_PROBE,
        ),
    ]
