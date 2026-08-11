from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from physicalai.robot import BimanualWidowXAI, WidowXAI
from physicalai_studio_plugin import RobotAdapterOptions, RobotAsset, RobotCatalogDefinition
from pydantic import BaseModel, ConfigDict, Field

from schemas.robot_type import BaseRobot

TrossenTypes = Literal["Trossen_WidowXAI_Follower", "Trossen_WidowXAI_Leader"]
TrossenBimanualTypes = Literal["Trossen_Bimanual_WidowXAI_Follower", "Trossen_Bimanual_WidowXAI_Leader"]

if TYPE_CHECKING:
    from physicalai_studio_plugin import CatalogRobot, CatalogRobotFactory, PortScanner

    from schemas import SerialPortInfo


class TrossenSingleArmPayload(BaseModel):
    """Connection configuration for Trossen single-arm robots."""

    connection_string: str = Field(..., description="IP address of the robot")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "connection_string": "192.168.1.100",
            },
        },
    )


class TrossenBimanualPayload(BaseModel):
    """Connection configuration for Trossen bimanual robots."""

    connection_string_left: str = Field(..., description="IP address of the left arm")
    connection_string_right: str = Field(..., description="IP address of the right arm")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "connection_string_left": "192.168.1.100",
                "connection_string_right": "192.168.1.101",
            },
        },
    )


class TrossenSingleArmRobot(BaseRobot):
    """Trossen WidowX AI follower or leader robot using an IP connection."""

    type: TrossenTypes = Field(..., description="Type of robot configuration")
    payload: TrossenSingleArmPayload = Field(..., description="Trossen single-arm connection configuration")


class TrossenBimanualRobot(BaseRobot):
    """Trossen Bimanual WidowX AI robot using two IP connections (left + right)."""

    type: TrossenBimanualTypes = Field(..., description="Type of robot configuration")
    payload: TrossenBimanualPayload = Field(..., description="Trossen bimanual connection configuration")


_TROSSEN_TO_URDF = {
    "shoulder_pan.pos": ["joint_0"],
    "shoulder_lift.pos": ["joint_1"],
    "elbow_flex.pos": ["joint_2"],
    "wrist_flex.pos": ["joint_3"],
    "wrist_yaw.pos": ["joint_4"],
    "wrist_roll.pos": ["joint_5"],
    "gripper.pos": ["left_carriage_joint", "right_carriage_joint"],
}

_BIMANUAL_TROSSEN_TO_URDF = {
    "left_shoulder_pan.pos": ["follower_left_joint_0"],
    "left_shoulder_lift.pos": ["follower_left_joint_1"],
    "left_elbow_flex.pos": ["follower_left_joint_2"],
    "left_wrist_flex.pos": ["follower_left_joint_3"],
    "left_wrist_yaw.pos": ["follower_left_joint_4"],
    "left_wrist_roll.pos": ["follower_left_joint_5"],
    "left_gripper.pos": ["follower_left_left_carriage_joint", "follower_left_right_carriage_joint"],
    "right_shoulder_pan.pos": ["follower_right_joint_0"],
    "right_shoulder_lift.pos": ["follower_right_joint_1"],
    "right_elbow_flex.pos": ["follower_right_joint_2"],
    "right_wrist_flex.pos": ["follower_right_joint_3"],
    "right_wrist_yaw.pos": ["follower_right_joint_4"],
    "right_wrist_roll.pos": ["follower_right_joint_5"],
    "right_gripper.pos": ["follower_right_left_carriage_joint", "follower_right_right_carriage_joint"],
}

_TROSSEN_SINGLE_ARM_ASSET = RobotAsset(
    urdf_relative_path=Path("widowx/urdf/generated/wxai/wxai_follower.urdf"),
    packages={"trossen_arm_description": Path("widowx")},
    joint_map=_TROSSEN_TO_URDF,
)

_TROSSEN_BIMANUAL_ASSET = RobotAsset(
    urdf_relative_path=Path("widowx/urdf/generated/stationary_ai.urdf"),
    packages={"trossen_arm_description": Path("widowx")},
    joint_map=_BIMANUAL_TROSSEN_TO_URDF,
)


async def _build_trossen_single_arm_driver(
    robot: CatalogRobot[TrossenSingleArmPayload], _factory: CatalogRobotFactory
) -> WidowXAI:
    # Studio generates the robot model per registered type, so match on the
    # payload; the model is never a ``TrossenSingleArmRobot`` instance.
    payload = robot.payload
    if not isinstance(payload, TrossenSingleArmPayload):
        raise TypeError(f"Expected TrossenSingleArmPayload, got {type(payload).__name__}")
    role = "follower" if robot.type == "Trossen_WidowXAI_Follower" else "leader"
    return WidowXAI(ip=payload.connection_string, role=role)


async def _build_trossen_bimanual_driver(
    robot: CatalogRobot[TrossenBimanualPayload], _factory: CatalogRobotFactory
) -> BimanualWidowXAI:
    payload = robot.payload
    if not isinstance(payload, TrossenBimanualPayload):
        raise TypeError(f"Expected TrossenBimanualPayload, got {type(payload).__name__}")
    mode = "follower" if robot.type == "Trossen_Bimanual_WidowXAI_Follower" else "leader"
    return BimanualWidowXAI(
        left=WidowXAI(ip=payload.connection_string_left, role=mode),
        right=WidowXAI(ip=payload.connection_string_right, role=mode),
    )


async def _ping(ip: str, ping_timeout: float = 1.0) -> bool:
    import asyncio
    import sys

    param = "-n" if sys.platform.lower().startswith("win") else "-c"
    command = ["ping", param, "1", "-W", str(int(ping_timeout * 1000)), ip]
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        return (await asyncio.wait_for(proc.wait(), timeout=ping_timeout + 0.5)) == 0
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False


async def identify_trossen_robot_visually(robot: TrossenSingleArmRobot) -> None:
    """Identify the robot by moving the gripper from current to open to closed to initial."""
    import trossen_arm
    from loguru import logger

    driver = trossen_arm.TrossenArmDriver()

    logger.info("Configuring the drivers...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        robot.payload.connection_string,
        True,
        timeout=5,
    )

    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.02, 0.5, True)
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.0, 0.5, True)


class TrossenSingleArmProbe:
    """Probe for Trossen single-arm robots — IP-based gripper identification."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:  # noqa: ARG002
        return []

    async def identify(
        self,
        payload: TrossenSingleArmPayload,
        manager: PortScanner | None = None,  # noqa: ARG002
        joint: str | None = None,  # noqa: ARG002
    ) -> None:
        now = datetime.now()
        robot = TrossenSingleArmRobot(
            id=UUID(int=0),
            name="",
            type="Trossen_WidowXAI_Follower",
            payload=payload,
            created_at=now,
            updated_at=now,
        )
        await identify_trossen_robot_visually(robot)

    async def is_online(self, payload: TrossenSingleArmPayload, manager: PortScanner | None = None) -> bool:  # noqa: ARG002
        if not payload.connection_string:
            return False
        return await _ping(payload.connection_string)


class TrossenBimanualProbe:
    """Probe for Trossen bimanual robots — per-arm IP-based identification."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:  # noqa: ARG002
        return []

    async def identify(
        self,
        payload: TrossenBimanualPayload,
        manager: PortScanner | None = None,  # noqa: ARG002
        joint: str | None = None,  # noqa: ARG002
    ) -> None:
        now = datetime.now()
        left_payload = TrossenSingleArmPayload(connection_string=payload.connection_string_left)
        left_robot = TrossenSingleArmRobot(
            id=UUID(int=0),
            name="",
            type="Trossen_WidowXAI_Follower",
            payload=left_payload,
            created_at=now,
            updated_at=now,
        )
        await identify_trossen_robot_visually(left_robot)

        right_payload = TrossenSingleArmPayload(connection_string=payload.connection_string_right)
        right_robot = TrossenSingleArmRobot(
            id=UUID(int=0),
            name="",
            type="Trossen_WidowXAI_Follower",
            payload=right_payload,
            created_at=now,
            updated_at=now,
        )
        await identify_trossen_robot_visually(right_robot)

    async def is_online(self, payload: TrossenBimanualPayload, manager: PortScanner | None = None) -> bool:  # noqa: ARG002
        import asyncio

        left = payload.connection_string_left
        right = payload.connection_string_right
        if not left or not right:
            return False
        left_ok, right_ok = await asyncio.gather(_ping(left), _ping(right))
        return left_ok and right_ok


_SINGLE_ARM_PROBE = TrossenSingleArmProbe()
_BIMANUAL_PROBE = TrossenBimanualProbe()


def get_definitions() -> list[RobotCatalogDefinition]:
    """Return built-in WidowX AI robot catalog definitions."""
    return [
        RobotCatalogDefinition(
            type="Trossen_WidowXAI_Follower",
            display_name="Trossen WidowX AI Follower",
            role="follower",
            robot_builder=_build_trossen_single_arm_driver,
            robot_payload=TrossenSingleArmPayload,
            asset=_TROSSEN_SINGLE_ARM_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_SINGLE_ARM_PROBE,
        ),
        RobotCatalogDefinition(
            type="Trossen_WidowXAI_Leader",
            display_name="Trossen WidowX AI Leader",
            role="leader",
            robot_builder=_build_trossen_single_arm_driver,
            robot_payload=TrossenSingleArmPayload,
            asset=_TROSSEN_SINGLE_ARM_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_SINGLE_ARM_PROBE,
        ),
        RobotCatalogDefinition(
            type="Trossen_Bimanual_WidowXAI_Follower",
            display_name="Trossen Bimanual WidowX AI Follower",
            role="follower",
            robot_builder=_build_trossen_bimanual_driver,
            robot_payload=TrossenBimanualPayload,
            asset=_TROSSEN_BIMANUAL_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_BIMANUAL_PROBE,
        ),
        RobotCatalogDefinition(
            type="Trossen_Bimanual_WidowXAI_Leader",
            display_name="Trossen Bimanual WidowX AI Leader",
            role="leader",
            robot_builder=_build_trossen_bimanual_driver,
            robot_payload=TrossenBimanualPayload,
            asset=_TROSSEN_BIMANUAL_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_BIMANUAL_PROBE,
        ),
    ]
