from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from physicalai.robot.interface import Robot as PhysicalAIRobot
from pydantic import BaseModel, Field

from schemas.robot import RobotType

if TYPE_CHECKING:
    from uuid import UUID

    from schemas.calibration import Calibration
    from schemas.robot import SO101Robot


@dataclass(frozen=True)
class RobotAdapterOptions:
    # Adapter tuning options used when wrapping catalog robots into the PhysicalAI robot interface.
    # Defaults are conservative so existing robots keep current behavior unless overridden.
    # Include joint velocity values in the adapted observation/state output.
    include_velocities: bool = False
    # Multiplier applied to goal timing (1.0 keeps original goal timing).
    goal_time_scale: float = 1.0
    # Optional gain for externally provided effort/torque signals; None disables it.
    external_effort_gain: float | None = 0.1


class CatalogRobotFactory(Protocol):
    async def find_so101_port(self, robot: SO101Robot) -> str: ...

    async def find_port_by_serial(self, serial_number: str) -> str | None: ...

    async def get_calibration_by_id(self, calibration_id: UUID | None) -> Calibration | None: ...


_PayloadT = TypeVar("_PayloadT")


class PayloadContainer(Protocol[_PayloadT]):
    payload: _PayloadT


class CatalogRobot(PayloadContainer[_PayloadT], Protocol[_PayloadT]):
    type: RobotType
    active_calibration_id: UUID | None


_RobotT = TypeVar("_RobotT", bound=CatalogRobot[Any])
_FactoryT = TypeVar("_FactoryT", bound="CatalogRobotFactory")


BuildRobotCallable = Callable[[_RobotT, _FactoryT], Awaitable[PhysicalAIRobot]]


class RobotCatalogDefinition(BaseModel):
    type: RobotType = Field(..., description="Stable backend robot type identifier")
    display_name: str = Field(..., description="Human-readable robot type label")
    role: Literal["follower", "leader"] = Field(..., description="Default robot role")
    urdf_path: str = Field(description="URDF URL used by the UI model loader")
    package_map: dict[str, str] = Field(default_factory=dict, description="URDF package name to URL prefix map")
    joint_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Observation joint name to URDF joint(s) mapping",
    )
    urdf_relative_path: str = Field(..., description="Relative path to the robot URDF asset")

    robot_builder: BuildRobotCallable
    adapter_options: RobotAdapterOptions = RobotAdapterOptions()

    @property
    def robot_type(self) -> RobotType:
        return self.type
