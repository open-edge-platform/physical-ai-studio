"""Core plugin catalog protocol and definition types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar

from physicalai.robot.interface import Robot as PhysicalAIRobot
from pydantic import BaseModel

from .factory import CatalogRobotFactory

if TYPE_CHECKING:
    from .assets import RobotAsset
    from .probe import RobotProbe


@dataclass(frozen=True)
class RobotAdapterOptions:
    """Controls adapter behavior for velocity, timing, and effort handling."""

    include_velocities: bool = False
    goal_time_scale: float = 1.0
    external_effort_gain: float | None = 0.1


_PayloadT = TypeVar("_PayloadT", bound=BaseModel)
_PayloadModelT = type[_PayloadT]


class PayloadContainer(Protocol[_PayloadT]):
    """Object with a typed ``payload`` attribute."""

    payload: _PayloadT


class CatalogRobot(PayloadContainer[_PayloadT], Protocol[_PayloadT]):
    """Typed robot descriptor passed to catalog robot builders.

    Studio builds the concrete model dynamically per registered robot type, so
    builders must not narrow on a hand-written model class — match on
    ``payload`` instead.
    """

    type: str


_RobotT = TypeVar("_RobotT", bound=CatalogRobot[Any])
_FactoryT = TypeVar("_FactoryT", bound=CatalogRobotFactory)


BuildRobotCallable = Callable[[_RobotT, _FactoryT], Awaitable[PhysicalAIRobot]]
"""Builds a physicalai driver for one robot type.

Return the plain driver (for example ``physicalai.robot.SO101``), not a
``SharedRobot``: Studio wraps it so a single owner process holds the hardware.

The driver's class must therefore be decorated with
``physicalai.config.export_config``, because Studio exports it with
``to_config`` to hand the owner a construction recipe. An undecorated class
fails the build with ``ConfigError``.
"""


@dataclass
class RobotCatalogDefinition(Generic[_PayloadT]):
    """Complete definition of a plugin robot type for Studio registration."""

    type: str
    display_name: str
    role: Literal["follower", "leader"]
    robot_builder: BuildRobotCallable | None = None
    robot_payload: _PayloadModelT | None = None
    asset: RobotAsset | None = None

    adapter_options: RobotAdapterOptions = field(default_factory=RobotAdapterOptions)
    probe: RobotProbe[_PayloadT] | None = None


class RobotCatalogRegistry(Protocol):
    """Registry interface used by plugin entry points to add robot definitions."""

    def register_robot(self, definition: RobotCatalogDefinition) -> None:
        """Register one robot catalog definition with Studio."""
        ...
