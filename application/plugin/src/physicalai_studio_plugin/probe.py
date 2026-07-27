"""Probe and scanner protocols used by plugin robot integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from .schemas import SerialPortInfo

_PayloadT_contra = TypeVar("_PayloadT_contra", bound=BaseModel, contravariant=True)


class PortScanner(Protocol):
    """Duck-type for serial port scanners (e.g. RobotConnectionManager).

    Used by RobotProbe to accept scan results without coupling to a specific
    manager implementation.
    """

    async def find_robots(self) -> None:
        """Refresh discovered robot connections."""

    @property
    def robots(self) -> list[SerialPortInfo]:
        """Return currently discovered robot connections."""


@runtime_checkable
class RobotProbe(Protocol[_PayloadT_contra]):
    """Hardware interaction interface for a robot type.

    Each built-in or plugin-provided robot type implements this protocol
    to encapsulate device discovery, visual identification, and online
    status checking. The type parameter ``_PayloadT`` is the robot's
    payload model (a pydantic ``BaseModel`` subclass).
    """

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        """Discover robots and return current connection metadata."""

    async def identify(
        self,
        payload: _PayloadT_contra,
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None:
        """Trigger physical identification behavior for a robot."""

    async def is_online(
        self,
        payload: _PayloadT_contra,
        manager: PortScanner | None = None,
    ) -> bool:
        """Return whether the robot appears online."""
