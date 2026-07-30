"""Factory protocol exposed to plugin robot-builder callables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .schemas import SerialPortInfo


class CatalogRobotFactory(Protocol):
    """Factory protocol provided by Studio to robot builders.

    Plugin ``robot_builder`` callables receive an instance of this protocol
    so they can resolve serial ports at build time.
    """

    async def find_port(self, port_info: SerialPortInfo) -> str | None:
        """Return the resolved connection port, if present."""
