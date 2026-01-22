"""Calibratable protocol for robot connections.

This module defines the Calibratable protocol that robot connections can
implement to support calibration commands.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Calibratable(Protocol):
    """Protocol for robot connections that support calibration.

    This protocol defines the interface for getting, setting, and reading
    calibration data. Robot connections that support calibration should
    implement these methods.

    The protocol is runtime-checkable, allowing isinstance() checks in the
    command handler to conditionally enable calibration commands.
    """

    async def get_calibration(self) -> dict[str, Any]:
        """Get current calibration from the robot.

        Returns:
            Event dict with calibration data in LeRobot format.
        """
        ...

    async def set_calibration(
        self,
        calibration: dict[str, Any],
        *,
        write_to_motor: bool = False,
    ) -> dict[str, Any]:
        """Set calibration on the robot.

        Args:
            calibration: Calibration data in LeRobot format (dict of motor dicts).
            write_to_motor: If True, also write to motor EEPROM (requires torque off).

        Returns:
            Event dict confirming calibration was set.
        """
        ...

    async def read_motor_calibration(self) -> dict[str, Any]:
        """Read calibration values from motor EEPROM registers.

        This reads the actual values stored in the motor hardware, which may
        differ from the Python-side calibration if they've drifted or weren't
        written to motors.

        Returns:
            Event dict with motor calibration data.
        """
        ...
