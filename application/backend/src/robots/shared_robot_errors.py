"""Map physicalai.robot SharedRobot errors to application BaseException types."""

from __future__ import annotations

from loguru import logger
from physicalai.robot import (
    RobotCalibrationError,
    RobotDeviceAlreadyOwned,
    RobotError,
    RobotNameConflict,
    RobotProtocolMismatch,
    RobotTransportError,
)

from exceptions import BaseException as AppBaseException
from exceptions import (
    RobotCalibrationRequiredError,
    RobotDeviceAlreadyOwnedError,
    RobotNameConflictError,
    RobotProtocolMismatchError,
    SharedRobotTransportError,
)


def translate_robot_error(
    exc: BaseException,
    *,
    robot_name: str | None = None,
) -> BaseException:
    """Map physicalai.robot errors to app BaseException at the integration boundary.

    User-facing copy lives on the app exceptions. Only structured fields are used:
    ``RobotDeviceAlreadyOwned.device_ids`` from physicalai, and ``robot_name`` from
    the Studio connect context. Library messages are never parsed or forwarded.
    """
    if isinstance(exc, AppBaseException) or not isinstance(exc, RobotError):
        return exc

    if isinstance(exc, RobotCalibrationError):
        return RobotCalibrationRequiredError(reason=exc.reason, calibration_path=exc.calibration_path)

    if isinstance(exc, RobotDeviceAlreadyOwned):
        device_ids = tuple(exc.device_ids or ())
        return RobotDeviceAlreadyOwnedError(device_ids=device_ids or None)

    if isinstance(exc, RobotNameConflict):
        return RobotNameConflictError(robot_name=robot_name)

    if isinstance(exc, RobotProtocolMismatch):
        # physicalai.RobotProtocolMismatch does not expose supported/remote
        # protocol versions as structured fields — only in the message string.
        return RobotProtocolMismatchError()

    if isinstance(exc, RobotTransportError):
        logger.warning("Unmapped SharedRobot transport failure (phase={!r}): {}", exc.phase, exc)
    else:
        logger.warning("Unmapped robot error: {}", exc)

    return SharedRobotTransportError()
