"""Shared command handling for WebSocket and ZMQ protocols."""

from http import HTTPStatus
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

from exceptions import GetiBaseException
from robots.robot_client import RobotClient
from workers.robots.calibratable_client import Calibratable


class UnknownCommandError(GetiBaseException):
    """Raised when an unknown command is received.

    Use the static factory method to create instances:
        UnknownCommandError.for_command("invalid_cmd")
    """

    @staticmethod
    def for_command(command: str) -> "UnknownCommandError":
        """Create error for when an unknown command is received."""
        return UnknownCommandError(
            message=f"Unknown command: {command}",
            error_code="unknown_command",
            http_status=HTTPStatus.BAD_REQUEST,
        )


class UnsupportedCommandError(GetiBaseException):
    """Raised when a command is not supported by the robot connection.

    Use the static factory method to create instances:
        UnsupportedCommandError.for_command("get_calibration")
    """

    @staticmethod
    def for_command(command: str) -> "UnsupportedCommandError":
        """Create error for when a command is not supported."""
        return UnsupportedCommandError(
            message=f"Command not supported by this robot: {command}",
            error_code="unsupported_command",
            http_status=HTTPStatus.BAD_REQUEST,
        )


class InvalidPayloadError(GetiBaseException):
    """Raised when a command payload is invalid.

    Use the static factory method to create instances:
        InvalidPayloadError.for_command("set_joints_state", "missing required field")
    """

    @staticmethod
    def for_command(command: str, reason: str) -> "InvalidPayloadError":
        """Create error for invalid payload for a specific command."""
        return InvalidPayloadError(
            message=f"Invalid {command} payload: {reason}",
            error_code="invalid_payload",
            http_status=HTTPStatus.BAD_REQUEST,
        )


# =============================================================================
# Command Models
# =============================================================================


class PingCommand(BaseModel):
    """Ping command - returns pong."""

    command: Literal["ping"] = "ping"


class EnableTorqueCommand(BaseModel):
    """Enable motor torque."""

    command: Literal["enable_torque"] = "enable_torque"


class DisableTorqueCommand(BaseModel):
    """Disable motor torque."""

    command: Literal["disable_torque"] = "disable_torque"


class SetJointsStateCommand(BaseModel):
    """Set joint positions.

    The joints field maps joint names to normalized position values.
    Values should be in [-1.0, 1.0] range for normalized positions.
    """

    command: Literal["set_joints_state"] = "set_joints_state"
    joints: dict[str, float]

    @field_validator("joints")
    @classmethod
    def validate_joints(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("joints dict cannot be empty")
        for name, value in v.items():
            if not isinstance(name, str):
                raise ValueError(f"joint name must be string, got {type(name).__name__}")
            if not isinstance(value, int | float):
                raise ValueError(f"joint '{name}' value must be numeric, got {type(value).__name__}")
        return v


class ReadStateCommand(BaseModel):
    """Read current robot state."""

    command: Literal["read_state"] = "read_state"
    normalize: bool = True


class GetCalibrationCommand(BaseModel):
    """Get current calibration data."""

    command: Literal["get_calibration"] = "get_calibration"


class SetCalibrationCommand(BaseModel):
    """Set calibration data.

    Expects calibration data for all motors.
    """

    command: Literal["set_calibration"] = "set_calibration"
    calibration: dict[str, dict[str, int]]
    write_to_motor: bool = False

    @field_validator("calibration")
    @classmethod
    def validate_calibration(cls, v: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        if not v:
            raise ValueError("calibration dict cannot be empty")

        required_fields = {
            "id",
            "drive_mode",
            "homing_offset",
            "range_min",
            "range_max",
        }
        for motor_name, motor_cal in v.items():
            if not isinstance(motor_name, str):
                raise ValueError(f"motor name must be string, got {type(motor_name).__name__}")
            missing = required_fields - set(motor_cal.keys())
            if missing:
                raise ValueError(f"motor '{motor_name}' missing required fields: {missing}")
        return v


class ReadMotorCalibrationCommand(BaseModel):
    """Read calibration directly from motor registers."""

    command: Literal["read_motor_calibration"] = "read_motor_calibration"


# Discriminated union of all command types
RobotCommand = Annotated[
    PingCommand
    | EnableTorqueCommand
    | DisableTorqueCommand
    | SetJointsStateCommand
    | ReadStateCommand
    | GetCalibrationCommand
    | SetCalibrationCommand
    | ReadMotorCalibrationCommand,
    Field(discriminator="command"),
]


def parse_command(data: dict[str, Any]) -> RobotCommand:
    """Parse and validate a command from raw dict data.

    Args:
        data: Raw dict from client (e.g., from JSON)

    Returns:
        Validated RobotCommand instance

    Raises:
        InvalidPayloadError: If the command is malformed or has invalid payload
        UnknownCommandError: If the command type is not recognized
    """
    # Check if command field exists
    if "command" not in data:
        raise InvalidPayloadError.for_command("unknown", "missing 'command' field")

    command_name = data.get("command", "")

    adapter: TypeAdapter[RobotCommand] = TypeAdapter(RobotCommand)

    try:
        return adapter.validate_python(data)
    except ValidationError as e:
        # Check if this is an unknown command type
        error_str = str(e)
        if "Unable to extract tag" in error_str or "Input tag" in error_str:
            raise UnknownCommandError.for_command(command_name) from e
        # Otherwise it's a payload validation error
        raise InvalidPayloadError.for_command(command_name, str(e)) from e


async def handle_command(  # noqa: PLR0911
    robot_connection: RobotClient,
    cmd: RobotCommand,
) -> dict[str, Any]:
    """Process a command and return the response.

    Args:
        robot_connection: The robot connection to execute commands on.
        cmd: The validated command object.

    Returns:
        Response dict with event type and data.

    Raises:
        UnsupportedCommandError: If the command is not supported by this connection.
    """
    match cmd:
        case PingCommand():
            return await robot_connection.ping()

        # State commands
        case EnableTorqueCommand():
            return await robot_connection.enable_torque()

        case DisableTorqueCommand():
            return await robot_connection.disable_torque()

        case SetJointsStateCommand(joints=joints):
            return await robot_connection.set_joints_state(joints)

        case ReadStateCommand(normalize=normalize):
            return await robot_connection.read_state(normalize=normalize)

        # Calibration commands
        case GetCalibrationCommand():
            if not isinstance(robot_connection, Calibratable):
                raise UnsupportedCommandError.for_command(cmd.command)
            return await robot_connection.get_calibration()

        case SetCalibrationCommand(calibration=calibration, write_to_motor=write_to_motor):
            if not isinstance(robot_connection, Calibratable):
                raise UnsupportedCommandError.for_command(cmd.command)

            return await robot_connection.set_calibration(calibration, write_to_motor=write_to_motor)

        case ReadMotorCalibrationCommand():
            if not isinstance(robot_connection, Calibratable):
                raise UnsupportedCommandError.for_command(cmd.command)

            return await robot_connection.read_motor_calibration()
