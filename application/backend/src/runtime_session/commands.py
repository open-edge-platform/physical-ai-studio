"""Forward commands to an existing session, shared by HTTP and WebSocket clients."""

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from runtime_session.bus import Bus, BusRequestError, NoResponderError, command_key
from schemas.runtime_session import AckMessage, SessionError

_COMMAND_TIMEOUT_S = 600.0
"""A backstop only: every handler bounds its own work (a dataset copy, a video encode)."""


class _CommandHeader(BaseModel):
    command: str = Field(pattern=r"^[a-z_]+$")
    request_id: str


async def request_command(bus: Bus, raw: str) -> AckMessage:
    """Validate the envelope and preserve the command handler's acknowledgement and errors."""
    try:
        header = _CommandHeader.model_validate_json(raw)
    except ValidationError as error:
        return AckMessage(request_id="", ok=False, error=SessionError(message=str(error), error_code="invalid_command"))

    def failed(message: str, error_code: str) -> AckMessage:
        return AckMessage(
            request_id=header.request_id,
            ok=False,
            error=SessionError(message=message, error_code=error_code),
        )

    try:
        result: dict[str, Any] = await bus.request(
            command_key(header.command), raw, dict[str, Any], timeout=_COMMAND_TIMEOUT_S
        )
    except BusRequestError as error:
        return AckMessage(
            request_id=header.request_id,
            ok=False,
            error=SessionError(message=error.message, error_code=error.error_code, **error.data),
        )
    except NoResponderError:
        return failed(f"Nothing can {header.command} right now; load what it needs first.", "command_unavailable")
    except TimeoutError:
        return failed(f"{header.command} did not finish in time", "command_timeout")
    return AckMessage(request_id=header.request_id, ok=True, result=result or None)
