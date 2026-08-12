from __future__ import annotations

from typing import Any, cast

import msgpack

from runtime.contract import AckEvent, Command, CommandAdapter, RuntimeEvent, RuntimeEventAdapter

_MAX_PAYLOAD_BYTES = 1024 * 1024
_FATAL_FIELD = "_fatal"


def _pack(payload: dict[str, Any]) -> bytes:
    return cast("bytes", msgpack.packb(payload, use_bin_type=True))


def _unpack(data: bytes) -> dict[str, Any]:
    if len(data) > _MAX_PAYLOAD_BYTES:
        raise ValueError(f"Runtime transport payload exceeds {_MAX_PAYLOAD_BYTES} bytes")
    payload = msgpack.unpackb(data, raw=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping payload, got {type(payload).__name__}")
    return payload


def encode_command(command: Command, *, instance_id: str | None = None) -> bytes:
    """Encode one validated runtime command."""
    return _pack(
        {
            "instance_id": instance_id,
            "command": command.model_dump(mode="json"),
        }
    )


def decode_command(data: bytes) -> tuple[Command, str | None]:
    """Decode and validate one runtime command."""
    payload = _unpack(data)
    instance_id = payload.get("instance_id")
    if instance_id is not None and not isinstance(instance_id, str):
        raise TypeError("Runtime command instance_id must be a string")
    return CommandAdapter.validate_python(payload["command"]), instance_id


def encode_event(
    event: RuntimeEvent,
    *,
    fatal: bool = False,
    instance_id: str | None = None,
) -> bytes:
    """Encode one runtime event and its internal fatal marker."""
    return _pack(
        {
            "instance_id": instance_id,
            "event": event.model_dump(mode="json"),
            _FATAL_FIELD: fatal,
        }
    )


def decode_event(data: bytes) -> tuple[RuntimeEvent, bool, str | None]:
    """Decode one runtime event and its internal fatal marker."""
    payload = _unpack(data)
    instance_id = payload.get("instance_id")
    if instance_id is not None and not isinstance(instance_id, str):
        raise TypeError("Runtime event instance_id must be a string")
    fatal = payload.get(_FATAL_FIELD, False) is True
    return RuntimeEventAdapter.validate_python(payload["event"]), fatal, instance_id


def encode_metadata(metadata: dict[str, Any]) -> bytes:
    """Encode session metadata."""
    return _pack(metadata)


def decode_metadata(data: bytes) -> dict[str, Any]:
    """Decode session metadata."""
    return _unpack(data)


def decode_ack(data: bytes) -> AckEvent:
    """Decode a request reply and require an acknowledgement event."""
    event, _, _ = decode_event(data)
    if not isinstance(event, AckEvent):
        raise TypeError(f"Expected an ack reply, got {event.event}")
    return event
