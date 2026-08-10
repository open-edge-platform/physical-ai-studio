import asyncio
from unittest.mock import MagicMock, call

from fastapi.websockets import WebSocketDisconnect

from api.robot_control import _websocket_error_payload, handle_incoming
from exceptions import RobotDeviceAlreadyOwnedError
from runtime.contract import DisconnectCommand, SetFollowerSourceCommand


def test_websocket_error_payload_from_app_exception():
    payload = _websocket_error_payload(RobotDeviceAlreadyOwnedError())
    assert payload["event"] == "error"
    assert payload["error_code"] == "robot_device_already_owned"
    assert "already in use" in payload["message"].lower()


def test_websocket_error_payload_from_generic_exception():
    payload = _websocket_error_payload(RuntimeError("boom"))
    assert payload["event"] == "error"
    assert payload["error_code"] == "robot_connection_failed"
    assert payload["message"] == "boom"


class FakeWebSocket:
    def __init__(self, messages: list[dict]) -> None:
        self._messages = messages

    async def receive_json(self, _: str) -> dict:
        if not self._messages:
            raise WebSocketDisconnect
        return self._messages.pop(0)


def test_handle_incoming_applies_a_valid_set_follower_source_command() -> None:
    session = MagicMock()
    websocket = FakeWebSocket(
        [
            {"event": "set_follower_source", "data": {"follower_source": "teleop"}},
        ]
    )

    asyncio.run(handle_incoming(websocket, session))

    session.apply.assert_any_call(SetFollowerSourceCommand(follower_source="teleop"))


def test_handle_incoming_drops_malformed_follower_source_without_crashing() -> None:
    """A malformed command must not tear down the session or the websocket task.

    Regression test: SetFollowerSourceCommand.model_validate() previously raised
    a pydantic.ValidationError that was not caught here, propagating out of the
    task and closing an otherwise-healthy session.
    """
    session = MagicMock()
    websocket = FakeWebSocket(
        [
            {"event": "set_follower_source", "data": {"follower_source": "not_a_real_mode"}},
            {"event": "set_follower_source", "data": {}},
            {"event": "set_follower_source", "data": {"follower_source": "hold"}},
        ]
    )

    asyncio.run(handle_incoming(websocket, session))

    # Only the valid, later command was applied (the malformed ones were dropped),
    # followed by the disconnect once the fake websocket runs out of messages.
    assert session.apply.call_args_list == [
        call(SetFollowerSourceCommand(follower_source="hold")),
        call(DisconnectCommand()),
    ]


def test_handle_incoming_applies_disconnect_on_websocket_disconnect() -> None:
    session = MagicMock()
    websocket = FakeWebSocket([])

    asyncio.run(handle_incoming(websocket, session))

    session.apply.assert_called_once_with(DisconnectCommand())
