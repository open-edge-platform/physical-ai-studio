import asyncio
from unittest.mock import AsyncMock, Mock

from fastapi.websockets import WebSocketDisconnect

from api.robot_control import _websocket_error_payload, handle_incoming
from exceptions import RobotDeviceAlreadyOwnedError


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


def test_handle_incoming_keeps_session_alive_for_message_without_event():
    websocket = AsyncMock()
    websocket.receive_json.side_effect = [{"follower_id": "not-a-command"}, WebSocketDisconnect]
    worker = Mock()
    worker.should_stop.side_effect = [False, False]

    asyncio.run(handle_incoming(websocket, worker))

    websocket.send_json.assert_awaited_once_with(
        {
            "event": "error",
            "message": "Robot control message is missing an event.",
            "error_code": "invalid_message",
        }
    )
    worker.set_action_read_state.assert_not_called()


def test_handle_incoming_sets_follower_source_for_valid_command():
    websocket = AsyncMock()
    websocket.receive_json.side_effect = [
        {"event": "set_follower_source", "data": 1},
        WebSocketDisconnect,
    ]
    worker = Mock()
    worker.should_stop.side_effect = [False, False]
    worker.loaded_event.is_set.return_value = True
    worker.get_action_read_state.return_value = 1

    asyncio.run(handle_incoming(websocket, worker))

    worker.set_action_read_state.assert_called_once_with(1)
    websocket.send_json.assert_awaited_once_with({"event": "state", "data": {"connected": True, "follower_source": 1}})
