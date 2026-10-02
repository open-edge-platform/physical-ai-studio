import asyncio
from unittest.mock import MagicMock, call

from fastapi.websockets import WebSocketDisconnect

from api.runtime_ws import _websocket_error_payload, handle_incoming, start_runtime_session
from exceptions import RobotDeviceAlreadyOwnedError
from runtime.contract import (
    LoadDatasetCommand,
    LoadModelCommand,
    SaveEpisodeCommand,
    SetFollowerSourceCommand,
    StartRecordingCommand,
    StartTaskCommand,
)


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
    session.apply.assert_called_once()


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

    assert session.apply.call_args_list == [
        call(SetFollowerSourceCommand(follower_source="hold")),
    ]


def test_handle_incoming_returns_on_websocket_close() -> None:
    session = MagicMock()
    websocket = FakeWebSocket([])

    asyncio.run(handle_incoming(websocket, session))

    session.apply.assert_not_called()


def test_handle_incoming_returns_on_an_explicit_disconnect() -> None:
    """Returning ends the websocket handler, which stops the session."""
    session = MagicMock()
    websocket = FakeWebSocket(
        [
            {"event": "disconnect"},
            {"event": "set_follower_source", "data": {"follower_source": "teleop"}},
        ]
    )

    asyncio.run(handle_incoming(websocket, session))

    session.apply.assert_not_called()


def test_handle_incoming_applies_load_model_and_start_task() -> None:
    session = MagicMock()
    model_id = "c3f3f886-8813-4b3b-ba48-165cdaa39995"
    websocket = FakeWebSocket(
        [
            {
                "event": "load_model",
                "request_id": "req-1",
                "data": {
                    "model_id": model_id,
                    "inference_device": {"backend": "torch", "device": "cpu"},
                },
            },
            {"event": "start_task", "data": {"task": "pick up the cube"}},
        ]
    )

    asyncio.run(handle_incoming(websocket, session))

    assert session.apply.call_count == 2
    load = session.apply.call_args_list[0].args[0]
    start = session.apply.call_args_list[1].args[0]
    assert isinstance(load, LoadModelCommand)
    assert str(load.model_id) == model_id
    assert load.request_id == "req-1"
    assert isinstance(start, StartTaskCommand)
    assert start.task == "pick up the cube"


def test_handle_incoming_applies_load_dataset_and_start_recording() -> None:
    session = MagicMock()
    dataset_id = "a3f3f886-8813-4b3b-ba48-165cdaa39995"
    websocket = FakeWebSocket(
        [
            {"event": "load_dataset", "data": {"dataset_id": dataset_id}},
            {"event": "start_recording", "data": {"task": "pick"}},
        ]
    )

    asyncio.run(handle_incoming(websocket, session))

    load = session.apply.call_args_list[0].args[0]
    start = session.apply.call_args_list[1].args[0]
    assert isinstance(load, LoadDatasetCommand)
    assert str(load.dataset_id) == dataset_id
    assert isinstance(start, StartRecordingCommand)
    assert start.task == "pick"


def test_handle_incoming_forwards_save_episode_as_a_command() -> None:
    """The worker answers with an ack on the event stream."""
    session = MagicMock()
    websocket = FakeWebSocket(
        [{"event": "save_episode", "request_id": "req-9", "data": {}}],
    )

    asyncio.run(handle_incoming(websocket, session))

    session.apply.assert_called_once_with(SaveEpisodeCommand(request_id="req-9"))


def test_start_runtime_session_starts_then_waits_for_readiness() -> None:
    handle = MagicMock()

    asyncio.run(start_runtime_session(handle))

    assert handle.mock_calls == [call.start(), call.wait_until_ready()]
