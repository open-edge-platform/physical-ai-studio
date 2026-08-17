import asyncio
from unittest.mock import MagicMock

from api.record import handle_incoming


class FakeWebSocket:
    def __init__(self, messages: list[dict]) -> None:
        self._messages = messages
        self.sent: list[dict] = []

    async def receive_json(self, _: str) -> dict:
        if not self._messages:
            raise RuntimeError("No more messages")
        return self._messages.pop(0)

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def test_record_path_refuses_a_held_follower(test_environment, monkeypatch) -> None:
    process = MagicMock()
    monkeypatch.setattr(
        "api.record.runtime_session_holder",
        lambda follower_id, timeout=1.0: {"pid": 41273},
    )
    websocket = FakeWebSocket(
        [
            {
                "event": "load_environment",
                "data": {"environment": test_environment.model_dump(mode="json")},
            },
            {"event": "disconnect", "data": {}},
        ]
    )

    asyncio.run(handle_incoming(websocket, process, set()))

    process.load_environment.assert_not_called()
    assert websocket.sent == [
        {
            "event": "error",
            "message": (
                "Robot 'Khaos' is already in use by a running session (pid 41273). "
                "Stop that session, or wait for it to disconnect, then try again."
            ),
            "error_code": "runtime_session_busy",
        }
    ]
    process.disconnect.assert_called_once()
