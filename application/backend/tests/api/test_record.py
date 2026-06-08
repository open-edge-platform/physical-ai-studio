import asyncio
from unittest.mock import MagicMock

from api.record import handle_incoming
from schemas import InferenceBackend, InferenceDevice, Model


class FakeWebSocket:
    def __init__(self, messages: list[dict]) -> None:
        self._messages = messages

    async def receive_json(self, _: str) -> dict:
        if not self._messages:
            raise RuntimeError("No more messages")
        return self._messages.pop(0)


def test_handle_incoming_load_model_requires_inference_device(test_model) -> None:
    process = MagicMock()
    websocket = FakeWebSocket(
        [
            {
                "event": "load_model",
                "data": {
                    "model": test_model.model_dump(mode="json"),
                    "inference_device": {"backend": "openvino", "device": "GPU"},
                },
            },
            {"event": "disconnect", "data": {}},
        ]
    )

    asyncio.run(handle_incoming(websocket, process, set()))

    process.load_model.assert_called_once_with(
        Model.model_validate(test_model.model_dump(mode="json")),
        InferenceDevice(backend=InferenceBackend.OPENVINO, device="GPU"),
    )
    process.disconnect.assert_called_once()
