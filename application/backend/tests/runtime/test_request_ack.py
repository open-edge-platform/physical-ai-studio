from __future__ import annotations

from uuid import UUID

import pytest

from runtime.contract import SaveEpisodeCommand
from runtime.transport.client import RuntimeSessionClient
from runtime.transport.ids import runtime_session_name
from runtime.transport.server import RuntimeZenohServer


def test_a_request_receives_an_ack_carrying_its_request_id() -> None:
    name = runtime_session_name(UUID("46770884-694f-453b-9a8f-a3d04b1ff974"))
    received: list = []
    server = RuntimeZenohServer(name, instance_id="server")
    client = RuntimeSessionClient(name)
    client.open()
    try:
        server.open(received.append)
        client.connect(timeout=3)

        ack = client.request(SaveEpisodeCommand(request_id="request-7"))

        assert ack.data.request_id == "request-7"
        assert ack.data.ok
        assert ack.data.error is None
        assert received == [SaveEpisodeCommand(request_id="request-7")]
    finally:
        server.close()
        client.close()


def test_a_failing_handler_replies_ok_false_with_the_message() -> None:
    name = runtime_session_name(UUID("8c1d2e3f-4051-6273-8495-a6b7c8d9e0f1"))
    server = RuntimeZenohServer(name, instance_id="server")
    client = RuntimeSessionClient(name)
    client.open()

    def fail(command: object) -> None:
        raise RuntimeError("episode folder is locked")

    try:
        server.open(lambda command: None, request_handler=fail)
        client.connect(timeout=3)

        ack = client.request(SaveEpisodeCommand(request_id="request-8"))

        assert ack.data.request_id == "request-8"
        assert not ack.data.ok
        assert ack.data.error == "episode folder is locked"
    finally:
        server.close()
        client.close()


def test_a_request_for_a_different_instance_is_rejected() -> None:
    name = runtime_session_name(UUID("9d0e1f20-3142-5364-7586-97a8b9c0d1e2"))
    server = RuntimeZenohServer(name, instance_id="live")
    client = RuntimeSessionClient(name, instance_id="stale")
    client.open()
    try:
        server.open(lambda command: None)
        client._metadata_ready.set()
        client._instance_id = "stale"

        with pytest.raises(TimeoutError):
            client.request(SaveEpisodeCommand(request_id="request-9"), timeout=1.0)
    finally:
        server.close()
        client.close()
