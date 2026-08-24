from __future__ import annotations

import hashlib
import queue
import threading
import time
from uuid import UUID

import pytest
import zenoh

from runtime.contract import Command, SaveEpisodeCommand, SetFollowerSourceCommand, StateData, StateEvent
from runtime.transport.client import RuntimeProcessError, RuntimeSessionClient
from runtime.transport.codec import encode_event
from runtime.transport.ids import (
    command_key,
    derive_endpoint_port,
    error_key,
    lifecycle_key,
    metadata_key,
    request_key,
    runtime_session_name,
    state_key,
    tick_key,
)
from runtime.transport.server import RuntimeZenohServer
from runtime.transport.session import build_session_config


class _FakeEndpoint:
    def __init__(self) -> None:
        self.matching_status = type("MatchingStatus", (), {"matching": True})()

    def undeclare(self) -> None:
        pass

    def try_recv(self) -> None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.publishers: dict[str, dict] = {}
        self.subscribers: dict[str, object] = {}
        self.queryables: list[str] = []

    def declare_publisher(self, key: str, **kwargs: object) -> _FakeEndpoint:
        self.publishers[key] = kwargs
        return _FakeEndpoint()

    def declare_subscriber(self, key: str, handler: object) -> _FakeEndpoint:
        self.subscribers[key] = handler
        return _FakeEndpoint()

    def declare_queryable(self, key: str, handler: object) -> _FakeEndpoint:
        self.queryables.append(key)
        return _FakeEndpoint()

    def close(self) -> None:
        pass


def test_runtime_topic_scheme_uses_rt_prefixed_follower_id() -> None:
    name = runtime_session_name(UUID("c3f3f886-8813-4b3b-ba48-165cdaa39995"))

    assert name == "rt-c3f3f886-8813-4b3b-ba48-165cdaa39995"
    assert metadata_key(name) == f"studio/rt/{name}/metadata"
    assert command_key(name) == f"studio/rt/{name}/command"
    assert request_key(name) == f"studio/rt/{name}/request"
    assert tick_key(name) == f"studio/rt/{name}/tick"
    assert state_key(name) == f"studio/rt/{name}/state"
    assert error_key(name) == f"studio/rt/{name}/error"
    assert lifecycle_key(name) == f"studio/rt/{name}/lifecycle"


@pytest.mark.parametrize(
    ("follower_id", "expected_port"),
    [
        ("00000000-0000-0000-0000-00000000e2e7", 17979),
        ("c3f3f886-8813-4b3b-ba48-165cdaa39995", 17885),
    ],
)
def test_endpoint_derivation_uses_studio_only_range(follower_id: str, expected_port: int) -> None:
    name = runtime_session_name(UUID(follower_id))

    assert derive_endpoint_port(name) == expected_port
    assert 10000 <= derive_endpoint_port(name) <= 19999


def test_session_port_cannot_collide_with_follower_or_leader_robot_ports() -> None:
    follower_id = "c3f3f886-8813-4b3b-ba48-165cdaa39995"
    # This leader hashes to the session's old shared-range port, 47885.
    leader_id = "00000000-0000-0000-0000-0000000078d7"

    def robot_port(robot_id: str) -> int:
        digest = hashlib.sha256(f"physicalai/robot/{robot_id}".encode()).digest()
        return 20000 + int.from_bytes(digest[:4], "big") % 40000

    session_port = derive_endpoint_port(runtime_session_name(follower_id))

    assert session_port == 17885
    assert robot_port(follower_id) == 46018
    assert robot_port(leader_id) == 47885
    assert session_port not in {robot_port(follower_id), robot_port(leader_id)}


def test_metadata_exposes_public_instance_id_without_token_or_secret_fields() -> None:
    name = runtime_session_name(UUID("f517da57-708a-4cbd-ae73-2a28877d2656"))
    server = RuntimeZenohServer(name, instance_id="public-generation")
    client = RuntimeSessionClient(name)
    client.open()
    try:
        server.open(lambda command: None)

        metadata = client.connect(timeout=3)

        assert metadata["instance_id"] == "public-generation"
        assert client._instance_id == "public-generation"
        assert not {"owner_token", "token", "secret"}.intersection(metadata)
    finally:
        server.close()
        client.close()


@pytest.mark.parametrize("listen", [True, False])
def test_session_config_is_peer_only_without_scouting_and_uses_loopback(listen: bool) -> None:
    name = runtime_session_name(UUID("c3f3f886-8813-4b3b-ba48-165cdaa39995"))
    config = build_session_config(name, listen=listen)

    assert config.get_json("mode") == '"peer"'
    assert config.get_json("scouting/multicast/enabled") == "false"
    assert config.get_json("scouting/gossip/enabled") == "false"
    endpoint_kind = "listen" if listen else "connect"
    assert config.get_json(f"{endpoint_kind}/endpoints") == '["tcp/127.0.0.1:17885"]'


def test_server_declares_required_qos(monkeypatch: pytest.MonkeyPatch) -> None:
    name = runtime_session_name(UUID("be2781f9-b165-4ffc-a78d-f77f258f4235"))
    fake_session = _FakeSession()
    ring_channels: list[int] = []
    monkeypatch.setattr("runtime.transport.server.open_session", lambda name, listen: fake_session)
    original_ring_channel = zenoh.handlers.RingChannel

    def record_ring_channel(capacity: int) -> object:
        ring_channels.append(capacity)
        return original_ring_channel(capacity)

    monkeypatch.setattr(
        zenoh.handlers,
        "RingChannel",
        record_ring_channel,
    )
    server = RuntimeZenohServer(name)
    try:
        server.open(lambda command: None)

        assert ring_channels == [1]
        for key in (tick_key(name), state_key(name), error_key(name)):
            assert fake_session.publishers[key]["reliability"] == zenoh.Reliability.BEST_EFFORT
            assert fake_session.publishers[key]["congestion_control"] == zenoh.CongestionControl.DROP
            assert fake_session.publishers[key]["express"] is True
        lifecycle_qos = fake_session.publishers[lifecycle_key(name)]
        assert lifecycle_qos["reliability"] == zenoh.Reliability.BEST_EFFORT
        assert lifecycle_qos["congestion_control"] == zenoh.CongestionControl.DROP
    finally:
        server.close()


def test_client_declares_command_qos(monkeypatch: pytest.MonkeyPatch) -> None:
    name = runtime_session_name(UUID("96943ea2-e512-49c6-956a-31fcc7d5138c"))
    fake_session = _FakeSession()
    monkeypatch.setattr("runtime.transport.client.open_session", lambda name, listen: fake_session)
    client = RuntimeSessionClient(name)
    try:
        client.open()

        command_qos = fake_session.publishers[command_key(name)]
        assert command_qos["reliability"] == zenoh.Reliability.BEST_EFFORT
        assert command_qos["congestion_control"] == zenoh.CongestionControl.DROP
    finally:
        client.close()


def test_client_buffers_command_until_metadata_answers() -> None:
    name = runtime_session_name(UUID("1dd21ef4-c77f-4d1f-a2fd-ed4622349901"))
    received: list = []
    client = RuntimeSessionClient(name)
    client.open()
    command = SetFollowerSourceCommand(follower_source="teleop")
    client.apply(command)

    connect_error: list[BaseException] = []

    def connect() -> None:
        try:
            client.connect(timeout=3)
        except BaseException as exc:
            connect_error.append(exc)

    connect_thread = threading.Thread(target=connect)
    connect_thread.start()
    time.sleep(0.1)
    assert received == []

    server = RuntimeZenohServer(name)
    try:
        server.open(received.append)
        connect_thread.join(timeout=3)
        deadline = time.monotonic() + 2
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)

        assert connect_error == []
        assert received == [command]
    finally:
        server.close()
        client.close()


def test_client_connect_keeps_a_received_fatal_error_when_the_process_has_exited() -> None:
    name = runtime_session_name(UUID("b65f0a6f-bfc9-4d42-8d4f-59b0f3f62a09"))
    client = RuntimeSessionClient(name)
    client.open()
    client.error = RuntimeProcessError("connect failed")
    dead_process = type("DeadProcess", (), {"is_alive": lambda self: False, "error": None})()
    try:
        with pytest.raises(RuntimeProcessError, match="connect failed"):
            client.connect(process=dead_process)
    finally:
        client.close()


def test_acked_request_returns_correlated_unsupported_reply() -> None:
    name = runtime_session_name(UUID("46770884-694f-453b-9a8f-a3d04b1ff974"))
    server = RuntimeZenohServer(name)
    client = RuntimeSessionClient(name)
    client.open()
    try:
        server.open(lambda command: None)
        client.connect(timeout=3)

        ack = client.request(SaveEpisodeCommand(request_id="request-7"))

        assert ack.data.request_id == "request-7"
        assert not ack.data.ok
        assert ack.data.error == "save_episode is not supported by this runtime session"
    finally:
        server.close()
        client.close()


def test_ready_state_can_be_recovered_from_metadata_when_publication_is_dropped() -> None:
    name = runtime_session_name(UUID("c27a49eb-1b90-4a52-8ec9-e329214233bc"))
    server = RuntimeZenohServer(name)
    client = RuntimeSessionClient(name)
    client.open()
    try:
        server.open(lambda command: None)
        client.connect(timeout=3)
        server.update_metadata(
            status="running",
            state={
                "event": "state",
                "data": {"connected": True, "follower_source": "hold"},
            },
        )

        client.wait_until_ready(type("AliveProcess", (), {"is_alive": lambda self: True, "error": None})(), timeout=2)

        assert client.get_nowait().model_dump() == {
            "event": "state",
            "data": {"connected": True, "follower_source": "hold"},
        }
    finally:
        server.close()
        client.close()


def test_metadata_recovery_and_matching_publication_emit_initial_state_once() -> None:
    client = RuntimeSessionClient(runtime_session_name(UUID("e8ee07c1-1bdd-4130-bc9e-d07f6ea6ad1f")))
    event = StateEvent(data={"connected": True, "follower_source": "hold"})

    client._accept_ready_state(event, initial_only=True)
    client._accept_ready_state(event)

    assert client.get_nowait() == event
    with pytest.raises(queue.Empty):
        client.get_nowait()


def test_command_for_replaced_instance_is_rejected() -> None:
    name = runtime_session_name(UUID("aefc176c-b3b9-482e-b00c-66a77be03658"))
    received: list[Command] = []
    stale_client = RuntimeSessionClient(name, instance_id="old-instance")
    stale_client.open()
    server = RuntimeZenohServer(name, instance_id="new-instance")
    try:
        server.open(received.append)
        stale_client._metadata_ready.set()
        stale_client.apply(SetFollowerSourceCommand(follower_source="teleop"))
        time.sleep(0.1)

        assert received == []
    finally:
        server.close()
        stale_client.close()


def test_event_from_replaced_instance_is_rejected() -> None:
    client = RuntimeSessionClient(
        runtime_session_name(UUID("c66b74d9-cea7-4ac0-94f8-39f86898589a")),
        instance_id="old-instance",
    )
    sample = type(
        "Sample",
        (),
        {
            "payload": type(
                "Payload",
                (),
                {
                    "to_bytes": lambda self: encode_event(
                        StateEvent(data=StateData(connected=True, follower_source="hold")),
                        instance_id="new-instance",
                    )
                },
            )()
        },
    )()

    client._receive_event(sample)

    with pytest.raises(queue.Empty):
        client.get_nowait()


def test_event_before_metadata_adoption_is_accepted() -> None:
    client = RuntimeSessionClient(runtime_session_name(UUID("77df4e92-bbd8-46c8-9ca1-640ce0fe79f4")))
    event = StateEvent(data=StateData(connected=True, follower_source="hold"))
    sample = type(
        "Sample",
        (),
        {
            "payload": type(
                "Payload",
                (),
                {"to_bytes": lambda self: encode_event(event, instance_id="new-instance")},
            )()
        },
    )()

    client._receive_event(sample)

    assert client.get_nowait() == event
