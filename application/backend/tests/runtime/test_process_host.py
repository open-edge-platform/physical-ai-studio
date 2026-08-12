from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import socket
import time
from typing import Any
from uuid import UUID

import pytest
from physicalai.config import Config

from api.robot_control import handle_outgoing
from exceptions import RobotDeviceAlreadyOwnedError
from runtime.contract import DisconnectCommand, SetFollowerSourceCommand, StateEvent
from runtime.hosts.process_host import RuntimeProcessHost
from runtime.transport.client import RuntimeProcessError, RuntimeSessionClient
from runtime.transport.ids import derive_endpoint_port, runtime_session_name
from tests.runtime.test_session import _document, _document_with_leader
from utils.multiprocessing import ensure_spawn_start_method


def _start_host(
    name: str,
    document: dict[str, Any],
) -> tuple[RuntimeProcessHost, RuntimeSessionClient]:
    client = RuntimeSessionClient(name)
    client.open()
    host = RuntimeProcessHost(name, document, stop_event=mp.Event())
    host.start()
    try:
        client.connect(timeout=5, process=host)
        client.wait_until_ready(host, timeout=5)
    except Exception:
        host.stop()
        client.close()
        raise
    return host, client


def _start_owned_host(
    name: str,
    document: dict[str, Any],
    instance_id: str,
) -> tuple[RuntimeProcessHost, RuntimeSessionClient]:
    client = RuntimeSessionClient(name, instance_id=instance_id)
    client.open()
    host = RuntimeProcessHost(name, document, stop_event=mp.Event(), instance_id=instance_id)
    host.start()
    return host, client


def _stop_host(host: RuntimeProcessHost, client: RuntimeSessionClient) -> None:
    client.apply(DisconnectCommand())
    host.join(timeout=3)
    if host.is_alive():
        host.stop()
    client.close()


def test_process_host_runs_session_and_streams_connected_state() -> None:
    ensure_spawn_start_method()
    name = runtime_session_name(UUID("73caa570-b399-4ce3-b54f-dc96a4275534"))
    host, client = _start_host(name, _document())
    try:
        events = []
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                events.append(client.get_nowait())
            except queue.Empty:
                time.sleep(0.01)
            if any(isinstance(event, StateEvent) for event in events):
                break

        assert any(
            isinstance(event, StateEvent) and event.data.connected and event.data.follower_source == "hold"
            for event in events
        )
        assert host.is_alive()
    finally:
        _stop_host(host, client)


def test_mode_command_crosses_process_boundary() -> None:
    ensure_spawn_start_method()
    name = runtime_session_name(UUID("c6ffef31-0e82-4da4-8b19-cfc4be33e097"))
    host, client = _start_host(name, _document_with_leader())
    try:
        while True:
            try:
                client.get_nowait()
            except queue.Empty:
                break
        client.apply(SetFollowerSourceCommand(follower_source="teleop"))

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                event = client.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue
            if isinstance(event, StateEvent) and event.data.follower_source == "teleop":
                break
        else:
            pytest.fail("Teleop state did not cross the process boundary")
    finally:
        _stop_host(host, client)


def test_killed_process_is_observable_without_waiting_indefinitely() -> None:
    ensure_spawn_start_method()
    name = runtime_session_name(UUID("d824be11-ffb4-4f72-afd6-d593e457950a"))
    host, client = _start_host(name, _document())
    try:
        host.kill()
        host.join(timeout=2)

        deadline = time.monotonic() + 1
        while host.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not host.is_alive()
        assert not client.shutdown_received
        with pytest.raises(RuntimeProcessError, match="stopped unexpectedly"):
            asyncio.run(asyncio.wait_for(handle_outgoing(_FakeWebSocket(), client, host), timeout=2))
    finally:
        _stop_host(host, client)


def test_process_host_propagates_setup_error() -> None:
    ensure_spawn_start_method()
    name = runtime_session_name(UUID("f7451520-6f14-4938-982c-d54c12e816dd"))
    client = RuntimeSessionClient(name)
    client.open()
    host = RuntimeProcessHost(name, _document(connect_error="connect failed"), stop_event=mp.Event())
    host.start()
    try:
        client.connect(timeout=5, process=host)

        with pytest.raises(RuntimeProcessError, match="connect failed"):
            client.wait_until_ready(host, timeout=5)
    finally:
        host.stop()
        client.close()


def test_second_spawn_does_not_attach_to_or_disconnect_existing_session() -> None:
    ensure_spawn_start_method()
    name = runtime_session_name(UUID("233f3b26-e411-4772-94aa-8580f7d44de2"))
    first_host, first_client = _start_owned_host(name, _document(), "first-instance")
    try:
        first_client.connect(timeout=5, process=first_host)
        first_client.wait_until_ready(first_host, timeout=5)
        second_host, second_client = _start_owned_host(name, _document(), "second-instance")
        try:
            with pytest.raises(RuntimeProcessError):
                second_client.connect(timeout=5, process=second_host)
        finally:
            second_host.stop()
            second_client.close()

        assert first_host.is_alive()
        assert first_host.error is None
    finally:
        _stop_host(first_host, first_client)


def test_listener_bind_failure_reaches_parent_as_robot_ownership_error() -> None:
    ensure_spawn_start_method()
    name = runtime_session_name(UUID("b927ab53-13e9-4787-8fb7-fc4d837c9fb3"))
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", derive_endpoint_port(name)))
    blocker.listen()
    host, client = _start_owned_host(name, _document(), "blocked-instance")
    try:
        with pytest.raises(RuntimeProcessError) as exc_info:
            client.connect(timeout=5, process=host)
        assert exc_info.value.error_code == RobotDeviceAlreadyOwnedError().error_code
        assert "already in use" in exc_info.value.message.lower()
    finally:
        host.stop()
        client.close()
        blocker.close()


@pytest.mark.integration
def test_runtime_session_and_own_shared_robot_use_different_listeners() -> None:
    ensure_spawn_start_method()
    follower_id = UUID("c3f3f886-8813-4b3b-ba48-165cdaa39995")
    robot_name = str(follower_id)
    name = runtime_session_name(follower_id)
    robot_config = Config(
        "physicalai.robot.SharedRobot",
        {
            "name": robot_name,
            "robot": {
                "class_path": "tests.runtime.fakes.FakeRobot",
                "init_args": {"positions": [[0.0]], "joint_names": ["joint"]},
            },
            "idle_timeout": 0.5,
        },
    ).to_dict()
    document = Config(
        "physicalai.runtime.RobotRuntime",
        {"robot": robot_config, "cameras": {}, "fps": 10.0},
    ).to_dict()

    host, client = _start_host(name, document)
    try:
        assert host.is_alive()
        assert derive_endpoint_port(name) == 17885
        assert derive_endpoint_port(name) != 46018
    finally:
        _stop_host(host, client)


class _FakeWebSocket:
    async def send_json(self, payload: dict[str, Any]) -> None:
        pass
