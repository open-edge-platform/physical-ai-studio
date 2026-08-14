from __future__ import annotations

import asyncio
import json
import os
import queue
import socket
import time
from typing import Any
from uuid import UUID

import pytest
from physicalai.config import Config

from api.robot_control import handle_outgoing
from exceptions import BaseException as AppBaseException
from exceptions import RobotDeviceAlreadyOwnedError
from runtime.contract import DisconnectCommand, SetFollowerSourceCommand, StateEvent
from runtime.hosts.process_host import RuntimeProcessHost
from runtime.transport.client import RuntimeProcessError, RuntimeSessionClient
from runtime.transport.ids import derive_endpoint_port, runtime_session_name
from tests.runtime.test_session import _document, _document_with_leader


def _start_host(
    name: str,
    document: dict[str, Any],
) -> tuple[RuntimeProcessHost, RuntimeSessionClient]:
    client = RuntimeSessionClient(name)
    client.open()
    host = RuntimeProcessHost(name, document)
    host.start()
    try:
        client.connect(timeout=5, process=host)
        client.wait_until_ready(host, timeout=5)
    except Exception:
        host.stop()
        client.close()
        raise
    return host, client


def _stop_host(host: RuntimeProcessHost, client: RuntimeSessionClient) -> None:
    client.apply(DisconnectCommand())
    host.join(timeout=3)
    if host.is_alive():
        host.stop()
    client.close()


def test_process_host_runs_session_and_streams_connected_state() -> None:
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
            asyncio.run(asyncio.wait_for(handle_outgoing(_FakeWebSocket(), client, _HostAsOwner(host)), timeout=2))
    finally:
        _stop_host(host, client)


def test_process_host_propagates_setup_error() -> None:
    name = runtime_session_name(UUID("f7451520-6f14-4938-982c-d54c12e816dd"))
    client = RuntimeSessionClient(name)
    client.open()
    host = RuntimeProcessHost(name, _document(connect_error="connect failed"))
    host.start()
    try:
        # connect_error is raised from session.run(), after READY, so this
        # covers the Zenoh fatal-error path rather than the stdout handshake.
        client.connect(timeout=5, process=host)

        with pytest.raises(RuntimeProcessError, match="connect failed"):
            client.wait_until_ready(host, timeout=5)
    finally:
        host.stop()
        client.close()


def test_listener_bind_failure_reaches_parent_as_robot_ownership_error() -> None:
    name = runtime_session_name(UUID("b927ab53-13e9-4787-8fb7-fc4d837c9fb3"))
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", derive_endpoint_port(name)))
    blocker.listen()
    client = RuntimeSessionClient(name)
    client.open()
    host = RuntimeProcessHost(name, _document())
    try:
        # server.open() fails before any Zenoh publisher exists, so this error
        # can only reach the parent through the worker's stdout handshake.
        with pytest.raises(AppBaseException) as exc_info:
            host.start()
        assert exc_info.value.error_code == RobotDeviceAlreadyOwnedError().error_code
        assert "already in use" in exc_info.value.message.lower()
    finally:
        host.stop()
        client.close()
        blocker.close()


@pytest.mark.integration
def test_runtime_session_and_own_shared_robot_use_different_listeners() -> None:
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


def test_spawn_payload_survives_json() -> None:
    payload = {
        "session_name": runtime_session_name(UUID("f6838e85-5d48-4a2f-91b2-0cfaf2829f01")),
        "document": _document_with_leader(),
        "follower_name": "follower",
        "leader_name": "leader",
        "idle_timeout_s": 45.0,
    }

    assert json.loads(json.dumps(payload)) == payload


def test_child_runs_in_its_own_process_group() -> None:
    name = runtime_session_name(UUID("68181d7e-2728-4493-9dd3-2472f3a2a88f"))
    host, client = _start_host(name, _document())
    try:
        assert host.pid is not None
        assert os.getpgid(host.pid) != os.getpgid(0)
    finally:
        _stop_host(host, client)


def test_stop_before_start_terminates_the_spawned_worker() -> None:
    name = runtime_session_name(UUID("15e54f25-b877-412a-9a40-8efc3f932b82"))
    host = RuntimeProcessHost(name, _document())

    host.stop()
    host.start()
    host.join(timeout=2)

    assert not host.is_alive()


class _HostAsOwner:
    def __init__(self, host: RuntimeProcessHost) -> None:
        self._host = host

    def is_alive(self) -> bool:
        return self._host.is_alive()

    def exited_cleanly(self) -> bool:
        return self._host.exited_cleanly

    @property
    def error(self) -> AppBaseException | None:
        return self._host.error


class _FakeWebSocket:
    async def send_json(self, payload: dict[str, Any]) -> None:
        pass
