from __future__ import annotations

import http
import json
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from exceptions import BaseException as AppBaseException
from exceptions import RuntimeSessionBusyError
from runtime.config_builder import runtime_identity_digest
from runtime.contract import DisconnectCommand, SetFollowerSourceCommand, StateEvent
from runtime.owner import RuntimeSessionOwner, probe_session_metadata, runtime_session_holder, stop_runtime_session
from runtime.transport.client import RuntimeSessionClient
from runtime.transport.ids import runtime_session_name
from runtime.transport.lock import SessionNameLock, live_session_pid, registered_session_names, session_lock_path
from tests.runtime.test_session import _document, _document_with_cameras, _document_with_leader


def _name() -> str:
    return runtime_session_name(uuid4())


def _connect_owner(
    name: str,
    document: dict[str, Any],
    *,
    idle_timeout_s: float = 30.0,
    follower_name: str | None = "follower",
    leader_name: str | None = None,
) -> tuple[RuntimeSessionOwner, RuntimeSessionClient]:
    client = RuntimeSessionClient(name)
    client.open()
    owner = RuntimeSessionOwner(
        client,
        session_name=name,
        document=document,
        follower_name=follower_name,
        leader_name=leader_name,
        idle_timeout_s=idle_timeout_s,
    )
    try:
        owner.connect()
        client.wait_until_ready(owner, timeout=5)
    except Exception:
        owner.stop()
        client.close()
        raise
    return owner, client


def _stop_session(owner: RuntimeSessionOwner, *clients: RuntimeSessionClient) -> None:
    if clients:
        clients[0].apply(DisconnectCommand())
    deadline = time.monotonic() + 3
    while owner.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    owner.stop()
    for client in clients:
        client.close()


def _drain_until_source(client: RuntimeSessionClient, source: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = client.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            continue
        if isinstance(event, StateEvent) and event.data.follower_source == source:
            return
    pytest.fail(f"Did not observe follower_source={source!r}")


def _follower_source(metadata: dict[str, Any]) -> str | None:
    state = metadata.get("state")
    if not isinstance(state, dict):
        return None
    data = state.get("data")
    if not isinstance(data, dict):
        return None
    source = data.get("follower_source")
    return source if isinstance(source, str) else None


def _missing_pid() -> int:
    for candidate in range(2**22, 2**22 + 10_000):
        try:
            os.kill(candidate, 0)
        except OSError:
            return candidate
    raise RuntimeError("Could not find an unused pid")


def test_second_client_attaches_to_the_running_session() -> None:
    name = _name()
    document = _document()
    first_owner, first_client = _connect_owner(name, document)
    try:
        second_owner, second_client = _connect_owner(name, document)
        try:
            assert first_owner.spawned is True
            assert second_owner.spawned is False
            assert first_owner.metadata["pid"] == second_owner.metadata["pid"]
            assert first_owner.host is not None
            assert first_owner.host.is_alive()
            assert second_owner.host is None
        finally:
            second_client.close()
    finally:
        _stop_session(first_owner, first_client)


def test_losing_the_spawn_race_attaches_to_the_winner() -> None:
    name = _name()
    document = _document()

    def connect() -> tuple[RuntimeSessionOwner, RuntimeSessionClient]:
        return _connect_owner(name, document)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=30) for future in (pool.submit(connect), pool.submit(connect))]

    owners = [owner for owner, _ in results]
    clients = [client for _, client in results]
    try:
        pids = {owner.metadata["pid"] for owner in owners}
        assert len(pids) == 1
        assert sum(owner.spawned for owner in owners) <= 1
        assert any(not owner.spawned for owner in owners)
    finally:
        _stop_session(owners[0], *clients)
        for owner in owners[1:]:
            owner.stop()


def test_losing_the_spawn_race_attaches_when_the_winner_has_the_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = _name()
    document = _document_with_cameras("front")
    client = MagicMock()
    owner = RuntimeSessionOwner(
        client,
        session_name=name,
        document=document,
        follower_name="follower",
        leader_name=None,
        idle_timeout_s=30.0,
    )
    winner_metadata = {
        "identity_digest": runtime_identity_digest(document),
        "camera_keys": ["front"],
        "pid": 41273,
        "instance_id": "winner",
    }
    client.probe.return_value = None
    client.probe_with_retry.return_value = winner_metadata
    contention = AppBaseException(
        message="lock held",
        error_code="runtime_session_busy",
        http_status=http.HTTPStatus.LOCKED,
        phase="name_lock_contention",
    )
    host = MagicMock()
    host.start.side_effect = contention
    monkeypatch.setattr("runtime.owner.RuntimeProcessHost", lambda *_args, **_kwargs: host)
    stop = MagicMock()
    monkeypatch.setattr("runtime.owner.stop_runtime_session", stop)

    owner.connect()

    stop.assert_not_called()
    assert owner.spawned is False
    client.attach.assert_called_once_with(winner_metadata)
    assert owner.metadata == winner_metadata


def test_losing_the_spawn_race_restarts_when_the_winner_lacks_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = _name()
    document = _document_with_cameras("front")
    client = MagicMock()
    owner = RuntimeSessionOwner(
        client,
        session_name=name,
        document=document,
        follower_name="follower",
        leader_name=None,
        idle_timeout_s=30.0,
    )
    winner_metadata = {
        "identity_digest": runtime_identity_digest(document),
        "camera_keys": [],
        "pid": 41273,
        "instance_id": "winner",
    }
    spawned_metadata = {
        "identity_digest": runtime_identity_digest(document),
        "camera_keys": ["front"],
        "pid": 41274,
        "instance_id": "ours",
    }
    probe_calls = {"n": 0}

    def probe(*_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        probe_calls["n"] += 1
        if probe_calls["n"] <= 2:
            return None
        return spawned_metadata

    client.probe.side_effect = probe
    client.probe_with_retry.return_value = winner_metadata
    contention = AppBaseException(
        message="lock held",
        error_code="runtime_session_busy",
        http_status=http.HTTPStatus.LOCKED,
        phase="name_lock_contention",
    )
    hosts: list[MagicMock] = []

    def fake_host(*_args: Any, **_kwargs: Any) -> MagicMock:
        host = MagicMock()
        if not hosts:
            host.start.side_effect = contention
        else:
            host.start.return_value = None
            host.is_alive.return_value = True
        hosts.append(host)
        return host

    monkeypatch.setattr("runtime.owner.RuntimeProcessHost", fake_host)
    stop = MagicMock()
    monkeypatch.setattr("runtime.owner.stop_runtime_session", stop)

    owner.connect()

    stop.assert_called_once_with(name)
    assert len(hosts) == 2
    hosts[1].start.assert_called_once()
    assert owner.spawned is True
    client.attach.assert_called_once_with(spawned_metadata)
    assert owner.metadata == spawned_metadata
    client.probe_with_retry.assert_called_once()


def test_attaching_with_a_different_rig_is_refused() -> None:
    name = _name()
    owner, client = _connect_owner(name, _document(), follower_name="left arm")
    try:
        different = _document()
        different["init_args"]["fps"] = 15.0
        second_client = RuntimeSessionClient(name)
        second_client.open()
        second_owner = RuntimeSessionOwner(
            second_client,
            session_name=name,
            document=different,
            follower_name="left arm",
            leader_name=None,
            idle_timeout_s=30.0,
        )
        try:
            with pytest.raises(RuntimeSessionBusyError) as exc_info:
                second_owner.connect()
            assert int(exc_info.value.http_status) == http.HTTPStatus.LOCKED
            assert exc_info.value.error_code == "runtime_session_busy"
            assert "left arm" in exc_info.value.message
            assert str(owner.metadata["pid"]) in exc_info.value.message
        finally:
            second_owner.stop()
            second_client.close()
    finally:
        _stop_session(owner, client)


def test_replace_stops_the_existing_session_and_spawns_with_the_new_rig() -> None:
    name = _name()
    owner, client = _connect_owner(name, _document(), follower_name="left arm")
    original_pid = owner.metadata["pid"]
    try:
        different = _document()
        different["init_args"]["fps"] = 15.0
        second_client = RuntimeSessionClient(name)
        second_client.open()
        second_owner = RuntimeSessionOwner(
            second_client,
            session_name=name,
            document=different,
            follower_name="left arm",
            leader_name=None,
            idle_timeout_s=30.0,
        )
        try:
            second_owner.connect(replace=True)
            second_client.wait_until_ready(second_owner, timeout=5)
            assert second_owner.spawned is True
            assert second_owner.metadata["pid"] != original_pid
            assert second_owner.metadata["identity_digest"] == runtime_identity_digest(different)
            assert second_owner.metadata["camera_keys"] == []
            assert not owner.is_alive()
        finally:
            _stop_session(second_owner, second_client)
    finally:
        if owner.is_alive():
            owner.stop()
        client.close()


def test_replace_spawns_even_when_the_rig_matches() -> None:
    name = _name()
    document = _document()
    owner, client = _connect_owner(name, document)
    original_pid = owner.metadata["pid"]
    try:
        second_client = RuntimeSessionClient(name)
        second_client.open()
        second_owner = RuntimeSessionOwner(
            second_client,
            session_name=name,
            document=document,
            follower_name="follower",
            leader_name=None,
            idle_timeout_s=30.0,
        )
        try:
            second_owner.connect(replace=True)
            second_client.wait_until_ready(second_owner, timeout=5)
            assert second_owner.spawned is True
            assert second_owner.metadata["pid"] != original_pid
        finally:
            _stop_session(second_owner, second_client)
    finally:
        if owner.is_alive():
            owner.stop()
        client.close()


def test_stop_runtime_session_is_a_noop_when_nothing_is_running() -> None:
    stop_runtime_session(_name())


def test_stop_runtime_session_terminates_a_live_child() -> None:
    name = _name()
    owner, client = _connect_owner(name, _document())
    try:
        stop_runtime_session(name)
        deadline = time.monotonic() + 3
        while owner.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not owner.is_alive()
        assert live_session_pid(name) is None
    finally:
        owner.stop()
        client.close()


def test_losing_the_last_subscriber_switches_to_hold() -> None:
    name = _name()
    document = _document_with_leader()
    owner, client = _connect_owner(name, document, idle_timeout_s=5.0, leader_name="leader")
    try:
        client.apply(SetFollowerSourceCommand(follower_source="teleop"))
        _drain_until_source(client, "teleop")
        client.close()
        time.sleep(0.4)

        second_owner, second_client = _connect_owner(name, document, idle_timeout_s=5.0, leader_name="leader")
        try:
            assert second_owner.spawned is False
            assert _follower_source(second_owner.metadata) == "hold"
        finally:
            _stop_session(second_owner, second_client)
    finally:
        if owner.is_alive():
            owner.stop()


def test_session_exits_when_no_client_is_attached() -> None:
    name = _name()
    owner, client = _connect_owner(name, _document(), idle_timeout_s=1.0)
    try:
        client.close()
        deadline = time.monotonic() + 5
        while owner.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not owner.is_alive()
    finally:
        owner.stop()


def test_a_client_attaching_on_the_idle_deadline_keeps_the_session() -> None:
    name = _name()
    owner, client = _connect_owner(name, _document(), idle_timeout_s=1.5)
    try:
        client.close()
        time.sleep(1.2)
        second_owner, second_client = _connect_owner(name, _document(), idle_timeout_s=1.5)
        try:
            time.sleep(1.0)
            assert second_owner.is_alive()
            assert second_owner.metadata["pid"] == owner.metadata["pid"]
        finally:
            _stop_session(second_owner, second_client)
    finally:
        if owner.is_alive():
            owner.stop()


def test_a_new_client_reattaches_after_every_client_goes_away() -> None:
    name = _name()
    document = _document()
    owner, client = _connect_owner(name, document, idle_timeout_s=5.0)
    original_pid = owner.metadata["pid"]
    try:
        client.close()
        time.sleep(0.2)
        second_owner, second_client = _connect_owner(name, document, idle_timeout_s=5.0)
        try:
            assert second_owner.spawned is False
            assert second_owner.metadata["pid"] == original_pid
            events: list[object] = []
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    events.append(second_client.get_nowait())
                except queue.Empty:
                    time.sleep(0.01)
                if any(isinstance(event, StateEvent) for event in events):
                    break
            assert any(isinstance(event, StateEvent) and event.data.connected for event in events)
        finally:
            _stop_session(second_owner, second_client)
    finally:
        if owner.is_alive():
            owner.stop()


def test_dead_pid_is_not_reported_as_a_holder() -> None:
    name = _name()
    path = session_lock_path(name)
    path.write_text(
        json.dumps(
            {
                "kind": "rt-name",
                "identity": name,
                "pid": _missing_pid(),
                "created_at": time.time(),
            }
        ),
        encoding="utf-8",
    )

    assert name not in registered_session_names()
    assert live_session_pid(name) is None


def test_released_lock_is_not_reported_as_a_holder() -> None:
    name = _name()
    lock = SessionNameLock(name)
    assert lock.acquire()
    assert name in registered_session_names()
    lock.release()

    assert lock.path.exists()
    assert name not in registered_session_names()
    assert live_session_pid(name) is None


def test_holder_does_not_probe_when_the_lock_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _probe(session_name: str, timeout: float = 1.0) -> dict[str, Any] | None:
        calls.append(session_name)
        return {"pid": 1}

    monkeypatch.setattr("runtime.owner.probe_session_metadata", _probe)
    assert runtime_session_holder(uuid4()) is None
    assert calls == []


def test_probe_session_metadata_is_reachable_after_lock_hit() -> None:
    name = _name()
    owner, client = _connect_owner(name, _document())
    try:
        follower_id = UUID(name.removeprefix("rt-"))
        holder = runtime_session_holder(follower_id)
        assert holder is not None
        assert holder["pid"] == owner.metadata["pid"]
        assert probe_session_metadata(name) is not None
    finally:
        _stop_session(owner, client)


def test_holder_treats_a_lock_without_metadata_as_held(monkeypatch: pytest.MonkeyPatch) -> None:
    follower_id = uuid4()
    name = runtime_session_name(follower_id)
    monkeypatch.setattr("runtime.owner.live_session_pid", lambda session_name: 41273 if session_name == name else None)
    monkeypatch.setattr("runtime.owner.probe_session_metadata", lambda *args, **kwargs: None)

    holder = runtime_session_holder(follower_id)

    assert holder == {"pid": 41273}


def _owner_with_mock_client(name: str | None = None) -> tuple[RuntimeSessionOwner, Any]:
    client = MagicMock()
    owner = RuntimeSessionOwner(
        client,
        session_name=name or _name(),
        document=_document(),
        follower_name="follower",
        leader_name=None,
        idle_timeout_s=30.0,
    )
    return owner, client


def test_abandoned_spawn_is_stopped_before_metadata_answers() -> None:
    owner, client = _owner_with_mock_client()
    host = MagicMock()
    owner._host = host
    client.probe.return_value = None

    owner.stop_abandoned_spawn()

    host.stop.assert_called_once()


def test_abandoned_spawn_is_not_stopped_once_metadata_answers() -> None:
    owner, client = _owner_with_mock_client()
    host = MagicMock()
    owner._host = host
    client.probe.return_value = {"pid": 41273}

    owner.stop_abandoned_spawn()

    host.stop.assert_not_called()


def test_abandoned_spawn_is_not_stopped_after_the_child_is_ready() -> None:
    owner, client = _owner_with_mock_client()
    host = MagicMock()
    owner._host = host
    owner._spawned = True

    owner.stop_abandoned_spawn()

    host.stop.assert_not_called()
    client.probe.assert_not_called()


def test_attached_owner_does_not_stop_on_abandon() -> None:
    owner, _client = _owner_with_mock_client()

    owner.stop_abandoned_spawn()


def test_attaching_with_extra_cameras_is_allowed() -> None:
    name = _name()
    running = _document_with_cameras("front", "wrist")
    owner, client = _connect_owner(name, running)
    try:
        subset = _document_with_cameras("front")
        second_owner, second_client = _connect_owner(name, subset)
        try:
            assert second_owner.spawned is False
            assert second_owner.metadata["pid"] == owner.metadata["pid"]
            assert second_owner.metadata["camera_keys"] == ["front", "wrist"]
            assert second_owner.metadata["identity_digest"] == runtime_identity_digest(subset)
        finally:
            second_client.close()
    finally:
        _stop_session(owner, client)


def test_a_client_needing_more_cameras_restarts_the_session() -> None:
    name = _name()
    owner, client = _connect_owner(name, _document())
    original_pid = owner.metadata["pid"]
    try:
        needing_cameras = _document_with_cameras("front")
        second_owner, second_client = _connect_owner(name, needing_cameras)
        try:
            assert second_owner.spawned is True
            assert second_owner.metadata["pid"] != original_pid
            assert second_owner.metadata["camera_keys"] == ["front"]
            assert not owner.is_alive()
            assert live_session_pid(name) == second_owner.metadata["pid"]
        finally:
            _stop_session(second_owner, second_client)
    finally:
        if owner.is_alive():
            owner.stop()
        client.close()


def test_a_client_needing_a_different_robot_is_refused_before_the_camera_check() -> None:
    name = _name()
    owner, client = _connect_owner(name, _document(), follower_name="left arm")
    try:
        different = _document_with_cameras("front", name="other-follower")
        second_client = RuntimeSessionClient(name)
        second_client.open()
        second_owner = RuntimeSessionOwner(
            second_client,
            session_name=name,
            document=different,
            follower_name="left arm",
            leader_name=None,
            idle_timeout_s=30.0,
        )
        try:
            with pytest.raises(RuntimeSessionBusyError) as exc_info:
                second_owner.connect()
            assert int(exc_info.value.http_status) == http.HTTPStatus.LOCKED
            assert not second_owner.spawned
            assert owner.is_alive()
            assert live_session_pid(name) == owner.metadata["pid"]
        finally:
            second_owner.stop()
            second_client.close()
    finally:
        _stop_session(owner, client)
