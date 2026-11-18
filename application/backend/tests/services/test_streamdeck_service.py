from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from exceptions import BaseException as APIError
from runtime_session.registry import RuntimeSessionRegistry
from schemas.streamdeck import StreamDeckHeartbeat
from services.streamdeck_service import StreamDeckService


def test_lease_expiry_requires_explicit_reattachment(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("services.streamdeck_service.time", SimpleNamespace(monotonic=lambda: clock[0]))
    registry = Mock(spec=RuntimeSessionRegistry)
    registry.get.return_value = SimpleNamespace(closed=Event())
    service = StreamDeckService()
    helper_id = uuid4()
    service.report(StreamDeckHeartbeat(helper_id=helper_id), registry)
    target = service.connect("session", "Pick block", registry).target
    assert target is not None
    assert service.report(
        StreamDeckHeartbeat(helper_id=helper_id, connection_id=target.connection_id, device="Mini"), registry
    ).connected

    clock[0] += service.LEASE_SECONDS
    assert not service.status(registry).helper_online
    assert service.status(registry).target is None
    assert service.report(StreamDeckHeartbeat(helper_id=helper_id), registry).target is None
    with pytest.raises(APIError):
        service.report(StreamDeckHeartbeat(helper_id=uuid4()), registry)


def test_task_updates_keep_connection_but_reconnect_replaces_it() -> None:
    registry = Mock(spec=RuntimeSessionRegistry)
    registry.get.return_value = SimpleNamespace(closed=Event())
    service = StreamDeckService()
    with pytest.raises(APIError):
        service.connect("one", "Pick block", registry)
    service.report(StreamDeckHeartbeat(helper_id=uuid4()), registry)
    first = service.connect("one", "Pick block", registry).target
    updated = service.connect("one", "", registry).target
    assert first is not None and updated is not None
    assert first.connection_id == updated.connection_id and updated.task == ""
    with pytest.raises(APIError):
        service.connect("two", "Other task", registry)
    assert service.disconnect("two", registry).target == updated
    assert service.disconnect("one", registry).target is None
    reconnected = service.connect("one", "New task", registry).target
    assert reconnected is not None and reconnected.connection_id != first.connection_id


def test_closed_browser_session_releases_target() -> None:
    closed = Event()
    registry = Mock(spec=RuntimeSessionRegistry)
    registry.get.return_value = SimpleNamespace(closed=closed)
    service = StreamDeckService()
    service.report(StreamDeckHeartbeat(helper_id=uuid4()), registry)
    service.connect("one", "Pick block", registry)
    closed.set()
    assert service.status(registry).target is None
