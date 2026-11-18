"""Inference controls attach to an existing session; voice only delivers stopped drafts."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_runtime_session_registry, get_streamdeck_service
from main import app
from schemas.runtime_session import AckMessage, ModelControlConfig, TeleopControlConfig
from services.streamdeck_service import StreamDeckService


@pytest.fixture
def deck(monkeypatch):
    session = SimpleNamespace(closed=Event(), bus=Mock())
    state = SimpleNamespace(control=None)
    session.state = lambda: state
    registry = Mock()
    registry.get.side_effect = lambda name: session if name == "one" else None
    service = StreamDeckService()
    helper_id = str(uuid4())

    async def dispatch(_bus, raw):
        command = json.loads(raw)
        state.control = ModelControlConfig.model_validate(command["control"]) if command["control"] else None
        return AckMessage(request_id=command["request_id"], ok=True)

    dispatch_mock = AsyncMock(side_effect=dispatch)
    monkeypatch.setattr("api.streamdeck.request_command", dispatch_mock)
    app.dependency_overrides[get_runtime_session_registry] = lambda: registry
    app.dependency_overrides[get_streamdeck_service] = lambda: service
    try:
        client = TestClient(app)
        assert client.post("/api/streamdeck/heartbeat", json={"helper_id": helper_id}).status_code == 200
        target = client.put(
            "/api/streamdeck/sessions/one",
            json={"task": "Old task", "page": "inference", "inference": {"hz": 15, "execution": {"mode": "async"}}},
        ).json()["target"]
        yield client, state, target, helper_id, dispatch_mock
    finally:
        app.dependency_overrides.clear()


def test_dictation_is_a_draft_and_explicit_start_uses_workspace_settings(deck):
    client, state, target, _, dispatch = deck
    headers = {"X-StreamDeck-Connection": target["connection_id"]}
    path = "/api/streamdeck/sessions/one"
    result = client.post(f"{path}/dictation", headers=headers, json={"task": "Put the block in the bowl"})
    assert result.status_code == 200
    draft = result.json()["target"]
    assert draft["task"] == draft["dictation"]["task"] == "Put the block in the bowl"
    assert draft["dictation"]["id"] and draft["connection_id"] == target["connection_id"]
    assert state.control is None
    dispatch.assert_not_called()
    command = {"action": "start", "request_id": "start"}
    assert client.post(f"{path}/inference", headers=headers, json=command).json()["ok"]
    assert state.control.task == draft["task"]
    assert state.control.hz == 15 and state.control.execution.mode == "async" and state.control.grace_s == 3
    assert client.post(f"{path}/inference", headers=headers, json=command).status_code == 409
    assert client.post(f"{path}/dictation", headers=headers, json={"task": "Change running task"}).status_code == 409
    assert state.control.task == draft["task"]
    stop = {"action": "stop", "request_id": "stop"}
    assert client.post(f"{path}/inference", headers=headers, json=stop).json()["ok"]
    assert state.control is None
    assert client.post(f"{path}/inference", headers=headers, json=stop).json()["ok"]
    assert dispatch.await_count == 2


def test_page_change_fences_old_buttons_and_late_transcripts(deck):
    client, _, target, _, dispatch = deck
    path = "/api/streamdeck/sessions/one"
    headers = {"X-StreamDeck-Connection": target["connection_id"]}
    # Recording commands cannot be sent from the inference page either.
    assert (
        client.post(
            "/api/runtime/sessions/one/recording",
            headers=headers,
            json={"command": "save_episode", "request_id": "wrong-page"},
        ).status_code
        == 409
    )
    record = client.put(path, json={"task": "record", "page": "record"}).json()["target"]
    assert record["connection_id"] != target["connection_id"] and record["dictation"] is None
    assert client.post(f"{path}/dictation", headers=headers, json={"task": "Late result"}).status_code == 409
    headers = {"X-StreamDeck-Connection": record["connection_id"]}
    assert (
        client.post(f"{path}/inference", headers=headers, json={"action": "start", "request_id": "old"}).status_code
        == 409
    )
    dispatch.assert_not_called()


@pytest.mark.parametrize("voice", ["preparing", "listening", "transcribing"])
def test_start_is_blocked_during_voice_and_stop_never_stops_teleop(deck, voice):
    client, state, target, helper, dispatch = deck
    headers = {"X-StreamDeck-Connection": target["connection_id"]}
    client.post(
        "/api/streamdeck/heartbeat",
        json={
            "helper_id": helper,
            "connection_id": target["connection_id"],
            "dictation_status": voice,
        },
    )
    path = "/api/streamdeck/sessions/one"
    assert client.get("/api/streamdeck").json()["dictation_status"] == voice
    assert (
        client.post(f"{path}/inference", headers=headers, json={"action": "start", "request_id": "busy"}).status_code
        == 409
    )
    state.control = TeleopControlConfig()
    assert (
        client.post(f"{path}/inference", headers=headers, json={"action": "stop", "request_id": "teleop"}).status_code
        == 409
    )
    assert client.post(f"{path}/dictation", headers=headers, json={"task": "No"}).status_code == 409
    dispatch.assert_not_called()


def test_inference_input_validation_and_detachment(deck):
    client, _, target, _, dispatch = deck
    path = "/api/streamdeck/sessions/one"
    headers = {"X-StreamDeck-Connection": target["connection_id"]}
    assert client.post(f"{path}/inference", json={"action": "start", "request_id": "no-header"}).status_code == 400
    assert (
        client.post(
            f"{path}/inference", headers=headers, json={"action": "teleop", "request_id": "invalid"}
        ).status_code
        == 400
    )
    for task, code in [(" ", 409), ("x" * 20_001, 400)]:
        assert client.post(f"{path}/dictation", headers=headers, json={"task": task}).status_code == code
    client.delete(path)
    assert client.post(f"{path}/dictation", headers=headers, json={"task": "Disconnected"}).status_code == 409
    dispatch.assert_not_called()
