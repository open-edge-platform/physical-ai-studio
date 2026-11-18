from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import (
    get_camera_claim_registry,
    get_camera_service,
    get_dataset_service,
    get_project_service,
    get_robot_client_factory,
    get_robot_service,
    get_runtime_session_registry,
    get_streamdeck_service,
)
from api.runtime_ws import observation_message
from main import app
from schemas.runtime_session import ObservationFrame, ObservationMessage
from services.streamdeck_service import StreamDeckService
from tests.runtime_session.fixtures import FOLLOWER_ID, LEADER_ID, make_environment

if TYPE_CHECKING:
    from collections.abc import Iterator

PROJECT_ID = uuid4()
ENVIRONMENT = make_environment()


def _draft() -> dict:
    return {
        "name": "Bench",
        "robots": [{"robot_id": str(FOLLOWER_ID), "tele_operator": {"type": "robot", "robot_id": str(LEADER_ID)}}],
        "cameras": [{"camera_id": str(ENVIRONMENT.cameras[0].id)}],
    }


@pytest.fixture
def client(robot_factory, registry, camera_claims, fake_cameras) -> Iterator[TestClient]:
    rows = {FOLLOWER_ID: ENVIRONMENT.robots[0].robot, LEADER_ID: ENVIRONMENT.robots[0].tele_operator.robot}
    robots = SimpleNamespace(get_robot_by_id=AsyncMock(side_effect=lambda _project, robot_id: rows[robot_id]))
    cameras = SimpleNamespace(get_camera_by_id=AsyncMock(return_value=ENVIRONMENT.cameras[0]))
    projects = SimpleNamespace(get_project_by_id=AsyncMock(return_value=SimpleNamespace(name="Alpha")))
    streamdeck = StreamDeckService()
    app.dependency_overrides.update(
        {
            get_project_service: lambda: projects,
            get_robot_service: lambda: robots,
            get_camera_service: lambda: cameras,
            get_robot_client_factory: lambda: robot_factory,
            get_runtime_session_registry: lambda: registry,
            get_camera_claim_registry: lambda: camera_claims,
            get_streamdeck_service: lambda: streamdeck,
        }
    )
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _url() -> str:
    return f"/api/projects/{PROJECT_ID}/runtime/ws?hz=30"


def _receive_until(websocket, predicate, limit: int = 200) -> dict:
    for _ in range(limit):
        message = websocket.receive()
        if "text" in message and predicate(parsed := json.loads(message["text"])):
            return parsed
    raise AssertionError("message never arrived")


class _Socket:
    """Send commands and read acks, keeping the latest state meanwhile.

    The session publishes its state as soon as it changes, so a state can arrive
    before the ack of the command that changed it.
    """

    def __init__(self, websocket) -> None:
        self.websocket = websocket
        self.state: dict = {}

    def receive_until(self, predicate, limit: int = 300) -> dict:
        for _ in range(limit):
            message = self.websocket.receive()
            if "text" not in message:
                continue
            parsed = json.loads(message["text"])
            if parsed["type"] == "state":
                self.state = parsed["data"]
            if predicate(parsed):
                return parsed
        raise AssertionError("message never arrived")

    def ack(self, command: dict) -> dict:
        self.websocket.send_json({**command, "request_id": command["command"]})
        return self.receive_until(lambda m: m["type"] == "ack" and m["request_id"] == command["command"])

    def until_state(self, predicate) -> dict:
        if not (self.state and predicate(self.state)):
            self.receive_until(lambda m: m["type"] == "state" and predicate(m["data"]))
        return self.state


def _parse_observation(data: bytes) -> tuple[dict, dict[str, bytes]]:
    length = int.from_bytes(data[:4], "big")
    header = json.loads(data[4 : 4 + length])
    frames, offset = {}, 4 + length
    for frame in header["frames"]:
        frames[frame["key"]] = data[offset : offset + frame["size"]]
        offset += frame["size"]
    return header, frames


def test_ui_manages_streamdeck_attachment_without_owning_a_session(client: TestClient) -> None:
    assert not client.get("/api/streamdeck").json()["helper_online"]
    with client.websocket_connect(_url()) as websocket:
        session_id = websocket.receive_json()["data"]["session_id"]
        path = f"/api/streamdeck/sessions/{session_id}"
        assert client.put(path, json={"task": "Pick block"}).status_code == 409
        report = {"helper_id": str(uuid4())}
        assert client.post("/api/streamdeck/heartbeat", json=report).json()["helper_online"]
        assert client.post("/api/streamdeck/heartbeat", json={"helper_id": str(uuid4())}).status_code == 409
        assert client.put("/api/streamdeck/sessions/missing", json={"task": "Pick block"}).status_code == 404
        assert client.put(path, json={}).status_code == 400
        target = client.put(path, json={"task": "Pick block"}).json()["target"]
        assert target["session_id"] == session_id and target["task"] == "Pick block"
        report.update(connection_id=target["connection_id"], device="Stream Deck Mini")
        assert client.post("/api/streamdeck/heartbeat", json=report).json()["connected"]
        updated = client.put(path, json={"task": "Place block"}).json()["target"]
        assert updated["connection_id"] == target["connection_id"]
        recording = f"/api/runtime/sessions/{session_id}/recording"
        headers = {"X-StreamDeck-Connection": target["connection_id"]}
        command = {"command": "start_episode", "request_id": "deck", "task": "Place block"}
        # A current attachment reaches the shared handler; there is deliberately no dataset loaded here.
        ack = client.post(recording, json=command, headers=headers).json()
        assert ack["error"]["error_code"] == "command_unavailable"
        assert client.delete(path).json()["target"] is None
        assert client.post(recording, json=command, headers=headers).status_code == 409
        assert client.get(f"/api/runtime/sessions/{session_id}/state").json()["session_id"] == session_id
        new_target = client.put(path, json={"task": "Pick block"}).json()["target"]
        assert new_target["connection_id"] != target["connection_id"]
        assert client.post(recording, json=command, headers=headers).status_code == 409
    assert client.get("/api/streamdeck").json()["target"] is None


def test_observation_message_layout() -> None:
    header = ObservationMessage(
        timestamp=2.0, state={"a.pos": 1.0}, action=None, frames=[ObservationFrame(key="cam 0", timestamp=2.0, size=4)]
    )
    parsed, frames = _parse_observation(observation_message(header, [b"jpeg"]))
    assert parsed == header.model_dump(mode="json")
    assert frames == {"cam 0": b"jpeg"}


def test_load_stream_and_unload(client: TestClient, registry) -> None:
    with client.websocket_connect(_url()) as websocket:
        first = websocket.receive_json()
        assert first["type"] == "state" and first["data"]["environment"] is None
        socket = _Socket(websocket)

        ack = socket.ack({"command": "load_environment", "environment": _draft()})
        assert ack == {"type": "ack", "request_id": "load_environment", "ok": True, "error": None, "result": None}
        state = socket.until_state(lambda data: data["environment"] is not None)
        assert state["environment"]["joint_names"] == ["shoulder_pan.pos", "gripper.pos"]
        assert [s.info().environment_name for s in registry.sessions()] == ["Bench"]

        for _ in range(200):
            message = websocket.receive()
            if message.get("bytes"):
                observation, frames = _parse_observation(message["bytes"])
                if frames:
                    break
        else:
            raise AssertionError("no observation with a camera frame")
        assert observation["state"] == {"shoulder_pan.pos": 1.0, "gripper.pos": 2.0}
        assert observation["action"] == {"shoulder_pan.pos": 1.0, "gripper.pos": 2.0}
        assert set(frames) == {"cam 0"}
        assert frames["cam 0"].startswith(b"\xff\xd8")  # a JPEG

        assert socket.ack({"command": "unload_environment"})["ok"]
    assert registry.sessions() == []
    assert registry.holder_of_robot(FOLLOWER_ID) is None


def test_bad_commands_get_a_failed_ack(client: TestClient) -> None:
    with client.websocket_connect(_url()) as websocket:
        websocket.receive_json()
        websocket.send_json({"command": "fly", "request_id": "7"})
        ack = websocket.receive_json()
        assert ack["request_id"] == "7" and not ack["ok"] and ack["error"]["error_code"] == "command_unavailable"
        websocket.send_json({"command": "set_control", "request_id": "8"})  # missing its control
        ack = websocket.receive_json()
        assert ack["request_id"] == "8" and ack["error"]["error_code"] == "invalid_command"
        websocket.send_text("not json")
        assert websocket.receive_json()["error"]["error_code"] == "invalid_command"


def test_sessions_socket_pushes_changes_and_stop_closes_the_session(client: TestClient) -> None:
    with client.websocket_connect("/api/runtime/sessions/ws") as sessions:
        assert sessions.receive_json() == {"sessions": []}
        with client.websocket_connect(_url()) as websocket:
            websocket.receive_json()
            websocket.send_json({"command": "load_environment", "request_id": "1", "environment": _draft()})
            assert _receive_until(websocket, lambda m: m["type"] == "ack")["ok"]

            listed = sessions.receive_json()["sessions"]
            while not listed or listed[0]["environment_name"] is None:  # opened, then loaded
                listed = sessions.receive_json()["sessions"]
            [session] = listed
            assert session["environment_name"] == "Bench"
            assert set(session["robot_ids"]) == {str(FOLLOWER_ID), str(LEADER_ID)}

            with client.websocket_connect(f"/api/projects/{PROJECT_ID}/robots/{FOLLOWER_ID}/observations/ws") as form:
                refused = form.receive_json()
            assert refused["error_code"] == "device_in_use"
            assert "Bench" in refused["message"]
            assert client.delete(f"/api/projects/{PROJECT_ID}/robots/{FOLLOWER_ID}").status_code == 423

            assert client.post(f"/api/runtime/sessions/{session['session_id']}/stop").status_code == 204
            closed = _receive_until(websocket, lambda m: m["type"] == "error")
            assert closed["error_code"] == "session_closed" and not closed["session_alive"]
        while sessions.receive_json()["sessions"]:
            pass
    assert client.post("/api/runtime/sessions/nope/stop").status_code == 404


def test_set_control_is_acked_and_reported_in_state(client: TestClient) -> None:
    with client.websocket_connect(_url()) as websocket:
        websocket.receive_json()
        socket = _Socket(websocket)
        refused = socket.ack({"command": "set_control", "control": {"type": "teleop"}})
        assert refused["error"]["error_code"] == "no_environment"

        assert socket.ack({"command": "load_environment", "environment": _draft()})["ok"]
        assert socket.ack({"command": "set_control", "control": {"type": "teleop", "hz": 50}})["ok"]
        state = socket.until_state(lambda data: data["control"] is not None)
        assert state["control"] == {"type": "teleop", "hz": 50.0, "grace_s": 3.0}  # the defaults

        assert socket.ack({"command": "set_control", "control": None})["ok"]
        assert socket.until_state(lambda data: data["control"] is None)["control"] is None


def test_recording_commands_over_the_socket(client: TestClient) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from schemas.dataset import Dataset
    from tests.workers.fake_dataset import FakeDatasetWorker

    ours = Dataset(id=uuid4(), name="Blocks", default_task="pick", project_id=PROJECT_ID, environment_id=uuid4())
    theirs = Dataset(id=uuid4(), name="Other", default_task="pick", project_id=uuid4(), environment_id=uuid4())
    Path(ours.path).mkdir(parents=True)
    datasets = {ours.id: ours, theirs.id: theirs}
    app.dependency_overrides[get_dataset_service] = lambda: SimpleNamespace(
        get_dataset_by_id=AsyncMock(side_effect=lambda dataset_id: datasets[dataset_id])
    )

    with (
        patch("runtime_session.dataset.DatasetWorker", FakeDatasetWorker),
        client.websocket_connect(_url()) as websocket,
    ):
        websocket.receive_json()
        socket = _Socket(websocket)
        assert socket.ack({"command": "load_environment", "environment": _draft()})["ok"]
        unavailable = socket.ack({"command": "start_episode", "task": "pick"})
        assert unavailable["error"]["error_code"] == "command_unavailable"
        refused = socket.ack({"command": "load_dataset", "dataset_id": str(theirs.id)})
        assert refused["error"]["error_code"] == "Dataset_not_found"  # as the REST API reports it

        assert socket.ack({"command": "load_dataset", "dataset_id": str(ours.id)})["ok"]
        assert socket.ack({"command": "start_episode", "task": "pick"})["ok"]
        time.sleep(0.3)  # record enough frames: a shorter episode is refused as too short
        assert socket.ack({"command": "save_episode"})["ok"]
        # The worker publishes its new status; the session shows it.
        state = socket.until_state(lambda data: (data["dataset"] or {}).get("episodes_recorded") == 1)
        assert not state["dataset"]["is_recording"]
        failed = socket.ack({"command": "save_episode"})
        assert failed["error"]["error_code"] == "no_episode"
        assert "No episode" in failed["error"]["message"]
        # Swapping to an environment the dataset does not fit is refused with what it would unload.
        Path(ours.path, "meta").mkdir()
        Path(ours.path, "meta", "info.json").write_text(
            json.dumps(
                {
                    "fps": 30,
                    "features": {
                        "observation.state": {"shape": [2], "names": ["shoulder_pan.pos", "gripper.pos"]},
                        "action": {"shape": [2], "names": ["shoulder_pan.pos", "gripper.pos"]},
                        "observation.images.cam 0": {"shape": [120, 160, 3]},
                    },
                }
            )
        )
        no_cameras = {**_draft(), "cameras": []}
        refused = socket.ack({"command": "load_environment", "environment": no_cameras})
        assert refused["error"]["error_code"] == "incompatible_parts"
        assert refused["error"]["parts"][0]["part"] == "dataset"
        swapped = socket.ack({"command": "load_environment", "environment": no_cameras, "unload_incompatible": True})
        assert swapped["ok"]
        assert socket.until_state(lambda data: data["dataset"] is None)["environment"]["cameras"] == []


def test_attached_recording_controls_share_the_browser_session(client: TestClient, registry) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from schemas.dataset import Dataset
    from tests.workers.fake_dataset import FakeDatasetWorker

    dataset = Dataset(id=uuid4(), name="Blocks", default_task="pick", project_id=PROJECT_ID, environment_id=uuid4())
    Path(dataset.path).mkdir(parents=True)
    app.dependency_overrides[get_dataset_service] = lambda: SimpleNamespace(
        get_dataset_by_id=AsyncMock(return_value=dataset)
    )
    assert client.get("/api/runtime/sessions").json() == {"sessions": []}
    with (
        patch("runtime_session.dataset.DatasetWorker", FakeDatasetWorker),
        client.websocket_connect(_url()) as websocket,
    ):
        websocket.receive_json()
        socket = _Socket(websocket)
        [session] = client.get("/api/runtime/sessions").json()["sessions"]
        path = f"/api/runtime/sessions/{session['session_id']}"
        assert client.get(f"{path}/state").json()["environment"] is None
        unavailable = client.post(
            f"{path}/recording", json={"command": "start_episode", "request_id": "empty", "task": "pick"}
        )
        assert unavailable.status_code == 200
        assert unavailable.json()["error"]["error_code"] == "command_unavailable"
        assert socket.ack({"command": "load_environment", "environment": _draft()})["ok"]
        assert socket.ack({"command": "load_dataset", "dataset_id": str(dataset.id)})["ok"]
        assert socket.ack({"command": "set_control", "control": {"type": "teleop"}})["ok"]

        def command(name: str, **payload) -> dict:
            response = client.post(f"{path}/recording", json={"command": name, "request_id": name, **payload})
            assert response.status_code == 200
            ack = response.json()
            assert ack["request_id"] == name and ack["ok"]
            return ack

        command("start_episode", task="from deck")
        socket.until_state(lambda data: (data["dataset"] or {}).get("is_recording"))
        time.sleep(0.3)
        command("save_episode")
        socket.until_state(lambda data: (data["dataset"] or {}).get("episodes_recorded") == 1)
        command("start_episode", task="discard me")
        socket.until_state(lambda data: data["dataset"]["is_recording"])
        command("discard_episode")
        socket.until_state(lambda data: not data["dataset"]["is_recording"])
        snapshot = client.get(f"{path}/state").json()
        assert snapshot["dataset"]["episodes_recorded"] == 1
        assert snapshot["control"]["type"] == "teleop"
        assert [s.id for s in registry.sessions()] == [session["session_id"]]
        # Attached clients cannot change motion, load hardware, or omit required fields.
        for payload in (
            {"command": "set_control", "control": None},
            {"command": "load_environment", "environment": _draft()},
            {"command": "start_episode"},
            {"command": "fly"},
        ):
            assert client.post(f"{path}/recording", json={**payload, "request_id": "invalid"}).status_code == 400
    assert client.get("/api/runtime/sessions").json() == {"sessions": []}
    assert client.get(f"{path}/state").status_code == 404
    assert client.post(f"{path}/recording", json={"command": "save_episode", "request_id": "gone"}).status_code == 404


def test_native_launch_requires_local_ui_and_non_simple_header(client, monkeypatch) -> None:
    from unittest.mock import Mock

    launch = Mock()
    monkeypatch.setattr(StreamDeckService, "start", launch)
    monkeypatch.setattr(StreamDeckService, "native_launch_available", staticmethod(lambda: True))
    headers = {"X-StreamDeck-Launch": "1"}
    assert client.post("/api/streamdeck/start", headers=headers).status_code == 403
    local = TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")
    assert local.get("/api/streamdeck").json()["can_start_native"]
    assert local.post("/api/streamdeck/start").status_code == 400
    for origin in ("http://attacker.example", "null", "http://[invalid"):
        assert local.post("/api/streamdeck/start", headers={**headers, "Origin": origin}).status_code == 403
    launch.assert_not_called()
    assert (
        local.post("/api/streamdeck/start", headers={**headers, "Origin": "http://localhost:3000"}).status_code == 200
    )
    launch.assert_called_once()
    assert local.post("/api/streamdeck/heartbeat", json={"helper_id": str(uuid4())}).status_code == 200
    assert local.post("/api/streamdeck/start", headers=headers).status_code == 200
    launch.assert_called_once()
