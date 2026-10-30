# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Passing a simulation's viewer and camera frames on, against small upstreams on free ports."""

import contextlib
import http.server
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
import websockets.exceptions
import websockets.sync.server
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect
from websockets.http11 import Request, Response
from websockets.sync.server import ServerConnection

import api.simulation_proxy as proxy_routes
import services.robot_simulation_service as service_module
from api.dependencies import get_robot_catalog_service, get_simulation_manager
from exceptions import ResourceNotFoundError, ResourceType
from main import app
from schemas.simulation import Simulation, SimulationStatus
from services.simulation_proxy import is_proxied_http_path, is_viewer_path, local_base_url

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()
PROTOCOL = "viser-v1.1.1"
BASE = f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/simulation"


class SimulationApi(http.server.BaseHTTPRequestHandler):
    """The simulation's HTTP API: a camera frame and stream, and a path the browser must not get."""

    requests: list[str] = []
    redirect_to: str | None = None

    def do_GET(self) -> None:
        SimulationApi.requests.append(self.path)
        if self.path == "/cameras/wrist/frame.jpg":
            self._send(b"frame", "image/jpeg")
        elif self.path == "/cameras/wrist/mjpeg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            for _ in range(3):
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\njpeg\r\n")
                self.wfile.flush()
        elif self.path == "/cameras/moved/frame.jpg" and SimulationApi.redirect_to:
            self.send_response(302)
            self.send_header("Location", SimulationApi.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/":
            self._send(b'{"service": "sim-a"}', "application/json")
        else:
            self._send(b"secret", "text/plain")

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


class Listener:
    """A TCP port that accepts connections and never answers; records them and when they close."""

    def __init__(self) -> None:
        self.socket = socket.create_server(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.socket.getsockname()[1]}"
        self.accepted = 0
        self.closed = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        with contextlib.suppress(OSError):
            while True:
                connection, _ = self.socket.accept()
                self.accepted += 1
                threading.Thread(target=self._hold, args=(connection,), daemon=True).start()

    def _hold(self, connection: socket.socket) -> None:
        with connection, contextlib.suppress(OSError):
            while connection.recv(65536):
                pass
        self.closed.set()


@pytest.fixture
def listener() -> Iterator[Listener]:
    listener = Listener()
    yield listener
    listener.socket.close()


class Viewer:
    """A viser-like viewer: its page over HTTP, and a WebSocket that echoes, on one port."""

    redirect_to: str | None = None

    def __init__(self) -> None:
        self.closed = threading.Event()
        self.subprotocols: list[str | None] = []
        self.server = websockets.sync.server.serve(
            self._handle,
            "127.0.0.1",
            0,
            subprotocols=[PROTOCOL],  # type: ignore[list-item]
            process_request=self._page,
        )
        self.url = f"http://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _page(self, connection: ServerConnection, request: Request) -> Response | None:
        if request.headers.get("Upgrade", "").lower() == "websocket":
            if self.redirect_to is None:
                return None
            response = connection.respond(302, "")
            response.headers["Location"] = self.redirect_to
            return response
        return connection.respond(200, "<html>viewer</html>")

    def _handle(self, connection: ServerConnection) -> None:
        self.subprotocols.append(connection.subprotocol)
        try:
            for message in connection:
                if message == b"close":
                    connection.close(code=1002, reason="version")
                    break
                connection.send(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        self.closed.set()


@pytest.fixture(scope="module")
def simulation_api() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SimulationApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def viewer() -> Iterator[Viewer]:
    viewer = Viewer()
    yield viewer
    viewer.server.shutdown()


class Payload(BaseModel):
    name: str = "sim-a"
    http_url: str = "http://127.0.0.1:9"


class Studio:
    def __init__(self, state: Simulation) -> None:
        self.current = state
        self.robot = SimpleNamespace(id=ROBOT_ID, name="Sim arm", type="Sim_Follower", payload=Payload())

    def state(self) -> Simulation:
        return self.current


@pytest.fixture
def studio(monkeypatch: pytest.MonkeyPatch, simulation_api: str, viewer: Viewer) -> Iterator[Studio]:
    studio = Studio(
        Simulation(
            status=SimulationStatus.RUNNING,
            project_id=PROJECT_ID,
            robot_id=ROBOT_ID,
            http_url=simulation_api,
            viewer_url=viewer.url,
        )
    )

    class FakeRobotService:
        def __init__(self, *args: Any) -> None:
            pass

        async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Any:
            if robot_id != ROBOT_ID:
                raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))
            return studio.robot

    @contextlib.asynccontextmanager
    async def no_session() -> AsyncIterator[None]:
        yield None

    launch = SimpleNamespace(owner_name=lambda payload: payload.name)
    registry = SimpleNamespace(get_definition=lambda robot_type: SimpleNamespace(simulation=launch))
    monkeypatch.setattr(service_module, "RobotService", FakeRobotService)
    monkeypatch.setattr(proxy_routes, "get_async_db_session_ctx", no_session)
    app.dependency_overrides[get_simulation_manager] = lambda: studio
    app.dependency_overrides[get_robot_catalog_service] = lambda: SimpleNamespace(registry=registry)
    SimulationApi.requests.clear()
    SimulationApi.redirect_to = None
    yield studio
    app.dependency_overrides.clear()


def test_a_camera_frame_is_passed_on(studio: Studio) -> None:
    response = TestClient(app).get(f"{BASE}/http/cameras/wrist/frame.jpg")

    assert response.status_code == 200
    assert response.content == b"frame"
    assert response.headers["content-type"] == "image/jpeg"


def test_a_camera_stream_is_streamed_on(studio: Studio) -> None:
    with TestClient(app).stream("GET", f"{BASE}/http/cameras/wrist/mjpeg") as response:
        body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert body.count(b"--frame") == 3


@pytest.mark.parametrize("path", ["health", "", "cameras/wrist", "cameras/../health/frame.jpg", "cameras/%2E%2E/mjpeg"])
def test_only_camera_frames_and_streams_are_passed_on(studio: Studio, path: str) -> None:
    response = TestClient(app).get(f"{BASE}/http/{path}")

    assert response.status_code == 404
    assert SimulationApi.requests == []


def test_the_viewer_page_is_passed_on_under_a_path_with_a_slash(studio: Studio) -> None:
    client = TestClient(app)

    redirect = client.get(f"{BASE}/viewer", follow_redirects=False)
    page = client.get(f"{BASE}/viewer/")

    assert redirect.status_code == 307
    assert redirect.headers["location"] == f"{BASE}/viewer/"
    assert (page.status_code, page.text) == (200, "<html>viewer</html>")


def test_the_viewer_websocket_passes_messages_both_ways(studio: Studio, viewer: Viewer) -> None:
    # The viser client connects to the page's URL without its trailing slash, with its version.
    with TestClient(app).websocket_connect(f"{BASE}/viewer", subprotocols=[PROTOCOL]) as socket:
        assert socket.accepted_subprotocol == PROTOCOL
        socket.send_bytes(b"\x00\x01binary")
        assert socket.receive_bytes() == b"\x00\x01binary"
        socket.send_text("text")
        assert socket.receive_text() == "text"

    assert viewer.closed.wait(5.0), "the upstream stayed open after the browser left"
    assert viewer.subprotocols == [PROTOCOL]


def test_the_viewer_closing_closes_the_browsers_websocket_with_its_code(studio: Studio) -> None:
    with TestClient(app).websocket_connect(f"{BASE}/viewer", subprotocols=[PROTOCOL]) as socket:
        socket.send_bytes(b"close")
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_bytes()

    assert closed.value.code == 1002


def test_an_unknown_robot_is_not_found(studio: Studio) -> None:
    response = TestClient(app).get(
        f"/api/projects/{PROJECT_ID}/robots/{uuid4()}/simulation/http/cameras/wrist/frame.jpg"
    )

    assert response.status_code == 404


def test_a_simulation_that_does_not_run_is_a_conflict(studio: Studio) -> None:
    studio.current = Simulation()  # stopped; nothing answers at the robot's own address either
    client = TestClient(app)

    assert client.get(f"{BASE}/http/cameras/wrist/frame.jpg").status_code == 409
    assert client.get(f"{BASE}/viewer/").status_code == 409
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(f"{BASE}/viewer") as socket:
        socket.receive_bytes()


def test_an_address_on_another_host_is_not_passed_on_to(studio: Studio) -> None:
    studio.current = studio.current.model_copy(
        update={"http_url": "http://192.0.2.1:8080", "viewer_url": "http://example.com:8081"}
    )
    client = TestClient(app)

    assert client.get(f"{BASE}/http/cameras/wrist/frame.jpg").status_code == 409
    assert client.get(f"{BASE}/viewer/").status_code == 409


def test_an_external_simulation_is_passed_on_from_the_robots_own_address(studio: Studio, simulation_api: str) -> None:
    studio.current = Simulation()
    studio.robot.payload = Payload(http_url=simulation_api)

    response = TestClient(app).get(f"{BASE}/http/cameras/wrist/frame.jpg")

    assert (response.status_code, response.content) == (200, b"frame")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:5000/", "http://127.0.0.1:5000"),
        ("http://localhost:5000", "http://localhost:5000"),
        ("http://[::1]:5000", "http://[::1]:5000"),
        ("http://10.0.0.2:5000", None),
        ("http://example.com", None),
        ("ftp://127.0.0.1:21", None),
        ("http://127.0.0.1:5000/?next=http://example.com", None),
        (None, None),
    ],
)
def test_only_addresses_on_this_machine_are_passed_on_to(url: str | None, expected: str | None) -> None:
    assert local_base_url(url) == expected


def test_the_path_rules() -> None:
    assert is_proxied_http_path("/cameras/wrist_1.left/frame.jpg")
    assert not is_proxied_http_path("/cameras/../frame.jpg")
    assert not is_proxied_http_path("/scene")
    assert is_viewer_path("") and is_viewer_path("assets/index.js")
    assert not is_viewer_path("../secret") and not is_viewer_path("a/%2e%2e/b")


def test_a_redirect_of_the_viewers_websocket_is_not_followed(
    studio: Studio, viewer: Viewer, listener: Listener
) -> None:
    viewer.redirect_to = "ws" + listener.url.removeprefix("http") + "/"

    with pytest.raises(WebSocketDisconnect), TestClient(app).websocket_connect(f"{BASE}/viewer") as socket_:
        socket_.receive_bytes()

    assert listener.accepted == 0


def test_a_redirect_of_a_camera_frame_is_not_followed(studio: Studio, listener: Listener) -> None:
    SimulationApi.redirect_to = f"{listener.url}/cameras/wrist/frame.jpg"

    response = TestClient(app).get(f"{BASE}/http/cameras/moved/frame.jpg", follow_redirects=False)

    assert response.status_code == 502
    assert "location" not in response.headers
    assert listener.accepted == 0


def test_a_simulation_that_never_answers_times_out_and_is_let_go(
    studio: Studio, listener: Listener, monkeypatch: pytest.MonkeyPatch
) -> None:
    import services.simulation_proxy as proxy

    monkeypatch.setattr(proxy, "HEADERS_TIMEOUT_S", 0.5)
    studio.current = studio.current.model_copy(update={"http_url": listener.url})
    started = time.monotonic()

    response = TestClient(app).get(f"{BASE}/http/cameras/wrist/mjpeg")

    assert response.status_code == 504
    assert time.monotonic() - started < 3.0
    assert listener.accepted == 1
    assert listener.closed.wait(2.0), "Studio kept the connection to the simulation open"
