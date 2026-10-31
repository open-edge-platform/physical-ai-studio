# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Setup endpoint validation and diagnostics with a fake public session."""

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from physicalai_so101_plugin import SO101Calibration
from physicalai_so101_plugin.constants import SO101_JOINT_ORDER

from api.robot_setup import router
from workers.robots import so101_setup_worker as worker_module


class FakeSession:
    def __init__(self, port, *, role):
        self.port, self.role = port, role
        self.servo_ids = dict(zip(SO101_JOINT_ORDER, range(1, 7), strict=True))
        self.closed = False
        self.torque_disabled = False

    def connect(self):
        pass

    def ping(self, name):
        return 777

    def read_voltage(self, name):
        return 120

    def disable_torque(self):
        self.torque_disabled = True

    def read_calibration(self):
        return SO101Calibration.from_dict(
            {
                name: {"id": servo_id, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
                for name, servo_id in self.servo_ids.items()
            }
        )

    def disconnect(self):
        self.closed = True


@pytest.fixture
def app():
    app = FastAPI()
    app.state.robot_manager = object()
    app.include_router(router)
    return app


@pytest.mark.parametrize("query", ["robot_type=Unsupported&connection_string=fake", "robot_type=SO101_Follower"])
def test_setup_endpoint_rejects_invalid_input(app, query):
    with (
        TestClient(app) as client,
        client.websocket_connect(f"/api/projects/{uuid4()}/robots/setup/ws?{query}") as websocket,
    ):
        assert websocket.receive_json()["error_code"] == "invalid_config"
        assert websocket.receive()["code"] == 1008


def test_setup_endpoint_diagnostics_pong_and_disconnect(app, monkeypatch):
    session = FakeSession("fake", role="follower")
    monkeypatch.setattr(worker_module, "SO101SetupSession", lambda *args, **kwargs: session)

    async def resolve_port(*args):
        return "fake"

    monkeypatch.setattr(worker_module, "find_so101_port", resolve_port)
    with (
        TestClient(app) as client,
        client.websocket_connect(
            f"/api/projects/{uuid4()}/robots/setup/ws?robot_type=SO101_Follower&connection_string=fake"
        ) as websocket,
    ):
        messages = [websocket.receive_json() for _ in range(6)]
        assert [message["event"] for message in messages] == [
            "status",
            "status",
            "status",
            "voltage_result",
            "status",
            "motor_probe_result",
        ]
        assert messages[-1]["all_motors_ok"]
        assert not messages[-1]["calibration"]["all_calibrated"]
        assert session.torque_disabled
        websocket.send_json({"command": "ping"})
        assert websocket.receive_json() == {"event": "pong"}
    assert session.closed
