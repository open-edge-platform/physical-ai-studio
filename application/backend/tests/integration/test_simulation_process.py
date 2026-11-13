# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A real MuJoCo simulation started by the manager stops when its Studio process is killed."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import httpx
import pytest

pytest.importorskip("physicalai_studio_plugin.simulation")
pytest.importorskip("physicalai_mujoco_plugin")

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SRC = Path(__file__).resolve().parents[2] / "src"

# Plays Studio: starts the simulation through the manager, prints the running state, then waits to be killed.
FAKE_STUDIO = textwrap.dedent(
    """
    import asyncio, json, sys
    from types import SimpleNamespace
    from uuid import uuid4

    from robots.catalog.registry import RobotCatalogRegistry
    from schemas.simulation import SimulationStatus, StartSimulationRequest
    from services.simulation_manager import SimulationManager

    class Robots:
        def __init__(self, robot):
            self.robot = robot

        async def get(self, project_id, robot_id):
            return self.robot

        async def set_http_url(self, project_id, robot_id, http_url):
            pass

    async def main():
        registry = RobotCatalogRegistry()
        definition = registry.get_definition("MuJoCo_SO101_Follower")
        robot = SimpleNamespace(name="Sim", type=definition.type, payload=definition.robot_payload(name=sys.argv[1]))
        manager = SimulationManager(registry, Robots(robot))
        await manager.start(StartSimulationRequest(project_id=uuid4(), robot_id=uuid4(), scene="single_pick_place"))
        async with manager.subscribe() as queue:
            while (state := await queue.get()).status not in {SimulationStatus.RUNNING, SimulationStatus.FAILED}:
                pass
        print(state.model_dump_json(), flush=True)
        await asyncio.Event().wait()

    asyncio.run(main())
    """
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _answers(url: str) -> bool:
    try:
        httpx.get(url + "/", timeout=1.0)
    except httpx.HTTPError:
        return False
    return True


def test_killing_studio_stops_the_simulation(tmp_path: Path) -> None:
    script = tmp_path / "fake_studio.py"
    script.write_text(FAKE_STUDIO)
    owner_name = f"studio-test-{uuid.uuid4().hex[:8]}"
    studio = subprocess.Popen(
        [sys.executable, str(script), owner_name],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    try:
        assert studio.stdout is not None
        state = json.loads(studio.stdout.readline())
        assert state["status"] == "running", state
        start_pid = subprocess.run(
            ["pgrep", "-f", "--", f"--name={owner_name}"], capture_output=True, text=True, check=True
        ).stdout.split()
        assert start_pid and _answers(state["http_url"])

        studio.send_signal(signal.SIGKILL)
        studio.wait(10)

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and (
            any(_alive(int(pid)) for pid in start_pid) or _answers(state["http_url"])
        ):
            time.sleep(0.2)
        assert not any(_alive(int(pid)) for pid in start_pid), "the simulation's start process outlived Studio"
        assert not _answers(state["http_url"]), "the simulation owner outlived Studio"
    finally:
        if studio.poll() is None:
            studio.kill()
