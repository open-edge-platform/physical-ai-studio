from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

import pytest

from runtime_session.controls import TeleopControl, build_control
from runtime_session.store import ACTION, STATE, FeatureSpec, Store
from schemas.runtime_session import TeleopControlConfig
from tests.workers.fake_robot import FakeRobot

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

JOINTS = ("shoulder_pan.pos", "gripper.pos")


@pytest.fixture
def store() -> Iterator[Store]:
    store = Store.create(FeatureSpec(JOINTS), uuid.uuid4().hex)
    store.write(STATE, [0.0, 10.0], timestamp=time.monotonic())
    yield store
    store.unlink()


def _leader(tmp_path: Path, joints: list[str] | None = None) -> FakeRobot:
    leader = FakeRobot(str(tmp_path / "leader"), joints or ["shoulder_pan", "gripper"], [5.0, 6.0])
    leader.connect()
    return leader


def test_teleop_commands_what_the_leader_measures(store: Store, tmp_path: Path) -> None:
    control = TeleopControl(store, _leader(tmp_path), TeleopControlConfig(grace_s=0))
    control.on_start()
    control.on_tick()
    action = store.read(ACTION)
    assert action.value.tolist() == [5.0, 6.0]
    assert action.timestamp <= time.monotonic() + 1 / control.hz  # due by the next reading


def test_teleop_commands_the_leader_due_at_the_end_of_the_grace_period(store: Store, tmp_path: Path) -> None:
    control = TeleopControl(store, _leader(tmp_path), TeleopControlConfig(grace_s=1.0))
    control.on_start()
    control.on_tick()
    # The leader's position, due once the grace period is over: the robot worker closes the gap over it.
    action = store.read(ACTION)
    assert action.value.tolist() == [5.0, 6.0]
    assert action.timestamp == pytest.approx(control.started_at + 1.0)

    control.started_at = time.monotonic() - 2.0  # past it: due one reading from now
    control.on_tick()
    assert store.read(ACTION).timestamp == pytest.approx(time.monotonic() + 1 / control.hz, abs=0.01)


async def test_every_start_of_teleop_has_its_own_grace_period(store: Store, tmp_path: Path) -> None:
    leader = _leader(tmp_path)
    for _ in range(2):
        async with TeleopControl(store, leader, TeleopControlConfig(grace_s=5.0)):
            await asyncio.sleep(0.05)
        # A fresh control is due at the end of its own grace period again.
        assert store.read(ACTION).timestamp > time.monotonic() + 4.0


def test_teleop_refuses_a_leader_with_other_joints(store: Store, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match"):
        TeleopControl(store, _leader(tmp_path, ["elbow", "gripper"]), TeleopControlConfig())


def test_teleop_needs_a_leader(store: Store) -> None:
    with pytest.raises(ValueError, match="leader arm"):
        build_control(TeleopControlConfig(), store, None)


async def test_a_control_runs_as_a_thread_until_stopped(store: Store, tmp_path: Path) -> None:
    async with TeleopControl(store, _leader(tmp_path), TeleopControlConfig()):
        await asyncio.sleep(0.1)
    first = store.read(ACTION).timestamp
    await asyncio.sleep(0.05)
    assert store.read(ACTION).timestamp == first  # stopped writing
