from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pytest

from runtime_session.controls import MoveControl, ReplayControl, TeleopControl, build_control
from runtime_session.store import ACTION, STATE, FeatureSpec, Store
from schemas.dataset import Episode
from schemas.runtime_session import (
    MoveControlConfig,
    ReplayControlConfig,
    SetTeleopSlowdownCommand,
    TeleopControlConfig,
)
from tests.workers.fake_robot import FakeRobot
from workers.base import WorkerStartupError

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


def test_relative_teleop_slows_the_arm_down_from_where_it_is_and_leaves_the_gripper_absolute(
    store: Store, tmp_path: Path
) -> None:
    leader = _leader(tmp_path)
    control = TeleopControl(store, leader, TeleopControlConfig(slowdown=2.0))
    control.on_start()
    control.on_tick()
    # The arm starts where the follower is; the gripper follows the leader.
    assert store.read(ACTION).value.tolist() == [0.0, 6.0]

    leader._position = np.array([7.0, 4.0])  # moved by [2, -2]
    control.on_tick()
    assert store.read(ACTION).value.tolist() == [1.0, 4.0]


def test_switching_teleop_mode_moves_on_from_the_last_command(store: Store, tmp_path: Path) -> None:
    leader = _leader(tmp_path)
    control = TeleopControl(store, leader, TeleopControlConfig(grace_s=1.0))
    control.on_start()
    control.started_at = time.monotonic() - 2.0  # past the grace period: on the leader
    control.on_tick()
    assert store.read(ACTION).value.tolist() == [5.0, 6.0]

    control.set_slowdown(SetTeleopSlowdownCommand(request_id="1", slowdown=2.0))
    assert control.config.slowdown == 2.0
    leader._position = np.array([7.0, 8.0])
    control.on_tick()
    assert store.read(ACTION).value.tolist() == [6.0, 8.0]  # the arm at half speed, the gripper absolute

    control.set_slowdown(SetTeleopSlowdownCommand(request_id="2", slowdown=None))
    control.on_tick()
    # Back to absolute, the leader due a grace period from now rather than at once.
    action = store.read(ACTION)
    assert action.value.tolist() == [7.0, 8.0]
    assert action.timestamp > time.monotonic() + 0.9


def test_relative_teleop_needs_the_followers_position(tmp_path: Path) -> None:
    store = Store.create(FeatureSpec(JOINTS), uuid.uuid4().hex)
    try:
        control = TeleopControl(store, _leader(tmp_path), TeleopControlConfig(slowdown=2.0))
        with pytest.raises(RuntimeError, match="follower's position"):
            control.on_start()
    finally:
        store.unlink()


def test_teleop_refuses_a_leader_with_other_joints(store: Store, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match"):
        TeleopControl(store, _leader(tmp_path, ["elbow", "gripper"]), TeleopControlConfig())


def test_teleop_needs_a_leader(store: Store) -> None:
    with pytest.raises(ValueError, match="leader arm"):
        build_control(TeleopControlConfig(), store, None)


def test_move_interpolates_then_holds_the_target(store: Store) -> None:
    control = MoveControl(store, MoveControlConfig(position={"shoulder_pan.pos": 4.0}, duration_s=0.2))
    control.on_start()
    control.started_at = time.monotonic() - 0.1  # halfway
    control.on_tick()
    np.testing.assert_allclose(store.read(ACTION).value, [2.0, 10.0], atol=0.2)
    control.started_at = time.monotonic() - 1.0  # long done
    control.on_tick()
    # The joint left out stays where it was.
    assert store.read(ACTION).value.tolist() == [4.0, 10.0]


async def test_move_fails_to_start_before_the_follower_is_measured() -> None:
    unmeasured = Store.create(FeatureSpec(JOINTS), uuid.uuid4().hex)
    try:
        with pytest.raises(WorkerStartupError, match="not been measured"):
            async with build_control(MoveControlConfig(position={}, duration_s=1), unmeasured, None):
                pass
    finally:
        unmeasured.unlink()


def test_move_refuses_unknown_joints(store: Store) -> None:
    with pytest.raises(ValueError, match="Unknown joints"):
        build_control(MoveControlConfig(position={"elbow.pos": 1.0}, duration_s=1), store, None)


async def test_a_control_runs_as_a_thread_until_stopped(store: Store, tmp_path: Path) -> None:
    async with TeleopControl(store, _leader(tmp_path), TeleopControlConfig()):
        await asyncio.sleep(0.1)
    first = store.read(ACTION).timestamp
    await asyncio.sleep(0.05)
    assert store.read(ACTION).timestamp == first  # stopped writing


def _episode(actions: list[list[float]], keys: tuple[str, ...] = JOINTS) -> Episode:
    return Episode(
        episode_index=0, length=len(actions), fps=10, tasks=[], actions=actions, action_keys=list(keys), videos={}
    )


def test_replay_moves_to_the_first_frame_then_plays_at_the_recorded_rate(store: Store) -> None:
    control = ReplayControl(
        store, _episode([[4.0, 0.0], [5.0, 1.0], [6.0, 2.0]]), ReplayControlConfig(episode_index=0, grace_s=1.0)
    )
    control.on_start()
    control.on_tick()
    # The first frame, due once the grace period is over: the robot worker closes the gap over it.
    action = store.read(ACTION)
    assert action.value.tolist() == [4.0, 0.0]
    assert action.timestamp == pytest.approx(control.started_at + 1.0)

    control.started_at = time.monotonic() - 1.15  # 0.15 s into the episode, at 10 fps: frame 1
    control.on_tick()
    action = store.read(ACTION)
    assert action.value.tolist() == [5.0, 1.0]
    assert action.timestamp == pytest.approx(control.started_at + 1.0 + 0.1)

    control.started_at = time.monotonic() - 10.0  # long done: holds the last frame
    control.on_tick()
    assert store.read(ACTION).value.tolist() == [6.0, 2.0]


def test_replay_refuses_an_episode_of_other_joints(store: Store) -> None:
    with pytest.raises(ValueError, match="drives joints"):
        ReplayControl(store, _episode([[1.0]], ("elbow.pos",)), ReplayControlConfig(episode_index=0))


def test_replay_needs_the_episode(store: Store) -> None:
    with pytest.raises(ValueError, match="not in the dataset"):
        build_control(ReplayControlConfig(episode_index=3), store, None)
