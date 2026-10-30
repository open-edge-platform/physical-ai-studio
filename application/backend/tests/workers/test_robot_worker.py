from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pytest

from exceptions import RobotDeviceAlreadyOwnedError
from runtime_session.store import ACTION, STATE, FeatureSpec, Store
from workers.base import WorkerStartupError
from workers.robot_worker import RobotRecipe, RobotWorker, claim_robot

from .fake_robot import FakeRobot, sent_actions

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

JOINTS = ["shoulder_pan", "gripper"]
HZ = 50.0


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "actions.jsonl"


def _recipe(log: Path, *, fail_connect: bool = False) -> RobotRecipe:
    return RobotRecipe.from_driver(uuid.uuid4(), FakeRobot(str(log), JOINTS, [1.0, 2.0], fail_connect=fail_connect))


@pytest.fixture
def store(log: Path) -> Iterator[Store]:
    store = Store.create(_recipe(log).spec, uuid.uuid4().hex)
    yield store
    store.unlink()


async def _until(condition: Callable[[], bool], within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        await asyncio.sleep(0.02)


def test_recipe_names_joints_like_the_dataset(log: Path) -> None:
    recipe = _recipe(log)
    assert recipe.joint_names == ("shoulder_pan.pos", "gripper.pos")
    assert recipe.device_ids == (f"fake:{log}",)


async def test_observe_only_measures_and_never_sends(log: Path, store: Store) -> None:
    recipe = _recipe(log)
    async with RobotWorker(recipe, store.name, store.spec, hz=HZ, observe_only=True):
        await _until(lambda: store.read(STATE) is not None)
        await asyncio.sleep(0.1)
    np.testing.assert_array_equal(store.read(STATE).value, [1.0, 2.0])
    assert store.read(ACTION) is None
    assert sent_actions(log) == []


async def test_holds_still_then_follows_the_action(log: Path, store: Store) -> None:
    recipe = _recipe(log)
    async with RobotWorker(recipe, store.name, store.spec, hz=HZ):
        # Seeded with the measured position, so the first commands hold still.
        await _until(lambda: len(sent_actions(log)) >= 2)
        assert sent_actions(log)[0] == {"action": [1.0, 2.0], "goal_time": pytest.approx(1 / HZ)}

        before = len(sent_actions(log))
        sent_at = time.monotonic()
        store.write(ACTION, [5.0, 6.0], timestamp=sent_at + 0.5)
        await _until(lambda: sent_actions(log)[-1]["action"] == [5.0, 6.0])
        arrived_after = time.monotonic() - sent_at
        await _until(lambda: store.read(STATE).value.tolist() == [5.0, 6.0])
    approach = [sent["action"][0] for sent in sent_actions(log)[before:]]
    # A future action is stepped towards, never past, and reached around its stamp rather than at once.
    assert len({round(value, 6) for value in approach}) > 10
    assert approach == sorted(approach)
    assert max(approach) == 5.0
    assert 0.4 < arrived_after < 1.0
    assert all(sent["goal_time"] == pytest.approx(1 / HZ) for sent in sent_actions(log))


async def test_a_robot_with_other_joints_fails_to_start(log: Path) -> None:
    recipe = _recipe(log)
    store = Store.create(FeatureSpec(("elbow.pos", "gripper.pos")), uuid.uuid4().hex)
    try:
        with pytest.raises(WorkerStartupError, match="session expects"):
            async with RobotWorker(recipe, store.name, store.spec, hz=HZ):
                pass
    finally:
        store.unlink()
    assert sent_actions(log) == []


async def test_a_connect_failure_fails_startup(log: Path, store: Store) -> None:
    recipe = _recipe(log, fail_connect=True)
    with pytest.raises(WorkerStartupError, match="Failed to open serial port"):
        async with RobotWorker(recipe, store.name, store.spec, hz=HZ):
            pass


def test_a_claimed_robot_cannot_be_claimed_again(log: Path) -> None:
    recipe = _recipe(log)
    with claim_robot(recipe):
        with pytest.raises(RobotDeviceAlreadyOwnedError), claim_robot(recipe):
            pass
        # Another robot entry for the same device is refused too.
        with pytest.raises(RobotDeviceAlreadyOwnedError, match="fake:"), claim_robot(_recipe(log)):
            pass
    with claim_robot(recipe):
        pass
