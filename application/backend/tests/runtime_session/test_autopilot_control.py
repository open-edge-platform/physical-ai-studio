"""The autopilot control's freshness and hold, and a recording stopped before its loop ran."""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from runtime_session.autopilot_recording import LEADER, STALE_AFTER_S, AutopilotControl, AutopilotRecording, Freshness
from runtime_session.store import ACTION, STATE, FeatureSpec, Store
from schemas.runtime_session import AutopilotControlConfig, AutopilotRecordingOptions
from tests.workers.fake_robot import FakeRobot

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

JOINTS = ("shoulder_pan.pos", "gripper.pos")


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def store() -> Iterator[Store]:
    store = Store.create(FeatureSpec(JOINTS), uuid.uuid4().hex)
    yield store
    store.unlink()


def test_the_first_sample_seen_is_not_new() -> None:
    clock = _Clock()
    freshness = Freshness(["top"], clock=clock)
    freshness.observe("top", 1.0, camera=True)  # left in the store by the devices before, say
    clock.now += 0.1
    assert freshness.stale(["top"], camera=True) == ["top (never)"]

    freshness.observe("top", 2.0, camera=True)
    assert freshness.stale(["top"], camera=True) == []


def test_a_camera_named_like_the_leader_is_a_stream_of_its_own() -> None:
    clock = _Clock()
    freshness = Freshness(["leader"], clock=clock)
    freshness.observe(LEADER, 1.0)
    freshness.observe(LEADER, 2.0)
    for stamp in (1.0, 2.0, 3.0):  # the camera keeps sending; the leader froze at 2.0
        clock.now += STALE_AFTER_S
        freshness.observe("leader", stamp, camera=True)
        freshness.observe(LEADER, 2.0)
    assert freshness.stale([LEADER]) == [f"{LEADER} ({3 * STALE_AFTER_S:.2f} s)"]
    assert freshness.stale(["leader"], camera=True) == []


def test_a_halt_holds_the_follower_where_it_is_not_where_it_was_sent(store: Store, tmp_path: Path) -> None:
    leader = FakeRobot(str(tmp_path / "leader"), ["shoulder_pan", "gripper"], [5.0, 6.0])
    leader.connect()
    control = AutopilotControl(store, leader, AutopilotControlConfig())
    store.write(ACTION, [5.0, 6.0], timestamp=time.monotonic())
    store.write(STATE, [1.0, 2.0], timestamp=time.monotonic())  # still on its way to the target

    control.halt()

    assert store.read(ACTION).value.tolist() == [1.0, 2.0]


async def test_a_stop_before_the_loop_ran_still_cleans_up() -> None:
    recording = AutopilotRecording(
        bus=SimpleNamespace(),
        store=SimpleNamespace(),
        dataset_id=uuid.uuid4(),
        http_url="http://127.0.0.1:9",
        spec=SimpleNamespace(path="/autopilot"),
        payload=None,
        options=AutopilotRecordingOptions(),
        emit=lambda: None,
    )
    recording.start(lambda _recording: None)

    await recording.stop("Stopped.")  # before the loop's task took its first step

    assert recording.state.phase == "stopped" and recording.state.message == "Stopped."
    assert recording.client._async.is_closed
