from __future__ import annotations

import multiprocessing as mp
import os
import uuid
from multiprocessing import shared_memory

import numpy as np
import pytest

from runtime_session.store import (
    ACTION,
    BLOCK_PREFIX,
    STATE,
    FeatureSpec,
    FeatureSpecMismatchError,
    ImageSpec,
    Store,
    sweep_stale_blocks,
)

JOINTS = ("shoulder_pan.pos", "elbow_flex.pos", "gripper.pos")
SPEC = FeatureSpec(JOINTS, {"wrist": ImageSpec(4, 6), "top": ImageSpec(2, 3)})


@pytest.fixture
def store():
    store = Store.create(SPEC, f"test-{uuid.uuid4().hex}")
    yield store
    store.unlink()


def test_every_feature_starts_unwritten(store: Store) -> None:
    for key in (STATE, ACTION):
        assert store.read(key) is None
    assert store.read_observation("wrist") is None


def test_vectors_round_trip_with_names_and_timestamp(store: Store) -> None:
    store.write(ACTION, [0.0, 1.5, -2.0], timestamp=12.5)

    sample = store.read(ACTION)
    assert sample is not None
    np.testing.assert_array_equal(sample.value, [0.0, 1.5, -2.0])
    assert sample.names == JOINTS
    assert sample.timestamp == 12.5
    # A zero position is a real value, not "unwritten".
    store.write(STATE, [0.0, 0.0, 0.0], timestamp=1.0)
    assert store.read(STATE) is not None
    assert store.read(ACTION) is not None


def test_frames_round_trip_per_camera(store: Store) -> None:
    wrist = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    store.write_observation("wrist", wrist, timestamp=3.0)

    sample = store.read_observation("wrist")
    assert sample is not None
    np.testing.assert_array_equal(sample.value, wrist)
    assert sample.names == ("wrist",)
    assert store.read_observation("top") is None


def test_read_returns_a_copy(store: Store) -> None:
    store.write(STATE, [1.0, 2.0, 3.0], timestamp=1.0)
    first = store.read(STATE)
    store.write(STATE, [4.0, 5.0, 6.0], timestamp=2.0)

    assert first is not None
    np.testing.assert_array_equal(first.value, [1.0, 2.0, 3.0])


def test_wrong_shape_is_refused(store: Store) -> None:
    with pytest.raises(ValueError, match="shape"):
        store.write(STATE, [1.0, 2.0], timestamp=1.0)
    with pytest.raises(ValueError, match="shape"):
        store.write_observation("wrist", np.zeros((6, 4, 3), np.uint8), timestamp=1.0)


def test_unknown_camera_is_refused(store: Store) -> None:
    with pytest.raises(KeyError):
        store.write_observation("side", np.zeros((4, 6, 3), np.uint8), timestamp=1.0)


def _write_from_child(name: str) -> None:
    store = Store.attach(SPEC, name)
    store.write(STATE, [7.0, 8.0, 9.0], timestamp=42.0)
    store.write_observation("top", np.full((2, 3, 3), 5, np.uint8), timestamp=43.0)
    store.close()


def test_another_process_attaches_and_writes(store: Store) -> None:
    child = mp.get_context("spawn").Process(target=_write_from_child, args=(store.name,))
    child.start()
    child.join(timeout=30)
    assert child.exitcode == 0

    state = store.read(STATE)
    top = store.read_observation("top")
    assert state is not None
    assert top is not None
    np.testing.assert_array_equal(state.value, [7.0, 8.0, 9.0])
    assert state.timestamp == 42.0
    assert np.all(top.value == 5)


def test_attach_refuses_a_block_too_small_for_the_spec(store: Store) -> None:
    bigger = FeatureSpec(JOINTS, {**SPEC.observations, "side": ImageSpec(480, 640)})
    with pytest.raises(ValueError, match="needs"):
        Store.attach(bigger, store.name)


def test_compatible_when_newcomer_needs_a_subset_of_cameras() -> None:
    model = FeatureSpec(JOINTS, {"wrist": ImageSpec(4, 6)})
    assert SPEC.check_compatible(model) == []


def test_mismatches_name_each_difference() -> None:
    dataset = FeatureSpec(
        ("shoulder_pan.pos", "gripper.pos"),
        {"wrist": ImageSpec(8, 6), "side": ImageSpec(4, 6)},
    )

    with pytest.raises(FeatureSpecMismatchError) as error:
        SPEC.require_compatible(dataset)

    messages = [str(mismatch) for mismatch in error.value.mismatches]
    assert messages == [
        "joint names: expected shoulder_pan.pos, elbow_flex.pos, gripper.pos, got shoulder_pan.pos, gripper.pos",
        "camera 'wrist': expected 4x6x3, got 8x6x3",
        "camera 'side': expected no such camera, got 4x6x3",
    ]


def test_sweep_removes_only_blocks_whose_creator_is_gone(store: Store) -> None:
    gone = mp.get_context("spawn").Process(target=int)
    gone.start()
    gone.join()
    # A block a dead backend left, and one from before blocks carried their creator's pid.
    orphans = [f"{BLOCK_PREFIX}{gone.pid}-{uuid.uuid4().hex}", f"{BLOCK_PREFIX}{uuid.uuid4().hex}"]
    for name in orphans:
        shared_memory.SharedMemory(name=name, create=True, size=8, track=False).close()

    removed = sweep_stale_blocks()

    assert set(orphans) <= set(removed)
    # This process's block is live: another backend starting (or a test run) must leave it alone.
    assert store.name not in removed
    assert store.name.startswith(f"{BLOCK_PREFIX}{os.getpid()}-")
    Store.attach(SPEC, store.name).close()
