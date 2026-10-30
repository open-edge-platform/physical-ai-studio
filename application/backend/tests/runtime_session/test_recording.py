from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from exceptions import IncompatiblePartsError, RuntimeSessionCommandError
from runtime_session.bus import BusRequestError, NoResponderError, command_key
from runtime_session.dataset import DEFAULT_RECORDING_FPS, dataset_fps, dataset_spec
from runtime_session.session import RuntimeSession
from runtime_session.store import FeatureSpecMismatchError, ImageSpec
from schemas.dataset import Dataset
from schemas.runtime_session import DiscardEpisodeCommand, ReplayControlConfig, SaveEpisodeCommand, StartEpisodeCommand
from tests.workers.fake_dataset import FakeDatasetWorker, recording_log
from workers.dataset_worker import RecordingStatus

from .fixtures import make_environment, next_status, subscribe_statuses


def _dataset() -> Dataset:
    dataset = Dataset(id=uuid4(), name="Blocks", default_task="pick", project_id=uuid4(), environment_id=uuid4())
    Path(dataset.path).mkdir(parents=True)
    return dataset


def _write_info(dataset: Dataset, joints: list[str], cameras: dict[str, list[int]], fps: int = 10) -> None:
    features: dict = {
        "observation.state": {"dtype": "float32", "shape": [len(joints)], "names": joints},
        "action": {"dtype": "float32", "shape": [len(joints)], "names": joints},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    for key, shape in cameras.items():
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    meta = Path(dataset.path) / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"fps": fps, "features": features}))


@pytest.fixture
def fake_worker():
    with patch("runtime_session.dataset.DatasetWorker", FakeDatasetWorker):
        yield


async def _until(condition, within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        await asyncio.sleep(0.02)


async def _start(session: RuntimeSession, task: str) -> RecordingStatus:
    command = StartEpisodeCommand(request_id="", task=task)
    return await session.bus.request(command_key("start_episode"), command, RecordingStatus, timeout=10)


async def _save(session: RuntimeSession) -> RecordingStatus:
    command = SaveEpisodeCommand(request_id="")
    return await session.bus.request(command_key("save_episode"), command, RecordingStatus, timeout=60)


async def _discard(session: RuntimeSession) -> RecordingStatus:
    command = DiscardEpisodeCommand(request_id="")
    return await session.bus.request(command_key("discard_episode"), command, RecordingStatus, timeout=10)


def test_a_new_dataset_has_no_shape_and_the_default_rate() -> None:
    dataset = _dataset()
    assert dataset_spec(dataset) is None
    assert dataset_fps(dataset) == DEFAULT_RECORDING_FPS
    assert dataset_fps(dataset, 60) == 60  # a new dataset takes the rate asked for


def test_a_recorded_dataset_reads_its_joints_cameras_and_rate() -> None:
    dataset = _dataset()
    _write_info(dataset, ["shoulder_pan.pos", "gripper.pos"], {"cam 0": [120, 160, 3]}, fps=10)
    spec = dataset_spec(dataset)
    assert spec is not None
    assert spec.joint_names == ("shoulder_pan.pos", "gripper.pos")
    assert spec.observations == {"cam 0": ImageSpec(120, 160, 3)}
    assert dataset_fps(dataset) == 10
    assert dataset_fps(dataset, 60) == 10  # a recorded one keeps its own


async def test_a_new_dataset_records_at_the_rate_it_is_loaded_with(new_session, fake_cameras, fake_worker) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(_dataset(), fps=60)
    assert session.state().dataset.fps == 60
    assert not session.state().dataset.fps_fixed
    await session.stop()


async def test_record_save_and_discard_episodes(new_session, fake_cameras, fake_worker) -> None:
    dataset = _dataset()
    log = Path(dataset.path) / "recording.jsonl"
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(dataset)
    assert session.state().dataset.fps == DEFAULT_RECORDING_FPS

    assert (await _start(session, "pick the block")).is_recording
    # The worker publishes its status; the session shows it.
    await _until(lambda: session.state().dataset.is_recording)
    await asyncio.sleep(0.3)
    assert (await _save(session)).episodes_recorded == 1
    await _until(lambda: session.state().dataset.episodes_recorded == 1)
    assert not session.state().dataset.is_recording

    await _start(session, "pick the block")
    await asyncio.sleep(0.1)
    assert (await _discard(session)).episodes_recorded == 1

    await _start(session, "left open")
    await asyncio.sleep(0.1)
    await session.unload_dataset()
    assert session.state().dataset is None

    events = recording_log(log)
    first = events[0]
    assert first["event"] == "first_frame" and first["task"] == "pick the block"
    assert first["joints"] == {"shoulder_pan.pos": 1.0, "gripper.pos": 2.0}
    assert first["action"] == {"shoulder_pan.pos": 1.0, "gripper.pos": 2.0}
    assert first["cameras"] == {"cam 0": [120, 160, 3]}
    saved = next(event for event in events if event["event"] == "save")
    assert saved["frames"] > 3  # ~0.3 s at 30 fps
    # The episode left open is discarded on unload, then the recording is finalized once.
    assert [event["event"] for event in events if event["event"] in {"discard", "teardown"}] == [
        "discard",
        "discard",
        "teardown",
    ]


async def test_a_too_short_episode_is_discarded_not_saved(new_session, fake_cameras, fake_worker) -> None:
    dataset = _dataset()
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(dataset)
    await _start(session, "pick")
    with pytest.raises(BusRequestError, match="too short") as error:
        await _save(session)  # at once: at most one frame in
    assert error.value.error_code == "episode_too_short"
    await _until(lambda: not session.state().dataset.is_recording)
    assert session.state().dataset.episodes_recorded == 0
    assert [event["event"] for event in recording_log(Path(dataset.path) / "recording.jsonl")] in (
        ["discard"],
        ["first_frame", "discard"],
    )


async def test_a_failed_save_clears_the_episode_so_the_next_records(new_session, fake_cameras, fake_worker) -> None:
    dataset = _dataset()
    fail = Path(dataset.path) / "fail_save"
    fail.touch()
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(dataset)
    await _start(session, "pick")
    await asyncio.sleep(0.2)
    with pytest.raises(BusRequestError, match="disk is full"):
        await _save(session)
    fail.unlink()
    await _start(session, "pick")
    await asyncio.sleep(0.2)
    assert (await _save(session)).episodes_recorded == 1
    events = [event["event"] for event in recording_log(Path(dataset.path) / "recording.jsonl")]
    assert events[:2] == ["first_frame", "discard"]
    assert "save" in events


async def test_a_frame_that_fails_to_record_loses_its_episode_not_the_recording(
    new_session, fake_cameras, fake_worker
) -> None:
    dataset = _dataset()
    log, fail = Path(dataset.path) / "recording.jsonl", Path(dataset.path) / "fail_frame"
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(dataset)
    fail.touch()
    await _start(session, "pick")
    await _until(lambda: "discard" in [event["event"] for event in recording_log(log)])
    await _until(lambda: not session.state().dataset.is_recording)
    fail.unlink()
    await _start(session, "pick")
    await asyncio.sleep(0.2)
    assert (await _save(session)).episodes_recorded == 1
    await session.unload_dataset()
    assert recording_log(log)[-1]["event"] == "teardown"


async def test_closing_finalizes_even_when_the_open_episode_cannot_be_discarded(
    new_session, fake_cameras, fake_worker
) -> None:
    dataset = _dataset()
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(dataset)
    await _start(session, "pick")
    await asyncio.sleep(0.2)
    (Path(dataset.path) / "fail_discard").touch()
    await session.unload_dataset()
    assert recording_log(Path(dataset.path) / "recording.jsonl")[-1]["event"] == "teardown"


async def test_unloading_the_environment_keeps_the_dataset_until_the_session_closes(
    new_session, fake_cameras, fake_worker
) -> None:
    dataset = _dataset()
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(dataset)
    await session.unload_environment()
    assert session.state().dataset is not None
    log = Path(dataset.path) / "recording.jsonl"
    assert "teardown" not in [event["event"] for event in recording_log(log)]
    await session.stop()
    assert recording_log(log)[-1]["event"] == "teardown"


async def test_recording_commands_are_refused_out_of_order(new_session, fake_cameras, fake_worker) -> None:
    session = await new_session()
    with pytest.raises(RuntimeSessionCommandError, match="Load an environment"):
        await session.load_dataset(_dataset())
    await session.load_environment(make_environment())
    with pytest.raises(NoResponderError):  # nothing records until a dataset is loaded
        await _start(session, "pick")
    await session.load_dataset(_dataset())
    with pytest.raises(BusRequestError, match="No episode") as error:
        await _save(session)
    assert error.value.error_code == "no_episode"
    with pytest.raises(BusRequestError, match="task is required"):
        await _start(session, "")
    await _start(session, "pick")
    with pytest.raises(BusRequestError, match="already being recorded"):
        await _start(session, "pick")


async def test_a_dataset_from_another_rig_is_refused(new_session, fake_cameras, fake_worker) -> None:
    dataset = _dataset()
    _write_info(dataset, ["elbow.pos", "gripper.pos"], {"wrist": [120, 160, 3]})
    session = await new_session()
    await session.load_environment(make_environment())
    with pytest.raises(FeatureSpecMismatchError) as mismatch:
        await session.load_dataset(dataset)
    assert {item.feature for item in mismatch.value.mismatches} == {"joint names", "camera 'wrist'"}
    assert session.state().dataset is None


async def test_a_real_recording_lands_in_the_dataset(new_session, fake_cameras) -> None:
    """Through lerobot: two saved episodes are copied back into a new dataset when it is unloaded."""
    dataset = _dataset()
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(dataset)
    for _ in range(2):
        await _start(session, "pick the block")
        await asyncio.sleep(0.4)
        await _save(session)
    await session.unload_dataset()

    info = json.loads((Path(dataset.path) / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["fps"] == DEFAULT_RECORDING_FPS
    spec = dataset_spec(dataset)
    assert spec is not None
    assert spec.joint_names == ("shoulder_pan.pos", "gripper.pos")
    assert spec.observations == {"cam 0": ImageSpec(120, 160, 3)}

    # Loading it again checks it against the environment, then appends.
    await session.load_dataset(dataset)
    await _start(session, "pick the block")
    await asyncio.sleep(0.4)
    await _save(session)
    await session.unload_dataset()
    assert json.loads((Path(dataset.path) / "meta" / "info.json").read_text())["total_episodes"] == 3


async def test_swapping_the_environment_keeps_a_dataset_that_fits(new_session, fake_cameras, fake_worker) -> None:
    dataset = _dataset()
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(dataset)
    # Fewer cameras: another store, so the dataset is finished and loaded again on it.
    await session.load_environment(make_environment(cameras=0))  # the same rig again: the dataset still fits
    assert session.state().dataset is not None
    events = [event["event"] for event in recording_log(Path(dataset.path) / "recording.jsonl")]
    assert events == ["teardown"]  # finished (copied back) once, before the reload


async def test_swapping_to_an_environment_a_dataset_does_not_fit_asks_first(
    new_session, fake_cameras, fake_worker
) -> None:
    dataset = _dataset()
    _write_info(dataset, ["shoulder_pan.pos", "gripper.pos"], {"cam 0": [120, 160, 3]})
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(dataset)

    with pytest.raises(IncompatiblePartsError) as refused:
        await session.load_environment(make_environment(cameras=0))
    assert refused.value.error_code == "incompatible_parts"
    [part] = refused.value.data["parts"]
    assert part["part"] == "dataset" and part["name"] == "Blocks"
    assert any("cam 0" in reason for reason in part["reasons"])
    # Nothing changed.
    assert session.environment is not None and len(session.environment.cameras) == 1
    assert session.state().dataset is not None

    await session.load_environment(make_environment(cameras=0), unload_incompatible=True)
    assert session.environment is not None and session.environment.cameras == []
    assert session.state().dataset is None


async def test_episodes_recorded_this_session_give_a_new_dataset_its_shape(
    new_session, fake_cameras, fake_worker
) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(_dataset())
    await _start(session, "pick")
    await asyncio.sleep(0.2)
    await _save(session)
    await _until(lambda: session.state().dataset.episodes_recorded == 1)
    # Nothing is on disk yet, but the saved episode has this environment's camera.
    with pytest.raises(IncompatiblePartsError):
        await session.load_environment(make_environment(cameras=0))


async def test_loading_another_dataset_finishes_the_first(new_session, fake_cameras, fake_worker) -> None:
    first, second = _dataset(), _dataset()
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_dataset(first)
    await session.load_dataset(second)
    assert session.state().dataset.dataset_id == second.id
    assert recording_log(Path(first.path) / "recording.jsonl")[-1]["event"] == "teardown"


async def test_a_failed_environment_keeps_its_dataset_loaded_but_not_recording(
    new_session, fake_cameras, fake_worker
) -> None:
    dataset = _dataset()
    session = await new_session()
    environment = make_environment()
    await session.load_environment(environment)
    await session.load_dataset(dataset)
    store = session.store
    await _start(session, "pick")
    await _until(lambda: session.state().dataset.is_recording)
    fake_cameras.return_value.read_latest.side_effect = RuntimeError("camera unplugged")

    # Failed: the open episode is discarded; the environment and the dataset stay loaded.
    await _until(lambda: not session.state().dataset.is_recording, within=15)
    assert session.environment is not None
    assert session.state().dataset.dataset_id == dataset.id
    assert session.store is store

    fake_cameras.return_value.read_latest.side_effect = None
    await session.load_environment(environment)
    # The same spec: the store and the dataset carry on.
    assert session.store is store
    assert session.state().dataset.dataset_id == dataset.id
    await _start(session, "pick")


async def test_a_dead_dataset_worker_unloads_the_dataset(new_session, fake_cameras, fake_worker) -> None:
    session = await new_session()
    statuses = subscribe_statuses(session)
    await session.load_environment(make_environment())
    await session.load_dataset(_dataset())
    os.kill(session._dataset.value._worker.pid, signal.SIGKILL)

    status = await next_status(statuses, "dataset")
    assert status.state == "failed"
    await _until(lambda: session.state().dataset is None)
    assert session.environment is not None  # the environment stays loaded


async def test_replay_plays_episodes_on_disk_only(new_session, fake_cameras) -> None:
    """Through lerobot: an episode reaches replay once the recording that saved it is finished."""
    dataset = _dataset()
    session = await new_session()
    await session.load_environment(make_environment())
    with pytest.raises(RuntimeSessionCommandError, match="Load a dataset"):
        await session.set_control(ReplayControlConfig(episode_index=0))
    await session.load_dataset(dataset)
    await _start(session, "pick the block")
    await asyncio.sleep(0.4)
    await _save(session)
    # Saved, but still in the recording's cache.
    with pytest.raises(ValueError, match="not in the dataset"):
        await session.set_control(ReplayControlConfig(episode_index=0))

    await session.unload_dataset()
    await session.load_dataset(dataset)
    await session.set_control(ReplayControlConfig(episode_index=0, grace_s=0))
    assert session.state().control == ReplayControlConfig(episode_index=0, grace_s=0)
