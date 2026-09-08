from __future__ import annotations

import threading
import time
from itertools import pairwise
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
from physicalai.inference.constants import IMAGES, STATE

from exceptions import ModelCameraMismatchError
from runtime.action_source import StudioActionSource
from runtime.contract import ErrorEvent, InMemoryCommandMailbox, LoadModelCommand, QueueEventSink, StartTaskCommand
from runtime.policy_loader import check_camera_keys
from schemas import InferenceBackend, InferenceDevice

from .fakes import FakeInferenceModel, FakeObservation, FakeRobot

_DEVICE = InferenceDevice(backend=InferenceBackend.TORCH, device="cpu")


def _observation(values: list[float], timestamp: float = 1.0) -> FakeObservation:
    return FakeObservation(np.array(values, dtype=np.float32), timestamp)


def _source(*, models_dir, camera_keys: tuple[str, ...] = ()):
    mailbox = InMemoryCommandMailbox()
    events = QueueEventSink()
    follower = FakeRobot([_observation([0.0, 0.0])])
    source = StudioActionSource(
        follower=follower,
        leader=None,
        mailbox=mailbox,
        event_sink=events,
        fps=50,
        camera_keys=camera_keys,
        models_dir=models_dir,
    )
    source.connect(bus=MagicMock(), session_id="test")
    return source, mailbox, events, follower


def _drain(events: QueueEventSink) -> list:
    items = []
    while True:
        try:
            items.append(events.get_nowait())
        except Exception:
            return items


def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("condition was not met")


def _export_dir(models_dir, model_id):
    path = models_dir / str(model_id) / "exports" / "torch"
    path.mkdir(parents=True)
    return path


def test_loading_does_not_stall_the_loop(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = uuid4()
    _export_dir(tmp_path, model_id)
    constructed = threading.Event()

    class SlowModel(FakeInferenceModel):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, construct_delay=1.0, **kwargs)
            constructed.set()

    monkeypatch.setattr("physicalai.inference.InferenceModel", SlowModel)
    source, mailbox, _events, follower = _source(models_dir=tmp_path)
    mailbox.apply(LoadModelCommand(model_id=model_id, inference_device=_DEVICE))

    timestamps: list[float] = []
    start = time.perf_counter()
    while time.perf_counter() - start < 1.2:
        tick = time.perf_counter()
        source.update(follower.get_observation(), {}, len(timestamps))
        timestamps.append(tick)
        remaining = 1 / 50 - (time.perf_counter() - tick)
        if remaining > 0:
            time.sleep(remaining)

    assert constructed.is_set()
    intervals = [later - earlier for earlier, later in pairwise(timestamps)]
    assert intervals
    assert max(intervals) < 3 / 50
    assert abs(float(np.median(intervals)) - 1 / 50) < 1 / 50
    source.shutdown_policy()


def test_warmup_runs_on_the_loader_thread(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = uuid4()
    _export_dir(tmp_path, model_id)
    warmup_threads: list[int] = []

    from physicalai.runtime.action_sources.policy import PolicySource

    real_warmup = PolicySource.warmup

    def tracking_warmup(self, observation):
        warmup_threads.append(threading.get_ident())
        return real_warmup(self, observation)

    monkeypatch.setattr("physicalai.inference.InferenceModel", FakeInferenceModel)
    monkeypatch.setattr(PolicySource, "warmup", tracking_warmup)
    source, mailbox, _events, follower = _source(models_dir=tmp_path)
    source.update(follower.get_observation(), {}, 0)
    mailbox.apply(LoadModelCommand(model_id=model_id, inference_device=_DEVICE))
    source.update(follower.get_observation(), {}, 1)
    loop_thread = threading.get_ident()

    _wait_until(lambda: source._policy is not None and source._model_loaded)
    assert warmup_threads
    assert all(thread != loop_thread for thread in warmup_threads)
    load_warmups = len(warmup_threads)
    source.update(follower.get_observation(), {}, 2)
    assert len(warmup_threads) == load_warmups
    assert source._policy is not None
    assert source._policy._warmed_up

    mailbox.apply(StartTaskCommand(task="pick"))
    source.update(follower.get_observation(), {}, 3)
    _wait_until(lambda: source.follower_source == "policy")
    assert len(warmup_threads) > load_warmups
    assert all(thread != loop_thread for thread in warmup_threads)
    after_play = len(warmup_threads)
    source.update(follower.get_observation(), {}, 4)
    assert len(warmup_threads) == after_play
    source.shutdown_policy()


def test_a_missing_export_directory_reports_model_not_found(tmp_path) -> None:
    source, mailbox, events, follower = _source(models_dir=tmp_path)
    mailbox.apply(LoadModelCommand(model_id=uuid4(), inference_device=_DEVICE))
    source.update(follower.get_observation(), {}, 0)
    seen: list = []

    def _has_error() -> bool:
        seen.extend(_drain(events))
        return any(isinstance(event, ErrorEvent) and event.error_code == "model_not_found" for event in seen)

    _wait_until(_has_error)
    source.update(follower.get_observation(), {}, 1)
    assert source.follower_source == "hold"


def test_a_second_load_supersedes_the_first(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_id = uuid4()
    second_id = uuid4()
    _export_dir(tmp_path, first_id)
    _export_dir(tmp_path, second_id)

    class LabeledModel(FakeInferenceModel):
        def __init__(self, *args, **kwargs) -> None:
            export_dir = kwargs.get("export_dir") or (args[0] if args else None)
            label = "first" if str(first_id) in str(export_dir) else "second"
            delay = 0.4 if label == "first" else 0.05
            super().__init__(*args, construct_delay=delay, label=label, **kwargs)

    monkeypatch.setattr("physicalai.inference.InferenceModel", LabeledModel)
    source, mailbox, _events, follower = _source(models_dir=tmp_path)
    mailbox.apply(LoadModelCommand(model_id=first_id, inference_device=_DEVICE))
    source.update(follower.get_observation(), {}, 0)
    mailbox.apply(LoadModelCommand(model_id=second_id, inference_device=_DEVICE))
    source.update(follower.get_observation(), {}, 1)

    _wait_until(lambda: source._policy is not None and getattr(source._policy._model, "label", None) == "second")
    time.sleep(0.5)
    assert source._policy is not None
    assert source._policy._model.label == "second"
    source.shutdown_policy()


def test_camera_mismatch_is_caught_at_load(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeInferenceModel(input_names=[STATE, f"{IMAGES}.wrist"])
    with pytest.raises(ModelCameraMismatchError, match="images.wrist") as mismatch:
        check_camera_keys(model, ["front"])
    assert "images.front" in mismatch.value.message

    model_id = uuid4()
    _export_dir(tmp_path, model_id)

    def _build(*args, **kwargs):
        return FakeInferenceModel(*args, input_names=[STATE, f"{IMAGES}.wrist"], **kwargs)

    monkeypatch.setattr("physicalai.inference.InferenceModel", _build)
    source, mailbox, events, follower = _source(models_dir=tmp_path, camera_keys=("front",))
    mailbox.apply(LoadModelCommand(model_id=model_id, inference_device=_DEVICE))
    source.update(follower.get_observation(), {}, 0)
    seen: list = []

    def _has_mismatch() -> bool:
        seen.extend(_drain(events))
        return any(isinstance(event, ErrorEvent) and event.error_code == "model_camera_mismatch" for event in seen)

    _wait_until(_has_mismatch)
    assert source._policy is None


def test_a_single_camera_model_ignores_the_camera_name() -> None:
    model = FakeInferenceModel(input_names=[STATE, IMAGES])
    check_camera_keys(model, ["overhead"])
    check_camera_keys(model, ["wrist"])


def test_loader_instantiates_async_execution(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai.runtime import AsyncExecution

    from runtime.config_builder import POLICY_REQUEST_THRESHOLD

    model_id = uuid4()
    _export_dir(tmp_path, model_id)
    monkeypatch.setattr("physicalai.inference.InferenceModel", FakeInferenceModel)
    source, mailbox, _events, follower = _source(models_dir=tmp_path)
    mailbox.apply(LoadModelCommand(model_id=model_id, inference_device=_DEVICE))
    source.update(follower.get_observation(), {}, 0)
    _wait_until(lambda: source._policy is not None)
    assert source._policy is not None
    assert isinstance(source._policy._execution, AsyncExecution)
    assert source._policy._execution._threshold_frac == POLICY_REQUEST_THRESHOLD
    source.shutdown_policy()


def _load_and_wait(source, mailbox, follower, command: LoadModelCommand, step: int) -> None:
    mailbox.apply(command)
    source.update(follower.get_observation(), {}, step)
    _wait_until(lambda: source._policy is not None and source._model_loaded)


def _count_load_requests(source) -> list[LoadModelCommand]:
    """Record what actually reaches the loader, so a skip is observable directly."""
    requested: list[LoadModelCommand] = []
    real_request = source._loader.request

    def tracking_request(command, *args, **kwargs):
        requested.append(command)
        return real_request(command, *args, **kwargs)

    source._loader.request = tracking_request
    return requested


def test_reloading_the_same_model_is_skipped(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = uuid4()
    _export_dir(tmp_path, model_id)
    monkeypatch.setattr("physicalai.inference.InferenceModel", FakeInferenceModel)
    source, mailbox, events, follower = _source(models_dir=tmp_path)
    _load_and_wait(source, mailbox, follower, LoadModelCommand(model_id=model_id, inference_device=_DEVICE), 0)

    attached = source._policy
    requested = _count_load_requests(source)
    _drain(events)
    mailbox.apply(LoadModelCommand(model_id=model_id, inference_device=_DEVICE))
    source.update(follower.get_observation(), {}, 1)

    assert requested == []
    assert source._policy is attached
    assert source._model_loaded is True
    # No False -> True flicker: the client is already past its readiness gate.
    assert _drain(events) == []
    source.shutdown_policy()


def test_a_forced_reload_rebuilds_the_same_model(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = uuid4()
    _export_dir(tmp_path, model_id)
    monkeypatch.setattr("physicalai.inference.InferenceModel", FakeInferenceModel)
    source, mailbox, _events, follower = _source(models_dir=tmp_path)
    _load_and_wait(source, mailbox, follower, LoadModelCommand(model_id=model_id, inference_device=_DEVICE), 0)

    attached = source._policy
    mailbox.apply(LoadModelCommand(model_id=model_id, inference_device=_DEVICE, force=True))
    source.update(follower.get_observation(), {}, 1)

    _wait_until(lambda: source._policy is not None and source._policy is not attached and source._model_loaded)
    source.shutdown_policy()


def test_the_same_model_on_another_device_is_not_a_match(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = uuid4()
    _export_dir(tmp_path, model_id)
    monkeypatch.setattr("physicalai.inference.InferenceModel", FakeInferenceModel)
    source, mailbox, _events, follower = _source(models_dir=tmp_path)
    _load_and_wait(source, mailbox, follower, LoadModelCommand(model_id=model_id, inference_device=_DEVICE), 0)

    attached = source._policy
    other_device = InferenceDevice(backend=InferenceBackend.TORCH, device="gpu")
    mailbox.apply(LoadModelCommand(model_id=model_id, inference_device=other_device))
    source.update(follower.get_observation(), {}, 1)

    _wait_until(lambda: source._policy is not None and source._policy is not attached and source._model_loaded)
    source.shutdown_policy()


def test_a_failed_load_does_not_leave_a_matching_identity(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The identity must be cleared before a load, not after it succeeds.

    Otherwise: session holds A, a load for B fails, ``_policy`` is still A with
    ``model_loaded`` False -- and a later request for A matches the stale
    identity, skips, and leaves the client on "Initializing" forever with a
    perfectly good policy attached.
    """
    loaded_id = uuid4()
    missing_id = uuid4()
    _export_dir(tmp_path, loaded_id)
    monkeypatch.setattr("physicalai.inference.InferenceModel", FakeInferenceModel)
    source, mailbox, events, follower = _source(models_dir=tmp_path)
    _load_and_wait(source, mailbox, follower, LoadModelCommand(model_id=loaded_id, inference_device=_DEVICE), 0)

    mailbox.apply(LoadModelCommand(model_id=missing_id, inference_device=_DEVICE))
    source.update(follower.get_observation(), {}, 1)
    seen: list = []

    def _failed() -> bool:
        seen.extend(_drain(events))
        return any(isinstance(event, ErrorEvent) and event.error_code == "model_not_found" for event in seen)

    _wait_until(_failed)
    assert source._model_loaded is False

    requested = _count_load_requests(source)
    mailbox.apply(LoadModelCommand(model_id=loaded_id, inference_device=_DEVICE))
    source.update(follower.get_observation(), {}, 2)

    assert len(requested) == 1
    _wait_until(lambda: source._model_loaded)
    source.shutdown_policy()


def test_a_skipped_load_leaves_a_running_policy_alone(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = uuid4()
    _export_dir(tmp_path, model_id)
    monkeypatch.setattr("physicalai.inference.InferenceModel", FakeInferenceModel)
    source, mailbox, _events, follower = _source(models_dir=tmp_path)
    _load_and_wait(source, mailbox, follower, LoadModelCommand(model_id=model_id, inference_device=_DEVICE), 0)

    mailbox.apply(StartTaskCommand(task="pick"))
    source.update(follower.get_observation(), {}, 1)
    _wait_until(lambda: source.follower_source == "policy")
    attached = source._policy

    requested = _count_load_requests(source)
    mailbox.apply(LoadModelCommand(model_id=model_id, inference_device=_DEVICE))
    source.update(follower.get_observation(), {}, 2)

    # No rebuild is even started, so nothing can swap the PolicySource out from
    # under the control loop or cancel the arming that put it in policy mode.
    assert requested == []
    assert source.follower_source == "policy"
    assert source._policy is attached
    source.shutdown_policy()


def test_shutdown_clears_the_identity_so_a_later_load_rebuilds(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_id = uuid4()
    _export_dir(tmp_path, model_id)
    monkeypatch.setattr("physicalai.inference.InferenceModel", FakeInferenceModel)
    source, mailbox, _events, follower = _source(models_dir=tmp_path)
    _load_and_wait(source, mailbox, follower, LoadModelCommand(model_id=model_id, inference_device=_DEVICE), 0)

    source.shutdown_policy()
    assert source._loaded_identity is None
