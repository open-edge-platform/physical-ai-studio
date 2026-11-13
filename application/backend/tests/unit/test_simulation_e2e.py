# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for application-owned simulation controls and evidence."""

import json
import os
import re
import subprocess
import sys
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import yaml
from packaging.requirements import Requirement
from physicalai.config import Config
from physicalai.inference import InferenceModel
from tests.integration import simulation_e2e as controls
from tests.integration import test_simulation_e2e as suite

from services.runtime_export import policy_source_fragment

ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "application/backend"


def snapshot(count=0, correct=7):
    return {
        "episode": {
            "episode_count": count,
            "items_per_episode": 10,
            "last_episode": {"correct": correct, "wrong": 1, "missed": 9 - correct},
        },
        "timing": {"real_time_factor": 1.0, "camera_fps": {"overview": 30, "wrist": 30}},
    }


def test_monitor_collects_one_complete_episode_and_stops():
    stop = threading.Event()
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), stop, 1)
    monitor.observe(snapshot(1))
    assert stop.is_set()
    assert monitor.result(42)["score"] == 0.7
    assert monitor.result(42)["seed"] == 42


@pytest.mark.parametrize("count", [-1, 2])
def test_monitor_rejects_lost_or_reset_episodes(count):
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), threading.Event(), 1)
    with pytest.raises(AssertionError, match="Lost simulator"):
        monitor.observe(snapshot(count))


def test_monitor_rejects_partial_item_accounting():
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), threading.Event(), 1)
    value = snapshot(1)
    value["episode"]["last_episode"]["missed"] = 0
    with pytest.raises(AssertionError, match="Incomplete"):
        monitor.observe(value)


def test_tick_metrics_do_not_poll_http(monkeypatch):
    monkeypatch.setattr(controls, "health", MagicMock(side_effect=AssertionError("HTTP in tick")))
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), threading.Event(), 1)
    for position in (0, 2):
        monitor.on_tick(
            SimpleNamespace(
                action_sent=np.ones(6),
                robot_state=SimpleNamespace(joint_positions=np.full(6, position)),
                camera_frames={"wrist": SimpleNamespace(timestamp=float(position), data=np.full((2, 2, 3), position))},
            )
        )
    monitor.on_inference(SimpleNamespace(latency_s=0.02))
    monitor.observe(snapshot(1))
    result = monitor.result(42)
    assert result["actions_sent"] == 2
    assert result["max_joint_change"] == 2
    assert result["unique_camera_frames"] == {"wrist": 2}
    assert result["inference_p95_s"] == 0.02
    controls.health.assert_not_called()


@pytest.mark.parametrize(
    "action",
    [np.ones(5), np.ones((1, 6)), np.full(6, np.nan), np.full(6, np.inf), np.full(6, -np.inf), np.full(6, "bad")],
)
@pytest.mark.parametrize("recording", [False, True])
def test_invalid_actions_fail_even_when_runtime_swallows_callback_errors(action, recording):
    from physicalai.runtime._callback_bus import _CallbackBus

    stop = threading.Event()
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), stop, 1)
    monitor.recording = recording
    monitor.thread = MagicMock()
    monitor.thread.is_alive.return_value = False
    _CallbackBus([monitor]).emit_tick(SimpleNamespace(action_sent=action))
    assert stop.is_set()
    assert monitor.actions_sent == 0
    with pytest.raises(AssertionError, match="six finite joint values"):
        monitor.close()
    if not recording:
        with pytest.raises(AssertionError, match="six finite joint values"):
            monitor.begin(snapshot())


def test_monitor_deadline_stops_runtime(monkeypatch):
    monkeypatch.setattr(controls, "health", lambda url: snapshot())
    stop = threading.Event()
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), stop, 0)
    monitor.thread.start()
    assert stop.wait(1)
    with pytest.raises(TimeoutError, match="Episode did not finish"):
        monitor.close()


def test_seeded_setup_uses_mode_barrier_before_resuming(monkeypatch):
    calls = []
    monkeypatch.setattr(controls, "command", lambda url, route, payload: calls.append((route, payload)))
    states = [
        {"episode": {"active": False}},
        {"autopilot": {"mode": "leader"}},
        {"seed": 42},
        {"autopilot": {"mode": "off"}, "episode": {"spawned": 0, "belt_speed": 0.04}},
    ]
    monkeypatch.setattr(controls, "health", lambda url: states.pop(0))
    controls.prepare_episode("http://localhost", 42, "off", 0.04)
    assert calls == [
        ("episode/auto-reset", {"enabled": False}),
        ("autopilot", {"mode": "leader"}),
        ("seed", {"seed": 42}),
        ("conveyor/belt-speed", {"speed": 0.04}),
        ("reset", {}),
        ("autopilot", {"mode": "off"}),
    ]


@pytest.mark.parametrize("accelerator", ["cpu", "cuda"])
def test_training_uses_ui_defaults(tmp_path, monkeypatch, accelerator):
    monkeypatch.setenv("SIM_E2E_ACCELERATOR", accelerator)
    monkeypatch.setenv("SIM_E2E_EPOCHS", "1")
    monkeypatch.setattr(suite.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(suite.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=40 * 1024**3))
    client = MagicMock()
    client.get.return_value.status_code = 200
    client.get.return_value.json.return_value = [{"id": "act", "min_vram": 0}]
    client.post.return_value.status_code = 200
    client.post.return_value.json.return_value = {"id": "job"}
    wait = MagicMock(side_effect=TimeoutError("stop after submission"))
    monkeypatch.setattr(suite, "wait_for_job", wait)
    with pytest.raises(TimeoutError, match="stop after submission"):
        suite._train(client, "act", str(uuid4()), str(uuid4()), tmp_path)
    payload = client.post.call_args.kwargs["json"]
    assert payload == json.loads((tmp_path / "training-request.json").read_text())
    assert payload["max_epochs"] == 5
    assert payload["batch_size"] == 8
    assert payload["num_workers"] == "auto"
    assert payload["val_split"] == 0.1
    assert payload["precision"] == ("bf16-mixed" if accelerator == "cuda" else "32-true")
    assert payload["export_backends"] is None
    assert payload["base_model_id"] is None
    for flag in ("auto_scale_batch_size", "compile_model", "lora_enabled", "snapflow_enabled", "augment_images"):
        assert payload[flag] is False
    assert wait.call_args.kwargs["timeout"] == 18000


@pytest.fixture
def control_report():
    return {
        "status": "passed",
        **{
            mode: {
                "score": correct / 10,
                "trials": [
                    {"seed": seed, "episodes": [{"correct": correct, "wrong": 0, "missed": 10 - correct}]}
                    for seed in controls.SEEDS
                ],
            }
            for mode, correct in (("off", 0), ("drive", 10))
        },
    }


@pytest.mark.parametrize(
    "correct,timing,error",
    [(7, True, None), (0, True, "no-action"), (7, False, "timing")],
)
def test_automated_training_runs_the_fresh_model(tmp_path, monkeypatch, control_report, correct, timing, error):
    report = tmp_path / "controls.json"
    report.write_text(json.dumps(control_report))
    for key, value in {
        "DATASET_DIR": str(tmp_path),
        "CONTROLS_FILE": str(report),
    }.items():
        monkeypatch.setenv(f"SIM_E2E_{key}", value)
    monkeypatch.setattr(suite, "runtime_versions", dict)
    monkeypatch.setattr(
        suite,
        "_recording_metadata",
        MagicMock(return_value={"fps": 30, "features": {"observation.images.overview": {}}}),
    )
    monkeypatch.setattr(suite, "_task_instruction", MagicMock(return_value="sort items"))
    environment = MagicMock(return_value=("project", "environment"))
    monkeypatch.setattr(suite, "_project_environment", environment)
    imported = MagicMock(return_value="dataset")
    monkeypatch.setattr(suite, "_import_dataset", imported)
    model = {"id": "fresh-model"}
    train = MagicMock(return_value=(model, {"model_id": model["id"], "job_id": "fresh-job"}))
    monkeypatch.setattr(suite, "_train", train)
    bundle = MagicMock(return_value=tmp_path / "runtime.yaml")
    monkeypatch.setattr(suite, "_runtime_bundle", bundle)
    rollout = MagicMock(return_value=[{"episodes": [{"correct": correct, "wrong": 0, "missed": 10 - correct}]}])
    monkeypatch.setattr(suite, "_trials", rollout)
    monkeypatch.setattr(suite, "timing_passes", MagicMock(return_value=timing))
    request = MagicMock()
    if error:
        with pytest.raises(AssertionError, match=error):
            suite.test_train_export_and_run_in_simulation(tmp_path, request, monkeypatch)
    else:
        suite.test_train_export_and_run_in_simulation(tmp_path, request, monkeypatch)
    client = request.getfixturevalue.return_value
    assert imported.call_args.args[3] == tmp_path.resolve()
    train.assert_called_once_with(client, "act", "project", "dataset", tmp_path.resolve())
    training = json.loads((tmp_path / "training.json").read_text())
    assert training["model_id"] == "fresh-model" and training["job_id"] == "fresh-job"
    bundle.assert_called_once_with(client, model, "environment", "sort items", tmp_path.resolve())
    assert rollout.call_args.args[0] == bundle.return_value
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["model_id"] == training["model_id"]
    assert result["status"] == ("failed" if error else "passed")
    assert result["quality"] == ("passed" if correct else "failed")
    assert result["timing"] == ("passed" if timing else "failed")


@pytest.mark.parametrize(
    "epoch,loss,exported,error",
    [
        (4, "0.1", True, None),
        (0, "0.1", True, "epoch budget"),
        (4, "nan", True, "validation loss"),
        (4, "0.1", False, "OpenVINO"),
    ],
)
def test_training_validates_deployment_export_epochs_and_loss(tmp_path, monkeypatch, epoch, loss, exported, error):
    from schemas.model import Model
    from services.model_metrics_service import ModelMetricsService

    monkeypatch.setenv("SIM_E2E_ACCELERATOR", "cpu")
    if exported:
        export = tmp_path / "exports/openvino"
        export.mkdir(parents=True)
        (export / "manifest.json").write_text("{}")
    model = Model.model_construct(
        id=uuid4(), name="ACT", policy="act", project_id=uuid4(), path=str(tmp_path), properties={}
    ).model_dump(mode="json")
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(f"epoch,val/loss\n{epoch},{loss}\n")
    monkeypatch.setattr(ModelMetricsService, "get_model_metrics_path", AsyncMock(return_value=metrics_path))
    client = MagicMock()
    client.get.return_value.status_code = 200
    client.get.return_value.json.return_value = [{"id": "act"}]
    client.post.return_value.status_code = 200
    client.post.return_value.json.return_value = {"id": "job"}
    monkeypatch.setattr(suite, "wait_for_job", MagicMock(return_value={}))
    monkeypatch.setattr(suite, "wait_for_model", MagicMock(return_value=model))
    if error:
        with pytest.raises(AssertionError, match=error):
            suite._train(client, "act", str(uuid4()), str(uuid4()), tmp_path)
    else:
        _, training = suite._train(client, "act", str(uuid4()), str(uuid4()), tmp_path)
        assert training["training_epochs"] == 5


def test_controls_scores_use_completed_episodes(tmp_path, control_report):
    path = tmp_path / "controls.json"
    control_report["off"]["score"] = 1  # Do not trust a stale summary over the actual scored episodes.
    path.write_text(json.dumps(control_report))
    assert controls.control_scores(path) == {"off": 0, "drive": 1}


@pytest.mark.parametrize("invalid", ["status", "seed", "score"])
def test_controls_must_pass_on_the_rollout_seeds(tmp_path, control_report, invalid):
    if invalid == "status":
        control_report["status"] = "incomplete"
    elif invalid == "seed":
        control_report["off"]["trials"][0]["seed"] = 0
    else:
        for trial in control_report["drive"]["trials"]:
            trial["episodes"] = [{"correct": 0, "wrong": 0, "missed": 10}]
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(control_report))
    with pytest.raises(ValueError):
        controls.control_scores(path)


def test_full_pipeline_requires_controls_before_training(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_E2E_DATASET_DIR", str(tmp_path))
    monkeypatch.setenv("SIM_E2E_CONTROLS_FILE", str(tmp_path / "missing.json"))
    monkeypatch.setattr(suite, "runtime_versions", dict)
    request = MagicMock()
    with pytest.raises(FileNotFoundError):
        suite.test_train_export_and_run_in_simulation(tmp_path, request, monkeypatch)
    request.getfixturevalue.assert_not_called()


def test_trial_loads_application_recipe_and_reseeds_after_warmup(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(suite, "SEEDS", (42,))
    monkeypatch.setenv("SIM_E2E_HTTP_URL", "http://localhost")
    monkeypatch.setenv("SIM_E2E_RECORD_VIDEO", "false")
    monkeypatch.setattr(suite, "prepare_episode", lambda *args: events.append("prepare") or snapshot())
    monkeypatch.setattr(
        suite, "command", lambda url, route, payload: events.append("resume" if payload["enabled"] else "pause")
    )
    monkeypatch.setattr(suite, "wait_health", MagicMock())
    monitor = MagicMock(actions_sent=2, max_joint_change=2)
    monitor.begin.side_effect = lambda before: events.append("begin")
    monitor.result.return_value = {"score": 0.7}
    monkeypatch.setattr(suite, "EpisodeMonitor", MagicMock(return_value=monitor))
    runtime = MagicMock()
    runtime.run.side_effect = lambda **kwargs: events.append("run")
    factory = MagicMock(return_value=runtime)
    monkeypatch.setattr(suite.RobotRuntime, "from_config", factory)
    recipe = tmp_path / "runtime.yaml"
    recipe.write_text("application-generated recipe remains untouched")
    suite._trials(recipe, {"fps": 30, "features": {"observation.images.wrist": {}}}, tmp_path, "act", monkeypatch)
    factory.assert_called_once_with(recipe)
    assert recipe.read_text() == "application-generated recipe remains untouched"
    assert events == ["prepare", "run", "prepare", "begin", "resume", "run", "pause"]
    runtime.disconnect.assert_called_once()


def test_trial_reconnects_the_application_source_without_replaying_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "SEEDS", (42,))
    monkeypatch.setenv("SIM_E2E_HTTP_URL", "http://localhost")
    monkeypatch.setenv("SIM_E2E_RECORD_VIDEO", "false")
    monkeypatch.setattr(suite, "prepare_episode", lambda *args: snapshot())
    monkeypatch.setattr(suite, "command", MagicMock())
    monkeypatch.setattr(suite, "wait_health", MagicMock())
    model = MagicMock()
    model.predict_action_chunk.side_effect = [
        np.ones((100, 6), dtype=np.float32),
        np.full((100, 6), 2, dtype=np.float32),
    ]
    monkeypatch.setattr(InferenceModel, "__new__", staticmethod(lambda cls, **kwargs: model))
    robot = MagicMock()
    robot.joint_names = [f"joint_{index}" for index in range(6)]
    robot.get_observation.return_value = SimpleNamespace(state=np.zeros(6), joint_positions=np.zeros(6), images=None)
    camera = MagicMock()
    camera.read_latest.return_value = SimpleNamespace(data=np.zeros((12, 16, 3), dtype=np.uint8), timestamp=0.0)

    def make_monitor(_url, _before, stop, _timeout):
        return SimpleNamespace(
            on_tick=lambda event: stop.set(),
            begin=MagicMock(),
            close=MagicMock(),
            result=lambda seed: {"seed": seed},
            actions_sent=1,
            max_joint_change=2,
            finished=threading.Event(),
            thread=SimpleNamespace(ident=None),
        )

    monkeypatch.setattr(suite, "EpisodeMonitor", make_monitor)

    def load_recipe(_path):
        fragment = policy_source_fragment(export_dir="unused", backend="torch", device="cpu", task="Sort")
        return suite.RobotRuntime(
            robot=robot, action_source=Config.from_dict(fragment).instantiate(), cameras={"wrist": camera}, fps=30
        )

    monkeypatch.setattr(suite.RobotRuntime, "from_config", load_recipe)
    recipe = tmp_path / "runtime.yaml"
    trials = suite._trials(
        recipe, {"fps": 30, "features": {"observation.images.wrist": {}}}, tmp_path, "act", monkeypatch
    )
    assert trials == [{"seed": 42}]
    assert model.reset.call_count == 2
    assert model.predict_action_chunk.call_count == 2
    assert robot.send_action.call_count == 2
    np.testing.assert_array_equal(robot.send_action.call_args_list[0].args[0], np.ones(6))
    np.testing.assert_array_equal(robot.send_action.call_args_list[1].args[0], np.full(6, 2))


@pytest.mark.parametrize("instructions", [[], [""], ["  "], [0], ["Sort", "Reject"]])
def test_rollout_rejects_missing_or_ambiguous_task(tmp_path, instructions):
    (tmp_path / "meta").mkdir()
    pd.DataFrame({"task_index": range(len(instructions))}, index=pd.Index(instructions, name="task")).to_parquet(
        tmp_path / "meta/tasks.parquet"
    )
    with pytest.raises(ValueError, match="exactly one non-empty task instruction"):
        suite._task_instruction(tmp_path)


def test_rollout_reads_recorded_task_verbatim(tmp_path):
    (tmp_path / "meta").mkdir()
    instruction = "Sort conveyor items by color and reject cracks"
    pd.DataFrame({"task_index": [0]}, index=pd.Index([instruction], name="task")).to_parquet(
        tmp_path / "meta/tasks.parquet"
    )
    assert suite._task_instruction(tmp_path) == instruction


def test_paused_warmup_does_not_enter_trial_metrics():
    stop = threading.Event()
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), stop, 1)
    monitor.recording = False
    monitor.on_tick(SimpleNamespace(action_sent=None))
    assert stop.is_set()
    assert monitor.ticks == []


def test_timing_includes_delay_before_first_action():
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), threading.Event(), 1)
    monitor.started_at = 0
    monitor.ticks = [10, 10.01]
    monitor.observe(snapshot(1))
    result = monitor.result(42)
    assert result["control_hz"] < 1
    assert result["first_action_delay_s"] == 10


@pytest.mark.parametrize(
    "timing",
    [
        {"real_time_factor": 1.0, "camera_fps": {"overview": 30, "wrist": 30}},
        {"real_time_factor": 0.5, "camera_fps": {"overview": 30, "wrist": 30}},
        {"real_time_factor": 1.2, "camera_fps": {"overview": 30, "wrist": 30}},
        {"real_time_factor": 1.0, "camera_fps": {"overview": 30, "wrist": 15}},
        {"real_time_factor": 1.0, "camera_fps": {"overview": 30, "wrist": None}},
        {"real_time_factor": 1.0, "camera_fps": {}},
        {"real_time_factor": 1.0, "camera_fps": {"overview": 30}},
        {},
    ],
)
def test_timing_gate_keeps_earlier_violations_after_recovery(timing, monkeypatch):
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), threading.Event(), 1)
    monitor.started_at = 0
    monitor.ticks = [(index + 1) / 30 for index in range(300)]
    monitor.frame_times = {camera: set(range(300)) for camera in ("overview", "wrist")}
    monitor.frame_updates = dict.fromkeys(monitor.frame_times, 10.0)
    monitor.frame_max_gaps = dict.fromkeys(monitor.frame_times, 1 / 30)
    monitor.changed_cameras.update(monitor.frame_times)
    monkeypatch.setattr(controls.time, "monotonic", lambda: 10.0)
    for _ in range(100):
        monitor.observe({**snapshot(), "timing": timing})
    monitor.observe(snapshot(1))
    result = monitor.result(42)
    assert result["simulator_timing"] == snapshot()["timing"]
    assert len(result["simulator_timing_samples"]) == 101
    assert controls.timing_passes([result], 30, ["overview", "wrist"]) == (timing == snapshot()["timing"])


@pytest.mark.parametrize("control_hz", [None, 0, 23.9])
def test_timing_gate_rejects_slow_control_loop(control_hz):
    assert not controls.timing_passes(
        [{"control_hz": control_hz, "simulator_timing_samples": [snapshot()["timing"]]}], 30, ["overview"]
    )


def test_timing_gate_requires_trials_and_simulator_measurements():
    assert not controls.timing_passes([], 30, ["overview"])
    trial = {
        "control_hz": 30,
        "camera_hz": {"overview": 30},
        "camera_max_gap_s": {"overview": 0.04},
        "changed_cameras": ["overview"],
    }
    assert not controls.timing_passes([{**trial, "simulator_timing_samples": []}], 30, ["overview"])
    assert not controls.timing_passes([trial], 30, [])


@pytest.mark.parametrize(
    "case", ["healthy", "frozen", "missing", "slow", "recovered", "stopped", "stale_pixels", "static_start"]
)
def test_timing_gate_requires_progress_from_every_runtime_camera(monkeypatch, case):
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), threading.Event(), 10)
    monitor.started_at = 0.0
    clock = iter([(index + 1) / 30 for index in range(300)] + [10.0])
    monkeypatch.setattr(controls.time, "monotonic", lambda: next(clock))
    for index in range(300):
        wrist = index
        if case == "frozen":
            wrist = 0
        elif case == "slow":
            wrist = index // 2
        elif case == "recovered" and 30 <= index < 45:
            wrist = 29
        elif case == "stopped":
            wrist = min(index, 29)
        pixels = 0 if case == "stale_pixels" or (case == "static_start" and index < 150) else wrist
        frames = {
            "overview": SimpleNamespace(timestamp=index, data=np.full((2, 2, 3), index)),
            "wrist": SimpleNamespace(timestamp=wrist, data=np.full((2, 2, 3), pixels)),
        }
        if case == "missing":
            del frames["wrist"]
        monitor.on_tick(
            SimpleNamespace(
                action_sent=np.ones(6),
                robot_state=SimpleNamespace(joint_positions=np.full(6, index / 100)),
                camera_frames=frames,
            )
        )
    monitor.observe(snapshot(1))
    result = monitor.result(42)
    assert result["control_hz"] == 30
    assert result["simulator_timing"]["camera_fps"]["wrist"] == 30
    assert controls.timing_passes([result], 30, ["overview", "wrist"]) == (case in {"healthy", "static_start"})
    if case == "recovered":
        assert result["camera_hz"]["wrist"] >= 24
        assert result["camera_max_gap_s"]["wrist"] > 0.25


@pytest.mark.parametrize("completion,passed", [(10.0, True), (10.5, False), (15.0, False)])
def test_timing_windows_include_a_stall_after_the_last_tick(monkeypatch, completion, passed):
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), threading.Event(), 20)
    monitor.started_at = 0.0
    monitor.ticks = [(index + 1) / 30 for index in range(300)]
    monitor.frame_times = {camera: set(range(300)) for camera in ("overview", "wrist")}
    monitor.frame_updates = dict.fromkeys(monitor.frame_times, 10.0)
    monitor.frame_max_gaps = dict.fromkeys(monitor.frame_times, 1 / 30)
    monitor.changed_cameras.update(monitor.frame_times)
    monkeypatch.setattr(controls.time, "monotonic", lambda: completion)
    monitor.observe(snapshot(1))
    result = monitor.result(42)
    assert result["control_hz"] == 300 / completion
    assert result["camera_hz"]["wrist"] == 300 / completion
    assert result["camera_max_gap_s"]["wrist"] == max(1 / 30, completion - 10)
    assert controls.timing_passes([result], 30, ["overview", "wrist"]) == passed


@pytest.mark.parametrize(
    "rates,passed", [({"front": 30, "hand": 30}, True), ({"front": 30}, False), ({"front": 30, "wrist": 30}, False)]
)
def test_timing_requires_every_mapped_simulator_camera(rates, passed):
    cameras = ["overview", "wrist"]
    trial = {
        "control_hz": 30,
        "camera_hz": dict.fromkeys(cameras, 30),
        "camera_max_gap_s": dict.fromkeys(cameras, 0.04),
        "changed_cameras": cameras,
        "simulator_timing_samples": [{"real_time_factor": 1, "camera_fps": rates}],
    }
    assert controls.timing_passes([trial], 30, cameras, {"overview": "front", "wrist": "hand"}) == passed


def test_content_evidence_owns_shared_camera_pixels():
    monitor = controls.EpisodeMonitor("http://localhost", snapshot(), threading.Event(), 10)
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    event = SimpleNamespace(
        action_sent=np.ones(6),
        robot_state=SimpleNamespace(joint_positions=np.zeros(6)),
        camera_frames={"wrist": SimpleNamespace(timestamp=0, data=pixels)},
    )
    monitor.on_tick(event)
    assert not np.shares_memory(monitor.initial_images["wrist"], pixels)
    pixels[:] = 1
    event.camera_frames["wrist"].timestamp = 1
    monitor.on_tick(event)
    assert monitor.changed_cameras == {"wrist"}
    assert "wrist" not in monitor.initial_images


def test_video_requests_require_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(suite.shutil, "which", lambda name: None)
    with (
        pytest.raises(RuntimeError, match="requires ffmpeg"),
        suite._record_video(tmp_path, "http://localhost", ["overview"], {}, 30, "act", "conveyor_sort"),
    ):
        pass


@pytest.mark.parametrize(
    "case,error",
    [
        ("running", None),
        ("finished", None),
        ("encoder_failed", RuntimeError),
        ("encoder_hung", RuntimeError),
        ("unexpected_interrupt", RuntimeError),
        ("early_exit", RuntimeError),
        ("few_frames", RuntimeError),
        ("missing", RuntimeError),
        ("empty", RuntimeError),
        ("no_frames", RuntimeError),
        ("zero_duration", RuntimeError),
        ("corrupt", subprocess.CalledProcessError),
        ("decode_timeout", subprocess.TimeoutExpired),
        ("rollout_failed", AssertionError),
    ],
)
def test_video_requires_successful_encoding_decoding_and_rollout_coverage(tmp_path, monkeypatch, case, error):
    path = tmp_path / "inference-act-conveyor_sort.mp4"
    if case != "missing":
        path.write_bytes(b"" if case == "empty" else b"encoded video")
    process = MagicMock(returncode=1 if case == "encoder_failed" else (255 if case == "unexpected_interrupt" else 0))
    process.poll.side_effect = [None, 255 if case == "unexpected_interrupt" else (0 if case == "finished" else None)]
    if case == "encoder_hung":
        process.communicate.side_effect = subprocess.TimeoutExpired("ffmpeg", 30)
    monkeypatch.setattr(suite.subprocess, "Popen", MagicMock(return_value=process))
    monkeypatch.setattr(suite.shutil, "which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(suite.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(suite.time, "monotonic", MagicMock(side_effect=[0, 1, 11]))
    duration = 8 if case == "early_exit" else (0 if case == "zero_duration" else 10)
    frames = 0 if case == "no_frames" else (10 if case == "few_frames" else 300)
    decoder = MagicMock(
        return_value=SimpleNamespace(stdout=f"frame={frames}\nout_time_us={duration * 1000000}\nprogress=end\n")
    )
    if case == "corrupt":
        decoder.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
    elif case == "decode_timeout":
        decoder.side_effect = subprocess.TimeoutExpired("ffmpeg", 120)
    monkeypatch.setattr(suite.subprocess, "run", decoder)

    def record():
        with suite._record_video(tmp_path, "http://localhost", ["overview", "wrist"], {}, 30, "act", "conveyor_sort"):
            if case == "rollout_failed":
                raise AssertionError("Policy rollout failed")

    if error:
        with pytest.raises(error):
            record()
    else:
        record()
    encoder_args = suite.subprocess.Popen.call_args.args[0]
    assert "-nostdin" not in encoder_args
    assert "-xerror" in encoder_args
    assert "hstack=inputs=2:shortest=1" in encoder_args[encoder_args.index("-filter_complex") + 1]
    assert suite.subprocess.Popen.call_args.kwargs["stdin"] == subprocess.PIPE
    if case not in {"finished", "unexpected_interrupt"}:
        process.communicate.assert_called_once_with(input=b"q\n", timeout=30)
    if decoder.called:
        args = decoder.call_args.args[0]
        assert "-xerror" in args and str(path) in args
        assert decoder.call_args.kwargs["check"] is True
    if case == "encoder_hung":
        process.kill.assert_called_once()
        process.wait.assert_called_once()
        decoder.assert_not_called()
    if case == "rollout_failed":
        assert path.is_file()
        decoder.assert_called_once()


@pytest.mark.parametrize("borrowed_pid", [False, True])
def test_cleanup_never_shuts_down_an_unowned_simulator(tmp_path, borrowed_pid):
    workflow = yaml.safe_load((ROOT / ".github/workflows/simulation-application-e2e.yml").read_text())
    cleanup = next(step["run"] for step in workflow["jobs"]["simulation"]["steps"] if step.get("name") == "Stop owner")
    binary = tmp_path / "bin"
    binary.mkdir()
    curl = binary / "curl"
    curl.write_text('#!/bin/sh\ntouch "$SIM_E2E_OUTPUT_DIR/unexpected-shutdown"\n')
    curl.chmod(0o755)
    if borrowed_pid:
        (tmp_path / "owner.pid").write_text(str(os.getpid()))
    subprocess.run(
        ["bash", "-c", cleanup],
        check=True,
        env={
            "PATH": f"{binary}:{os.environ['PATH']}",
            "SIM_E2E_OUTPUT_DIR": str(tmp_path),
            "SIM_E2E_HTTP_URL": "http://localhost",
            "SIM_E2E_OWNER": "not-owned-by-this-test",
        },
    )
    assert not (tmp_path / "unexpected-shutdown").exists()


def test_lockfiles_resolve_the_configured_runtime_and_plugin_pin():
    for project in (ROOT / "library", BACKEND):
        config = tomllib.loads((project / "pyproject.toml").read_text())
        lock = tomllib.loads((project / "uv.lock").read_text())
        requirements = [Requirement(dependency) for dependency in config["project"]["dependencies"]]
        runtime = next(requirement for requirement in requirements if requirement.name == "physicalai")
        package = next(package for package in lock["package"] if package["name"] == "physicalai")
        assert package["version"] in runtime.specifier
        # Between runtime releases the runtime may be pinned to a full commit, never a branch or PR ref.
        runtime_git = re.fullmatch(
            r"https://github\.com/openvinotoolkit/physicalai\.git\?rev=([0-9a-f]{40})#\1",
            package["source"].get("git", ""),
        )
        assert package["source"] == {"registry": "https://pypi.org/simple"} or runtime_git
        if project == BACKEND:
            plugin = next(package for package in lock["package"] if package["name"] == "physicalai-mujoco-plugin")
            sim_dependencies = config["project"]["optional-dependencies"]["sim-e2e"]
            assert f"physicalai-mujoco-plugin=={plugin['version']}" in sim_dependencies
            assert plugin["source"] == {"registry": "https://pypi.org/simple"}


@pytest.mark.parametrize("runtime_version,plugin_version", [("0.3.1", "0.5.0"), ("0.4.0", "0.6.0")])
def test_runtime_versions_reports_installed_versions(monkeypatch, runtime_version, plugin_version):
    runtime = MagicMock(version=runtime_version)
    plugin = MagicMock(version=plugin_version)
    monkeypatch.setattr(controls, "distribution", lambda name: runtime if name == "physicalai" else plugin)
    assert controls.runtime_versions() == {
        "physicalai": runtime_version,
        "physicalai-mujoco-plugin": plugin_version,
    }
    runtime.read_text.assert_not_called()
    plugin.read_text.assert_not_called()


@pytest.mark.parametrize("mismatch", [None, "scene", "belt_speeds", "episode count"])
def test_recording_metadata_matches_pinned_dataset(tmp_path, mismatch):
    recording = {"scene": "conveyor_sort", "belt_speed": 0.04}
    metadata = {
        "total_episodes": 100,
        "fps": 30,
        "features": {key: {"shape": [6]} for key in ("action", "observation.state")},
    }
    if mismatch == "scene":
        recording["scene"] = "other_scene"
    elif mismatch == "belt_speeds":
        recording["belt_speeds_m_s"] = [0.04, 0.08]
    elif mismatch == "episode count":
        metadata["total_episodes"] = 50
    (tmp_path / "simulation.json").write_text(json.dumps(recording))
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps(metadata))
    if mismatch is None:
        assert suite._recording_metadata(tmp_path, "conveyor_sort") == metadata
    else:
        with pytest.raises(ValueError, match="episode count" if mismatch == "episode count" else "scene and belt"):
            suite._recording_metadata(tmp_path, "conveyor_sort")


def test_e2e_workflow_runs_only_act_daily_at_1900_utc():
    workflow = yaml.load(
        (ROOT / ".github/workflows/simulation-application-e2e.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert workflow["on"]["schedule"] == [{"cron": "0 19 * * *"}]
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"simulation"}
    job = workflow["jobs"]["simulation"]
    assert int(job["timeout-minutes"]) >= 30 + 360 + 60
    assert [step["uses"] for step in job["steps"] if step.get("uses", "").startswith("$/")] == [
        "$/.github/actions/simulation-environment",
        "$/.github/actions/start-simulator",
    ]
    steps = job["steps"]
    controls_step = next(step for step in steps if step.get("name") == "Verify seeded expert and no-action episodes")
    training_step = next(
        step for step in steps if step.get("name") == "Train fresh model and run its deployment bundle"
    )
    assert steps.index(controls_step) < steps.index(training_step)
    assert "if" not in training_step
    assert controls_step["timeout-minutes"] == "30" and training_step["timeout-minutes"] == "360"
    assert workflow["env"]["SIM_E2E_CONTROLS_FILE"].endswith("/full-act/controls.json")
    upload = next(step for step in steps if step.get("name") == "Retain training and deployment outputs")
    assert upload["if"] == "always()"
    assert upload["with"]["name"] == "training-outputs-act"
    assert upload["with"]["retention-days"] == "2"
    evidence = next(step for step in steps if step.get("id") == "evidence")
    assert evidence["with"]["name"] == "sim-full-act"
    summary = steps[-1]
    assert summary["name"] == "Publish simulation job summary"
    assert summary["if"] == "always()"
    assert summary["env"]["SIM_E2E_ARTIFACT_URL"] == "${{ steps.evidence.outputs.artifact-url }}"
    assert summary["env"]["SIM_E2E_CONTROLS_OUTCOME"] == f"${{{{ steps.{controls_step['id']}.outcome }}}}"
    assert summary["env"]["SIM_E2E_TRAINING_OUTCOME"] == f"${{{{ steps.{training_step['id']}.outcome }}}}"
    assert summary["working-directory"] == "application/backend"
    assert "uv run --isolated --no-project --with 'defusedxml>=0.7.1' python" in summary["run"]
    assert '>> "$GITHUB_STEP_SUMMARY"' in summary["run"]


def _workflow_python(path, step):
    workflow = yaml.safe_load(path.read_text())
    steps = workflow["runs"]["steps"]
    script = next(item["run"] for item in steps if item.get("name") == step)
    return script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def test_workflow_download_uses_pinned_recording(tmp_path):
    action = ROOT / ".github/actions/simulation-environment/action.yml"
    script = tmp_path / "download.py"
    script.write_text(_workflow_python(action, "Fetch pinned recording"))
    (tmp_path / "huggingface_hub.py").write_text(
        "import json, os\ndef snapshot_download(repo_id, **kwargs):\n"
        '    print(json.dumps({"repo_id": repo_id, **kwargs}, default=str))\n'
        "    return os.environ['FAKE_HF_SNAPSHOT']\n"
    )
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"total_episodes": 100}))
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(BACKEND),
            "FAKE_HF_SNAPSHOT": str(tmp_path),
            "GITHUB_ENV": str(tmp_path / "github-env"),
        },
    )
    fetched = json.loads(result.stdout)
    assert fetched["repo_id"] == controls.DATASET["repo_id"]
    assert fetched["revision"] == controls.DATASET["revision"]
    assert fetched["repo_type"] == "dataset"
    assert fetched["allow_patterns"] == ["simulation.json", "meta/*", "data/*", "videos/*"]
    assert "local_dir" not in fetched and "cache_dir" not in fetched
    assert (tmp_path / "github-env").read_text() == f"SIM_E2E_DATASET_DIR={tmp_path}\n"
