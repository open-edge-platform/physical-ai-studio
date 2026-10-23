# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for application-owned simulation controls and evidence."""

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import yaml
from scripts.simulation_report import summarize
from tests.integration import simulation_e2e as controls
from tests.integration import test_simulation_e2e as suite

from runtime.config_builder import policy_source_fragment, policy_source_from_fragment

ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "application/backend"


def snapshot(count=0, correct=7):
    return {
        "episode": {
            "episode_count": count,
            "items_per_episode": 10,
            "last_episode": {"correct": correct, "wrong": 1, "missed": 9 - correct},
        },
        "timing": {"real_time_factor": 1.0},
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
                camera_frames={"wrist": SimpleNamespace(timestamp=float(position))},
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


def test_missing_checkpoint_is_explicitly_unqualified(tmp_path, monkeypatch):
    manifest = tmp_path / "checkpoints.json"
    manifest.write_text(json.dumps({"act": None}))
    for key, value in {
        "MODE": "evaluate",
        "POLICY": "act",
        "CHECKPOINT_MANIFEST": str(manifest),
        "OUTPUT_DIR": str(tmp_path),
        "DATASET_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(f"SIM_E2E_{key}", value)
    monkeypatch.setattr(suite, "attest_runtime", dict)
    request = MagicMock()
    with pytest.raises(pytest.skip.Exception):
        suite.test_train_export_and_run_in_simulation("conveyor_sort", tmp_path, request, monkeypatch)
    request.getfixturevalue.assert_not_called()
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "quality_baseline_unavailable"


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "main"),
        ("filename", "../model.ckpt"),
        ("filename", "nested/model.ckpt"),
        ("sha256", "bad"),
        ("min_score", 0),
        ("min_score", True),
    ],
)
def test_qualified_checkpoint_manifest_validation(tmp_path, field, value):
    entry = {
        "repo_id": "org/policy",
        "revision": "a" * 40,
        "filename": "model.ckpt",
        "sha256": "b" * 64,
        "min_score": 0.5,
    }
    entry[field] = value
    manifest = tmp_path / "checkpoints.json"
    manifest.write_text(json.dumps({"act": entry}))
    with pytest.raises(ValueError):
        controls.checkpoint_entry(manifest, "act")


def test_checkpoint_checksum(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    controls.verify_checkpoint(checkpoint, hashlib.sha256(b"checkpoint").hexdigest())
    with pytest.raises(ValueError, match="checksum mismatch"):
        controls.verify_checkpoint(checkpoint, "b" * 64)


def test_weekly_report_does_not_promote_missing_or_zero_scores(tmp_path):
    directory = tmp_path / "sim-evaluate-act"
    directory.mkdir()
    (directory / "result.json").write_text(
        json.dumps({"status": "failed", "quality": "failed", "wiring": "passed", "timing": "passed", "score": 0})
    )
    directory = tmp_path / "sim-evaluate-xr0"
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps({"status": "quality_baseline_unavailable"}))
    report = summarize(tmp_path)
    assert "| act | failed/incomplete | passed | passed | failed | 0.0% |" in report
    assert "| xr0 | failed/incomplete | not evaluated | not evaluated | baseline unavailable |" in report


def test_unsupported_hardware_is_not_a_training_pass(tmp_path):
    directory = tmp_path / "sim-train-xr0"
    directory.mkdir()
    (directory / "training.json").write_text(json.dumps({"status": "unsupported_hardware"}))
    assert "| xr0 | unsupported hardware |" in summarize(tmp_path)


@pytest.mark.parametrize("name", controls.POLICIES)
def test_trial_loads_application_recipe_and_reseeds_after_warmup(tmp_path, monkeypatch, name):
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
    suite._trials(recipe, {"fps": 30, "features": {"observation.images.wrist": {}}}, tmp_path, name, monkeypatch)
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
    monkeypatch.setattr("physicalai.inference.InferenceModel", MagicMock(return_value=model))
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
            robot=robot, action_source=policy_source_from_fragment(fragment), cameras={"wrist": camera}, fps=30
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
    monitor.on_tick(SimpleNamespace())
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


def test_video_requests_require_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(suite.shutil, "which", lambda name: None)
    with (
        pytest.raises(RuntimeError, match="requires ffmpeg"),
        suite._record_video(tmp_path, "http://localhost", ["overview"], {}, 30, "act", "conveyor_sort"),
    ):
        pass


@pytest.mark.parametrize("job", ["controls", "policy"])
@pytest.mark.parametrize("borrowed_pid", [False, True])
def test_cleanup_never_shuts_down_an_unowned_simulator(tmp_path, job, borrowed_pid):
    workflow = yaml.safe_load((ROOT / ".github/workflows/simulation-application-e2e.yml").read_text())
    cleanup = next(step["run"] for step in workflow["jobs"][job]["steps"] if step.get("name") == "Stop owner")
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
        names = ("physicalai", "physicalai-mujoco-plugin") if project == BACKEND else ("physicalai",)
        for name in names:
            expected = config["tool"]["uv"]["sources"][name]["rev"]
            package = next(package for package in lock["package"] if package["name"] == name)
            assert package["source"]["git"].endswith(f"#{expected}")
            assert expected == controls.RUNTIME_REVISION


def test_runtime_attestation_rejects_mixed_pins(monkeypatch):
    dist = MagicMock()
    dist.read_text.return_value = json.dumps({"vcs_info": {"commit_id": "f" * 40}})
    monkeypatch.setattr(controls, "distribution", lambda name: dist)
    with pytest.raises(RuntimeError, match="pinned Runtime"):
        controls.attest_runtime()


@pytest.mark.parametrize("choice", [None, "50", "100"])
def test_dataset_selection_is_pinned(monkeypatch, choice):
    monkeypatch.delenv("SIM_E2E_DATASET", raising=False)
    if choice is not None:
        monkeypatch.setenv("SIM_E2E_DATASET", choice)
    entry = controls.dataset_entry()
    assert entry["episodes"] == int(choice or "50")
    assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"])
    assert entry["repo_id"] == "Albertvh97/conveyor-sort-sim-demos" + ("-100-episodes" if choice == "100" else "")


@pytest.mark.parametrize("choice", ["", "51", "../100", "100; echo unexpected"])
def test_unknown_dataset_is_rejected(monkeypatch, choice):
    monkeypatch.setenv("SIM_E2E_DATASET", choice)
    with pytest.raises(ValueError, match="Unknown simulation dataset"):
        controls.dataset_entry()


@pytest.mark.parametrize("dataset", ["50", "100"])
def test_wrong_recording_is_rejected_before_starting_application(tmp_path, monkeypatch, dataset):
    monkeypatch.setenv("SIM_E2E_DATASET", dataset)
    (tmp_path / "simulation.json").write_text(json.dumps({"scene": "conveyor_sort", "belt_speed": 0.04}))
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"total_episodes": 100 if dataset == "50" else 50}))
    with pytest.raises(ValueError, match="episode count"):
        suite._recording_metadata(tmp_path, "conveyor_sort")


@pytest.mark.parametrize("name", ["simulation-application-e2e", "simulation-application-training"])
def test_workflows_offer_both_recordings_with_separate_download_directories(name):
    workflow = yaml.load((ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader)
    choice = workflow["on"]["workflow_dispatch"]["inputs"]["dataset"]
    assert choice["type"] == "choice" and choice["default"] == "50"
    assert set(choice["options"]) == set(controls.DATASETS)
    env = workflow["env"] if "env" in workflow else workflow["jobs"]["train"]["env"]
    assert env["SIM_E2E_DATASET"] == "${{ inputs.dataset || '50' }}"
    assert env["SIM_E2E_DATASET_DIR"].endswith(".simulation-dataset-${{ inputs.dataset || '50' }}")


def _workflow_python(path, step, composite=False):
    workflow = yaml.safe_load(path.read_text())
    steps = workflow["runs"]["steps"] if composite else workflow["jobs"]["train"]["steps"]
    script = next(item["run"] for item in steps if item.get("name") == step)
    return script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


@pytest.mark.parametrize("dataset", ["50", "100"])
def test_workflow_download_uses_selected_pins(tmp_path, dataset):
    action = ROOT / ".github/actions/simulation-environment/action.yml"
    script = tmp_path / "download.py"
    script.write_text(_workflow_python(action, "Fetch pinned recording", composite=True))
    (tmp_path / "huggingface_hub.py").write_text(
        "import json\ndef snapshot_download(repo_id, **kwargs):\n"
        '    print(json.dumps({"repo_id": repo_id, **kwargs}, default=str))\n'
    )
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"total_episodes": int(dataset)}))
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(BACKEND),
            "SIM_E2E_DATASET": dataset,
            "SIM_E2E_DATASET_DIR": str(tmp_path),
        },
    )
    fetched = json.loads(result.stdout)
    assert fetched["repo_id"] == controls.DATASETS[dataset]["repo_id"]
    assert fetched["revision"] == controls.DATASETS[dataset]["revision"]
    assert fetched["repo_type"] == "dataset"
    assert fetched["allow_patterns"] == ["simulation.json", "meta/*", "data/*", "videos/*"]


@pytest.mark.parametrize("mismatch", [None, "policy", "source_sha", "dataset_repo_id", "dataset_revision", "workflow"])
def test_base_model_requires_matching_application_provenance(tmp_path, mismatch):
    workflow = ROOT / ".github/workflows/simulation-application-training.yml"
    script = tmp_path / "base.py"
    script.write_text(_workflow_python(workflow, "Validate application base model"))
    resume = tmp_path / ".simulation-resume"
    model_id = str(uuid4())
    model_dir = resume / "studio/models" / model_id
    (model_dir / "exports/torch").mkdir(parents=True)
    (model_dir / "model.ckpt").touch()
    (model_dir / "exports/torch/manifest.json").write_text("{}")
    (resume / "application-model.json").write_text(json.dumps({"id": model_id}))
    entry = controls.DATASETS["100"]
    provenance = {
        "policy": "act",
        "source_sha": "a" * 40,
        "workflow": "application_jobs_runtime_recipe",
        "dataset_repo_id": entry["repo_id"],
        "dataset_revision": entry["revision"],
    }
    if mismatch is not None:
        provenance[mismatch] = "wrong"
    (resume / "provenance.json").write_text(json.dumps(provenance))
    envfile = tmp_path / "env"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(BACKEND),
            "SIM_E2E_DATASET": "100",
            "SIM_E2E_POLICY": "act",
            "GITHUB_SHA": "a" * 40,
            "GITHUB_ENV": str(envfile),
            "GITHUB_WORKSPACE": str(tmp_path),
        },
    )
    if mismatch is None:
        assert result.returncode == 0, result.stderr
        assert envfile.read_text() == f"SIM_E2E_BASE_MODEL_DIR={model_dir}\n"
    else:
        assert result.returncode != 0 and "same policy, source revision and dataset" in result.stderr
        assert not envfile.exists()
