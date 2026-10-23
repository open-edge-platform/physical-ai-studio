# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in simulation E2E through Studio jobs, model import, and Runtime recipes.

The harness owns fixtures, simulator controls, scoring, and evidence only.
Training, snapshots, normalization, export, camera configuration, and action
execution belong to the application and its normal background workers.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import torch
from physicalai.config import Config
from physicalai.data.archive_safety import SafeZipArchive
from physicalai.runtime import RobotRuntime

from tests.integration.application_flow import wait_for_job, wait_for_model
from tests.integration.simulation_e2e import (
    POLICIES,
    SEEDS,
    EpisodeMonitor,
    attest_runtime,
    checkpoint_entry,
    command,
    dataset_entry,
    prepare_episode,
    score,
    verify_checkpoint,
    wait_health,
)
from training.job import CHECKPOINT_NAME, TrainingJobSpec, _export, build_policy, resolve_checkpoint

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi.testclient import TestClient

pytestmark = [pytest.mark.integration, pytest.mark.slow]
SCENES = ("conveyor_sort",)


def _task_instruction(dataset: Path) -> str:
    """Read the single conveyor task without silently choosing an instruction."""
    tasks = pd.read_parquet(dataset / "meta/tasks.parquet")
    if len(tasks) != 1 or not isinstance(tasks.index[0], str) or not tasks.index[0].strip():
        raise ValueError("Simulation recording must contain exactly one non-empty task instruction")
    return tasks.index[0]


def _recording_metadata(dataset: Path, scene: str) -> dict:
    recording = json.loads((dataset / "simulation.json").read_text())
    if recording["scene"] != scene or recording["belt_speed"] != 0.04:
        raise ValueError("Recording does not match the conveyor scene and belt speed")
    metadata = json.loads((dataset / "meta/info.json").read_text())
    if metadata["total_episodes"] != dataset_entry()["episodes"]:
        raise ValueError("Recording episode count does not match selected dataset")
    if metadata["fps"] != 30:
        raise ValueError("Conveyor recording must use its native 30 FPS clock")
    for key in ("action", "observation.state"):
        if metadata["features"][key]["shape"] != [6]:
            raise ValueError(f"Conveyor {key} must contain six normalized joints")
    return metadata


def _smoke_dataset(source: Path, destination: Path) -> Path:
    """Keep 20 contiguous frames from three episodes; this is not a quality dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    recording = LeRobotDataset(dataset_entry()["repo_id"], root=source, episodes=[0, 1, 2])
    keys = [
        key
        for key in recording.features
        if key in {"action", "observation.state"} or key.startswith("observation.images.")
    ]
    features = {key: {name: value for name, value in recording.features[key].items() if name != "info"} for key in keys}
    for key, feature in features.items():
        if key.startswith("observation.images."):
            feature["dtype"] = "image"
    fixture = LeRobotDataset.create(
        repo_id="physicalai-studio-tests/conveyor-smoke",
        root=destination,
        fps=recording.fps,
        features=features,
        robot_type="so101_follower",
        use_videos=False,
    )
    for episode in range(3):
        start = recording.meta.episodes[episode]["dataset_from_index"]
        for index in range(start, start + 20):
            sample = recording[index]
            frame = {}
            for key in keys:
                value = sample[key]
                frame[key] = (
                    (value.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    if key.startswith("observation.images.")
                    else value.numpy()
                )
            fixture.add_frame({**frame, "task": sample["task"]})
        fixture.save_episode()
    fixture.finalize()
    (destination / "smoke-fixture.json").write_text(
        json.dumps(
            {
                "purpose": "API/training/export wiring only; not behavioral qualification",
                "source": dataset_entry(),
                "episodes": [0, 1, 2],
                "contiguous_frames_per_episode": 20,
            }
        )
    )
    return destination


def _project_environment(client: TestClient, metadata: dict, *, simulation: bool) -> tuple[str, str]:
    project_id, environment_id = str(uuid4()), str(uuid4())
    response = client.post("/api/projects", json={"id": project_id, "name": "Simulation application E2E"})
    assert response.status_code == 201, response.text
    robots, cameras = [], []
    if simulation:
        robot_id = str(uuid4())
        response = client.post(
            f"/api/projects/{project_id}/robots",
            json={
                "id": robot_id,
                "name": "Conveyor follower",
                "type": "MuJoCo_SO101_Follower",
                "payload": {"name": os.environ["SIM_E2E_OWNER"], "allow_remote": False, "connect_timeout": 10},
            },
        )
        assert response.status_code == 201, response.text
        robots.append({"robot_id": robot_id, "tele_operator": {"type": "none"}})
        url = os.environ["SIM_E2E_HTTP_URL"].rstrip("/")
        camera_map = json.loads(os.environ.get("SIM_E2E_CAMERA_MAP", "{}"))
        for key, feature in metadata["features"].items():
            if not key.startswith("observation.images."):
                continue
            name = key.removeprefix("observation.images.")
            camera_id = str(uuid4())
            camera_url = f"{url}/cameras/{camera_map.get(name, name)}/mjpeg"
            response = client.post(
                f"/api/projects/{project_id}/cameras",
                json={
                    "id": camera_id,
                    "name": name,
                    "driver": "ipcam",
                    "fingerprint": {"url": camera_url},
                    "hardware_name": f"Conveyor {name}",
                    "payload": {
                        "url": camera_url,
                        "fps": metadata["fps"],
                        "height": feature["shape"][0],
                        "width": feature["shape"][1],
                    },
                },
            )
            assert response.status_code == 201, response.text
            cameras.append({"camera_id": camera_id})
    response = client.post(
        f"/api/projects/{project_id}/environments",
        json={
            "id": environment_id,
            "name": "Conveyor simulation" if simulation else "Offline training",
            "robots": robots,
            "cameras": cameras,
        },
    )
    assert response.status_code == 201, response.text
    return project_id, environment_id


def _import_dataset(client: TestClient, project: str, environment: str, dataset: Path, task: str, output: Path) -> str:
    archive = output / "dataset-upload.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(dataset.rglob("*")):
            if path.is_file() and ".cache" not in path.relative_to(dataset).parts:
                bundle.write(path, path.relative_to(dataset))
    response = client.post(
        f"/api/projects/{project}/imports/datasets:prepare",
        data={"format_hint": "auto", "dataset_name": "Conveyor E2E fixture"},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    with archive.open("rb") as stream:
        response = client.put(
            f"/api/projects/{project}/imports/datasets/{job_id}:upload",
            files={"archive": (archive.name, stream, "application/zip")},
        )
    assert response.status_code == 202, response.text
    wait_for_job(client, job_id, step="awaiting_user_review")
    response = client.post(
        f"/api/projects/{project}/imports/datasets/{job_id}:finalize",
        json={"environment_id": environment, "default_task": task},
    )
    assert response.status_code == 202, response.text
    imported = wait_for_job(client, job_id)
    archive.unlink()
    return imported["payload"]["result_dataset_id"]


def _import_model(client: TestClient, project: str, dataset: str, directory: Path) -> dict:
    # Use the supported operator CLI, rather than hand-creating model/job rows.
    subprocess.run(
        [
            str(Path(sys.executable).with_name("physicalai-studio")),
            "models",
            "import-dir",
            "--copy",
            "--source-dir",
            str(directory),
            "--project-id",
            project,
            "--dataset-id",
            dataset,
            "--model-name",
            "Imported conveyor policy",
        ],
        check=True,
        timeout=300,
    )
    response = client.get(f"/api/projects/{project}/models")
    assert response.status_code == 200, response.text
    models = response.json()
    assert len(models) == 1, models
    return models[0]


def _train(
    client: TestClient,
    name: str,
    project: str,
    dataset: str,
    output: Path,
    *,
    smoke: bool,
    base_model: dict | None = None,
    evaluate: bool = False,
) -> tuple[dict, dict]:
    response = client.get("/api/policies/available")
    assert response.status_code == 200, response.text
    available = {policy["id"]: policy for policy in response.json()}
    assert name in available, f"Studio cannot select the requested policy: {name}"
    accelerator = os.environ.get("SIM_E2E_ACCELERATOR", "cuda")
    if accelerator == "cuda":
        assert torch.cuda.is_available(), "Application E2E requested CUDA but it is unavailable"
        capacity = math.ceil(torch.cuda.get_device_properties(0).total_memory / 1024**3)
        required = math.ceil((available[name]["min_vram"] or 0) / 1024**3)
        if capacity < required:
            reason = f"Studio advertises {required} GiB for {name}; selected device has {capacity} GiB"
            (output / "training.json").write_text(
                json.dumps({"policy": name, "status": "unsupported_hardware", "reason": reason})
            )
            pytest.skip(reason)
    elif accelerator != "cpu" or name != "act":
        raise ValueError("Use CUDA for the policy matrix, or CPU for the ACT application smoke")
    payload = {
        "project_id": project,
        "dataset_id": dataset,
        "policy": name,
        "model_name": f"Conveyor {name}",
        "max_epochs": 1 if smoke else int(os.environ.get("SIM_E2E_EPOCHS", "3")),
        "batch_size": 2,
        "num_workers": 0,
        "val_split": 0.2,
        "device": {"type": accelerator, **({"index": 0} if accelerator == "cuda" else {})},
        "precision": "32-true" if name == "act" else "bf16-true",
        "export_backends": ["torch", "openvino"] if evaluate else ["torch"],
    }
    if base_model is not None:
        payload["base_model_id"] = base_model["id"]
    (output / "training-request.json").write_text(json.dumps(payload))
    start = time.monotonic()
    response = client.post("/api/jobs:train", json=payload)
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    job = wait_for_job(client, job_id, timeout=float(os.environ.get("SIM_E2E_JOB_TIMEOUT_SECONDS", "1200")))
    model = wait_for_model(client, project, job_id)
    (output / "application-job.json").write_text(json.dumps(job))
    (output / "application-model.json").write_text(json.dumps(model))
    model_dir = Path(model["path"])
    checkpoint = resolve_checkpoint(model_dir)
    assert checkpoint.is_file(), "Studio must publish its selected checkpoint"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    steps = state["global_step"]
    assert steps > 0 and state["optimizer_states"], "Published checkpoint must include optimizer state"
    optimizer_steps = [
        float(value["step"])
        for optimizer in state["optimizer_states"]
        for value in optimizer["state"].values()
        if "step" in value
    ]
    assert optimizer_steps and max(optimizer_steps) == steps, "Optimizer state must confirm real updates"
    assert not smoke or steps == 20, "Three 20-frame episodes, batch two and one epoch must make 20 updates"
    del state
    restored = build_policy(TrainingJobSpec(policy=name), resume_from=checkpoint)
    del restored
    assert "torch" in model["available_backends"], "Application did not publish its requested Torch export"
    if evaluate:
        assert "openvino" in model["available_backends"], "Application did not publish its requested OpenVINO export"
    from schemas.model import Model
    from services.model_metrics_service import ModelMetricsService

    metrics_path = asyncio.run(ModelMetricsService.get_model_metrics_path(Model.model_validate(model)))
    metrics = pd.read_csv(metrics_path, keep_default_na=False)
    validation = metrics["val/loss"].loc[lambda values: values != ""].astype(float)
    assert not validation.empty and np.isfinite(validation).all(), "Application validation loss must be finite"
    for column in (key for key in metrics if key.startswith("train/loss")):
        values = metrics[column].loc[lambda values: values != ""].astype(float)
        assert np.isfinite(values).all(), "Application training loss must be finite when logged"
    training = {
        "checkpoint_steps": steps,
        "training_seconds": time.monotonic() - start,
        "checkpoint_reload": "passed",
        "job_id": job_id,
        "model_id": model["id"],
        "validation_loss": float(validation.iloc[-1]),
        "training_payload": payload,
        "fixture": "wiring_only" if smoke else "full_recording",
    }
    return model, training


def _runtime_bundle(client: TestClient, model: dict, environment: str, task: str, output: Path) -> Path:
    response = client.get(
        f"/api/models/{model['id']}/exports/openvino/download",
        params={"environment_id": environment, "device": "CPU", "task": task},
    )
    assert response.status_code == 200, response.text
    archive = output / "runtime.zip"
    archive.write_bytes(response.content)
    destination = output / "runtime"
    SafeZipArchive(archive, max_uncompressed_bytes=100 * 1024**3, max_file_count=10000).extract_to(destination)
    recipe = destination / "runtime.yaml"
    assert recipe.is_file()
    Config.load(recipe)  # Validate the application-generated construction recipe.
    return recipe


@contextmanager
def _record_video(
    output_dir: Path, url: str, cameras: list[str], camera_map: dict[str, str], fps: int, policy: str, scene: str
) -> Iterator[Path]:
    """Stream simulator cameras to MP4 without buffering frames in memory."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("SIM_E2E_RECORD_VIDEO requires ffmpeg on PATH")
    path = output_dir / f"inference-{policy}-{scene}.mp4"
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning", "-y"]
    for camera in cameras:
        args += [
            "-thread_queue_size",
            "512",
            "-use_wallclock_as_timestamps",
            "1",
            "-i",
            f"{url.rstrip('/')}/cameras/{camera_map.get(camera, camera)}/mjpeg",
        ]
    if len(cameras) == 1:
        args += ["-vf", f"fps={fps},setsar=1"]
    else:
        streams = ";".join(f"[{i}:v]fps={fps},setsar=1[v{i}]" for i in range(len(cameras)))
        stack = "".join(f"[v{i}]" for i in range(len(cameras)))
        args += ["-filter_complex", f"{streams};{stack}hstack=inputs={len(cameras)}[v]", "-map", "[v]"]
    args += [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    log_path = output_dir / "inference-video-ffmpeg.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            time.sleep(1)
            if process.poll() is not None:
                raise RuntimeError(f"ffmpeg failed to start; see {log_path}")
            yield path
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError(f"ffmpeg did not finish recording; see {log_path}") from None
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"ffmpeg did not save an inference video; see {log_path}")


def _trials(recipe: Path, metadata: dict, output: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    import physicalai.runtime.core as runtime_core

    url = os.environ["SIM_E2E_HTTP_URL"]
    cameras = sorted(
        key.removeprefix("observation.images.") for key in metadata["features"] if key.startswith("observation.images.")
    )
    assert cameras
    camera_map = json.loads(os.environ.get("SIM_E2E_CAMERA_MAP", "{}"))
    recorder = (
        _record_video(output, url, cameras, camera_map, int(metadata["fps"]), name, "conveyor_sort")
        if os.environ.get("SIM_E2E_RECORD_VIDEO", "false").lower() == "true"
        else nullcontext(None)
    )
    trials = []
    with recorder, monkeypatch.context() as context:
        context.chdir(recipe.parent)
        for seed in SEEDS:
            before = prepare_episode(url, seed, "off", 0.04)
            stop = threading.Event()
            monitor = EpisodeMonitor(url, before, stop, float(os.environ.get("SIM_E2E_ROLLOUT_SECONDS", "180")))
            # Observe the unmodified recipe; the SDK config loader has no live-callback argument.
            callback_bus = runtime_core._CallbackBus
            with monkeypatch.context() as observer:
                observer.setattr(runtime_core, "_CallbackBus", lambda callbacks: callback_bus([*callbacks, monitor]))
                runtime = RobotRuntime.from_config(recipe)
            try:
                with runtime:
                    monitor.recording = False
                    runtime.run(stop_event=stop)
                    stop.clear()
                    before = prepare_episode(url, seed, "off", 0.04)
                    monitor.begin(before)
                    command(url, "episode/auto-reset", {"enabled": True})
                    wait_health(url, lambda health: health["episode"]["active"])
                    runtime.run(stop_event=stop)
                monitor.close()
                assert monitor.actions_sent > 0 and monitor.max_joint_change > 0.5
                trials.append(monitor.result(seed))
                (output / "rollouts.json").write_text(json.dumps(trials))
            finally:
                monitor.finished.set()
                if monitor.thread.ident is not None:
                    monitor.thread.join(timeout=6)
                runtime.disconnect()
                command(url, "episode/auto-reset", {"enabled": False})
    return trials


@pytest.mark.parametrize("scene", SCENES)
def test_train_export_and_run_in_simulation(
    scene: str, tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use normal application jobs and Runtime recipes, never a parallel trainer/controller."""
    name = os.environ.get("SIM_E2E_POLICY", "act")
    mode = os.environ.get("SIM_E2E_MODE", "full")
    if name not in POLICIES or mode not in {"train", "evaluate", "full"}:
        raise ValueError("Unknown E2E policy or mode")
    if "SIM_E2E_DATASET_DIR" not in os.environ:
        pytest.skip("Set SIM_E2E_DATASET_DIR to opt in to the application simulation E2E")
    output = Path(os.environ.get("SIM_E2E_OUTPUT_DIR", str(tmp_path))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    recording_entry = dataset_entry()
    provenance = {
        "policy": name,
        "scene": scene,
        "mode": mode,
        "source_sha": os.environ.get("GITHUB_SHA"),
        "dataset_repo_id": recording_entry["repo_id"],
        "dataset_revision": recording_entry["revision"],
        "runtime_commits": attest_runtime(),
        "workflow": "application_jobs_runtime_recipe",
    }
    (output / "provenance.json").write_text(json.dumps(provenance))
    entry = None
    if mode == "evaluate":
        entry = checkpoint_entry(
            Path(os.environ.get("SIM_E2E_CHECKPOINT_MANIFEST", "configs/simulation-e2e/checkpoints.json")), name
        )
        if entry is None:
            (output / "result.json").write_text(json.dumps({**provenance, "status": "quality_baseline_unavailable"}))
            pytest.skip(f"No qualified checkpoint for {name}; quality baseline unavailable")
    dataset = Path(os.environ["SIM_E2E_DATASET_DIR"]).resolve()
    metadata = _recording_metadata(dataset, scene)
    task = _task_instruction(dataset)
    smoke = mode == "train" and os.environ.get("SIM_E2E_FULL_DATASET", "false").lower() != "true"
    fixture = _smoke_dataset(dataset, tmp_path / "smoke") if smoke else dataset
    client = request.getfixturevalue("application_client")
    project, environment = _project_environment(client, metadata, simulation=mode != "train")
    dataset_id = _import_dataset(client, project, environment, fixture, task, output)
    if entry is not None:
        from huggingface_hub import hf_hub_download

        checkpoint = Path(hf_hub_download(entry["repo_id"], entry["filename"], revision=entry["revision"]))
        verify_checkpoint(checkpoint, entry["sha256"])
        provenance["checkpoint"] = entry
        staged = output / "qualified-model"
        staged.mkdir()
        shutil.copy2(checkpoint, staged / CHECKPOINT_NAME)
        # Stage a reviewed checkpoint with the same loader/exporter used by application training.
        policy = build_policy(TrainingJobSpec(policy=name), resume_from=checkpoint)
        _export(policy, staged, lambda *_args: None, ["openvino"])
        del policy
        model = _import_model(client, project, dataset_id, staged)
    else:
        base_dir = os.environ.get("SIM_E2E_BASE_MODEL_DIR")
        base = _import_model(client, project, dataset_id, Path(base_dir)) if base_dir else None
        model, training = _train(
            client, name, project, dataset_id, output, smoke=smoke, base_model=base, evaluate=mode == "full"
        )
        provenance.update(training)
        (output / "training.json").write_text(json.dumps({**provenance, "status": "passed"}))
        if mode == "train":
            return
    recipe = _runtime_bundle(client, model, environment, task, output)
    trials = _trials(recipe, metadata, output, name, monkeypatch)
    mean_score = score([episode for trial in trials for episode in trial["episodes"]])
    timing_ok = all(
        trial["control_hz"] is not None
        and trial["control_hz"] >= metadata["fps"] * 0.8
        and 0.9 <= (trial["simulator_timing"].get("real_time_factor") or 0) <= 1.1
        and bool(trial["simulator_timing"].get("camera_fps"))
        and all((rate or 0) >= metadata["fps"] * 0.8 for rate in trial["simulator_timing"]["camera_fps"].values())
        for trial in trials
    )
    quality_ok = entry is not None and mean_score >= entry["min_score"]
    result = {
        **provenance,
        "status": "passed" if quality_ok and timing_ok else "failed",
        "wiring": "passed",
        "quality": "passed" if quality_ok else ("failed" if entry else "unqualified"),
        "timing": "passed" if timing_ok else "failed",
        "score": mean_score,
        "rollouts": trials,
        "model_id": model["id"],
        "project_id": project,
        "dataset_id": dataset_id,
    }
    (output / "result.json").write_text(json.dumps(result))
    assert timing_ok, "Control/camera rates or simulator real-time factor violate the timing contract"
    if entry is not None:
        assert quality_ok, "Sorting score is below the qualified checkpoint threshold"


def test_application_simulation_environment(application_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the simulator's devices through the real application API without connecting."""
    pytest.importorskip("physicalai_mujoco_plugin")
    monkeypatch.setenv("SIM_E2E_OWNER", "simulation-registration-test")
    monkeypatch.setenv("SIM_E2E_HTTP_URL", "http://127.0.0.1:8080")
    metadata = {"fps": 30, "features": {"observation.images.overview": {"shape": [480, 640, 3]}}}
    _project_environment(application_client, metadata, simulation=True)


def test_application_training_smoke(
    application_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the migrated job path on CPU without a simulator or physical devices."""
    from tests.integration.conftest import _build_synthetic_lerobot_v3_dataset

    monkeypatch.setenv("SIM_E2E_ACCELERATOR", "cpu")
    source = _build_synthetic_lerobot_v3_dataset(tmp_path / "source", num_episodes=3, frames_per_episode=25)
    fixture = _smoke_dataset(source, tmp_path / "smoke")
    output = tmp_path / "evidence"
    output.mkdir()
    project, environment = _project_environment(application_client, {}, simulation=False)
    dataset = _import_dataset(application_client, project, environment, fixture, "do the thing", output)
    model, training = _train(application_client, "act", project, dataset, output, smoke=True)
    assert training["checkpoint_steps"] == 20
    assert training["checkpoint_reload"] == "passed"
    assert model["available_backends"] == ["torch"]
    assert resolve_checkpoint(Path(model["path"])).is_file()


def test_simulator_controls(tmp_path: Path) -> None:
    """The scripted expert must beat no-action on identical seeded episodes."""
    if "SIM_E2E_HTTP_URL" not in os.environ:
        pytest.skip("Set SIM_E2E_HTTP_URL to opt in to simulator controls")
    commits = attest_runtime()
    url = os.environ["SIM_E2E_HTTP_URL"]
    output = Path(os.environ.get("SIM_E2E_OUTPUT_DIR", str(tmp_path)))
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    try:
        for mode in ("off", "drive"):
            trials = []
            for seed in SEEDS:
                before = prepare_episode(url, seed, mode, 0.04)
                stop = threading.Event()
                monitor = EpisodeMonitor(url, before, stop, 180)
                command(url, "episode/auto-reset", {"enabled": True})
                wait_health(url, lambda health: health["episode"]["active"])
                monitor.thread.start()
                stop.wait(190)
                monitor.close()
                trials.append(monitor.result(seed))
            results[mode] = {
                "score": score([episode for trial in trials for episode in trial["episodes"]]),
                "trials": trials,
            }
            (output / "controls.json").write_text(
                json.dumps(
                    {
                        **results,
                        "status": "incomplete",
                        "runtime_commits": commits,
                        "source_sha": os.environ.get("GITHUB_SHA"),
                    }
                )
            )
        passed = results["drive"]["score"] > results["off"]["score"]
        (output / "controls.json").write_text(
            json.dumps(
                {
                    **results,
                    "status": "passed" if passed else "failed",
                    "runtime_commits": commits,
                    "source_sha": os.environ.get("GITHUB_SHA"),
                }
            )
        )
        assert passed, "Scripted expert does not beat no-action baseline"
    finally:
        command(url, "autopilot", {"mode": "off"})
        command(url, "episode/auto-reset", {"enabled": False})
