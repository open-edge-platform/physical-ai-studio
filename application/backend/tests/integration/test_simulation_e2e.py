# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in simulation E2E through fresh Studio training jobs and Runtime recipes.

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
import subprocess
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

from schemas.job import LocalTrainJobPayload
from tests.integration.application_flow import wait_for_job, wait_for_model
from tests.integration.simulation_e2e import (
    DATASET,
    SEEDS,
    EpisodeMonitor,
    command,
    control_scores,
    prepare_episode,
    runtime_versions,
    score,
    timing_passes,
    wait_health,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi.testclient import TestClient

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _task_instruction(dataset: Path) -> str:
    """Read the single conveyor task without silently choosing an instruction."""
    tasks = pd.read_parquet(dataset / "meta/tasks.parquet")
    if len(tasks) != 1 or not isinstance(tasks.index[0], str) or not tasks.index[0].strip():
        raise ValueError("Simulation recording must contain exactly one non-empty task instruction")
    return tasks.index[0]


def _recording_metadata(dataset: Path, scene: str) -> dict:
    recording = json.loads((dataset / "simulation.json").read_text())
    entry = DATASET
    belt_speeds = recording.get("belt_speeds_m_s", [recording.get("belt_speed")])
    if recording["scene"] != scene or belt_speeds != entry["belt_speeds"]:
        raise ValueError("Recording does not match the conveyor scene and belt speeds")
    metadata = json.loads((dataset / "meta/info.json").read_text())
    if metadata["total_episodes"] != entry["episodes"]:
        raise ValueError("Recording episode count does not match selected dataset")
    if metadata["fps"] != 30:
        raise ValueError("Conveyor recording must use its native 30 FPS clock")
    for key in ("action", "observation.state"):
        if metadata["features"][key]["shape"] != [6]:
            raise ValueError(f"Conveyor {key} must contain six normalized joints")
    return metadata


def _project_environment(client: TestClient, metadata: dict) -> tuple[str, str]:
    project_id, environment_id = str(uuid4()), str(uuid4())
    response = client.post("/api/projects", json={"id": project_id, "name": "Simulation application E2E"})
    assert response.status_code == 201, response.text
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
    cameras = []
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
            "name": "Conveyor simulation",
            "robots": [{"robot_id": robot_id, "tele_operator": {"type": "none"}}],
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


def _train(
    client: TestClient,
    name: str,
    project: str,
    dataset: str,
    output: Path,
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
        assert capacity >= required, f"Studio requires {required} GiB for {name}; selected device has {capacity} GiB"
    elif accelerator != "cpu" or name != "act":
        raise ValueError("Use CUDA for the policy matrix, or CPU for ACT application training")
    # API defaults match the UI; only precision depends on the selected device.
    payload = LocalTrainJobPayload(
        project_id=project,
        dataset_id=dataset,
        policy=name,
        model_name=f"Conveyor {name}",
        device={"type": accelerator, **({"index": 0} if accelerator == "cuda" else {})},
        precision="bf16-mixed" if accelerator == "cuda" else "32-true",
    ).model_dump(mode="json")
    (output / "training-request.json").write_text(json.dumps(payload))
    start = time.monotonic()
    response = client.post("/api/jobs:train", json=payload)
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    job = wait_for_job(client, job_id, timeout=float(os.environ.get("SIM_E2E_JOB_TIMEOUT_SECONDS", "18000")))
    model = wait_for_model(client, project, job_id)
    (output / "application-job.json").write_text(json.dumps(job))
    (output / "application-model.json").write_text(json.dumps(model))
    assert "openvino" in model["available_backends"], "Application did not publish its requested OpenVINO export"
    from schemas.model import Model
    from services.model_metrics_service import ModelMetricsService

    metrics_path = asyncio.run(ModelMetricsService.get_model_metrics_path(Model.model_validate(model)))
    metrics = pd.read_csv(metrics_path, keep_default_na=False)
    epochs = metrics["epoch"].loc[lambda values: values != ""].astype(float)
    assert epochs.max() == payload["max_epochs"] - 1, "Training must complete the UI's default epoch budget"
    validation = metrics["val/loss"].loc[lambda values: values != ""].astype(float)
    assert not validation.empty and np.isfinite(validation).all(), "Application validation loss must be finite"
    for column in (key for key in metrics if key.startswith("train/loss")):
        values = metrics[column].loc[lambda values: values != ""].astype(float)
        assert np.isfinite(values).all(), "Application training loss must be finite when logged"
    training = {
        "training_seconds": time.monotonic() - start,
        "job_id": job_id,
        "model_id": model["id"],
        "validation_loss": float(validation.iloc[-1]),
        "training_payload": payload,
        "training_epochs": int(epochs.max()) + 1,
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
    args = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-xerror", "-y"]
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
        args += ["-filter_complex", f"{streams};{stack}hstack=inputs={len(cameras)}:shortest=1[v]", "-map", "[v]"]
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
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            time.sleep(1)
            if process.poll() is not None:
                raise RuntimeError(f"ffmpeg failed to start; see {log_path}")
            started = time.monotonic()
            yield path
        finally:
            elapsed = time.monotonic() - started
            if process.poll() is None:
                try:
                    process.communicate(input=b"q\n", timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError(f"ffmpeg did not finish recording; see {log_path}") from None
            if process.returncode != 0:
                raise RuntimeError(f"ffmpeg failed while recording; see {log_path}")
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"ffmpeg did not save an inference video; see {log_path}")
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-xerror",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-progress",
                    "pipe:1",
                    "-nostats",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=max(120, elapsed),
                check=True,
            )
            progress = dict(line.split("=", 1) for line in decoded.stdout.splitlines() if "=" in line)
            duration = int(progress.get("out_time_us", "0")) / 1e6
            frames = int(progress.get("frame", "0"))
            if frames < max(1, (elapsed - 1) * fps) or duration <= 0 or duration < elapsed - 1:
                raise RuntimeError("Inference video is empty or does not cover the full rollout (1s tolerance)")


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


def test_train_export_and_run_in_simulation(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use normal application jobs and Runtime recipes, never a parallel trainer/controller."""
    name, scene = "act", "conveyor_sort"
    if "SIM_E2E_DATASET_DIR" not in os.environ:
        pytest.skip("Set SIM_E2E_DATASET_DIR to opt in to the application simulation E2E")
    output = Path(os.environ.get("SIM_E2E_OUTPUT_DIR", str(tmp_path))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    recording_entry = DATASET
    provenance = {
        "policy": name,
        "scene": scene,
        "source_sha": os.environ.get("GITHUB_SHA"),
        "dataset_repo_id": recording_entry["repo_id"],
        "dataset_revision": recording_entry["revision"],
        "runtime_versions": runtime_versions(),
        "workflow": "application_jobs_runtime_recipe",
    }
    (output / "provenance.json").write_text(json.dumps(provenance))
    controls = control_scores(Path(os.environ["SIM_E2E_CONTROLS_FILE"]))
    dataset = Path(os.environ["SIM_E2E_DATASET_DIR"]).resolve()
    metadata = _recording_metadata(dataset, scene)
    task = _task_instruction(dataset)
    client = request.getfixturevalue("application_client")
    project, environment = _project_environment(client, metadata)
    dataset_id = _import_dataset(client, project, environment, dataset, task, output)
    model, training = _train(client, name, project, dataset_id, output)
    provenance.update(training)
    (output / "training.json").write_text(json.dumps({**provenance, "status": "passed"}))
    recipe = _runtime_bundle(client, model, environment, task, output)
    trials = _trials(recipe, metadata, output, name, monkeypatch)
    mean_score = score([episode for trial in trials for episode in trial["episodes"]])
    timing_ok = timing_passes(
        trials,
        metadata["fps"],
        [
            key.removeprefix("observation.images.")
            for key in metadata["features"]
            if key.startswith("observation.images.")
        ],
        camera_map=json.loads(os.environ.get("SIM_E2E_CAMERA_MAP", "{}")),
    )
    quality_ok = mean_score > controls["off"]
    result = {
        **provenance,
        "status": "passed" if quality_ok and timing_ok else "failed",
        "wiring": "passed",
        "quality": "passed" if quality_ok else "failed",
        "controls": controls,
        "timing": "passed" if timing_ok else "failed",
        "score": mean_score,
        "rollouts": trials,
        "model_id": model["id"],
        "project_id": project,
        "dataset_id": dataset_id,
    }
    (output / "result.json").write_text(json.dumps(result))
    assert timing_ok, "Control/camera rates or simulator real-time factor violate the timing contract"
    assert quality_ok, "Freshly trained policy must beat the same-seed no-action control"


def test_application_simulation_environment(application_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the simulator's devices through the real application API without connecting."""
    pytest.importorskip("physicalai_mujoco_plugin")
    monkeypatch.setenv("SIM_E2E_OWNER", "simulation-registration-test")
    monkeypatch.setenv("SIM_E2E_HTTP_URL", "http://127.0.0.1:8080")
    metadata = {"fps": 30, "features": {"observation.images.overview": {"shape": [480, 640, 3]}}}
    _project_environment(application_client, metadata)


def test_simulator_controls(tmp_path: Path) -> None:
    """The scripted expert must beat no-action on identical seeded episodes."""
    if "SIM_E2E_HTTP_URL" not in os.environ:
        pytest.skip("Set SIM_E2E_HTTP_URL to opt in to simulator controls")
    versions = runtime_versions()
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
                        "runtime_versions": versions,
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
                    "runtime_versions": versions,
                    "source_sha": os.environ.get("GITHUB_SHA"),
                }
            )
        )
        assert passed, "Scripted expert does not beat no-action baseline"
    finally:
        command(url, "autopilot", {"mode": "off"})
        command(url, "episode/auto-reset", {"enabled": False})
