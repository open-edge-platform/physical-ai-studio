# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in train -> export -> live Runtime rollout against a MuJoCo owner.

Start the physicalai-mujoco simulation separately. Point SIM_E2E_DATASET_DIR
at a LeRobot dataset recorded in that scene, not a synthetic fixture. This
suite deliberately fails if its inputs are absent when explicitly selected.
The default budget trains for three full epochs. Set SIM_E2E_TRAIN_STEPS
for a shorter, step-limited smoke run.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator
from urllib.request import Request, urlopen

import numpy as np
import pytest
from lightning.pytorch.callbacks import ModelCheckpoint
from physicalai.capture import IPCamera
from physicalai.robot import SharedRobot
from physicalai.runtime import PolicySource, RobotRuntime, SyncExecution
from physicalai.runtime.events import TickEvent

from physicalai.data import LeRobotDataModule
from physicalai.inference import InferenceModel
from physicalai.policies import get_policy
from physicalai.train import Trainer

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SCENES = ("conveyor_sort", "single_pick_place")
FIRST_PARTY_POLICIES = ("act", "molmoact2", "pi05", "rldx1", "smolvla", "xr0")


def _score(scene: str, episodes: list[dict]) -> float:
    """Score every completed episode, never just the simulator's most recent one."""
    if scene == "conveyor_sort":
        if not episodes:
            raise AssertionError("Conveyor episode did not finish within the rollout deadline")
        total = sum(result[key] for result in episodes for key in ("correct", "wrong", "missed"))
        if total <= 0:
            raise AssertionError("Conveyor episode has no scored items")
        return sum(result["correct"] for result in episodes) / total
    if scene == "single_pick_place":
        # A timed trial with zero completed successes is a valid performance result.
        return float(bool(episodes))
    raise ValueError(f"Unsupported scene: {scene}")


def _health(url: str) -> dict:
    with urlopen(f"{url.rstrip('/')}/health", timeout=5) as response:  # noqa: S310 - explicit local simulator
        return json.load(response)


def _reset(url: str) -> None:
    with urlopen(Request(f"{url.rstrip('/')}/reset", method="POST"), timeout=5):  # noqa: S310 - local simulator
        pass


class _RolloutMonitor:
    """Count actions, observed joint motion and complete simulator episodes."""

    def __init__(self, url: str, before: dict) -> None:
        self.url = url
        self.episode_count = before["episode"]["episode_count"]
        self.episodes: list[dict] = []
        self.actions_sent = 0
        self.max_joint_change = 0.0
        self.initial_joints: np.ndarray | None = None
        self.error: Exception | None = None
        self.next_poll = time.monotonic() + 1

    def on_tick(self, event: TickEvent) -> None:
        """Inspect robot observations after Runtime has sent each policy action."""
        if event.action_sent is not None:
            self.actions_sent += 1
        joints = np.asarray(event.robot_state.joint_positions)
        if self.initial_joints is None:
            self.initial_joints = joints.copy()
        self.max_joint_change = max(self.max_joint_change, float(np.max(np.abs(joints - self.initial_joints))))
        if time.monotonic() >= self.next_poll and self.error is None:
            try:
                self.observe(_health(self.url))
            except Exception as exc:  # Runtime isolates callback exceptions, so surface them after the run.
                self.error = exc
            self.next_poll = time.monotonic() + 1

    def observe(self, snapshot: dict) -> None:
        """Capture the outcome before the next episode overwrites `last_episode`."""
        count = snapshot["episode"]["episode_count"]
        if count < self.episode_count or count > self.episode_count + 1:
            raise AssertionError("Simulator reset or completed episodes between health polls")
        if count > self.episode_count:
            self.episodes.append(snapshot["episode"].get("last_episode") or {})
        self.episode_count = count

    def check_actions(self) -> None:
        """Reject a rollout that never sent actions or moved the robot."""
        if self.error is not None:
            raise self.error
        assert self.actions_sent > 0, "Runtime did not send any policy actions"
        assert self.max_joint_change > 0.5, "Policy actions did not move the robot"


@contextmanager
def _record_video(
    output_dir: Path, url: str, cameras: list[str], camera_map: dict[str, str], fps: int, policy: str, scene: str
) -> Iterator[Path]:
    """Stream the simulator cameras to an MP4 without buffering frames in memory."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("SIM_E2E_RECORD_VIDEO requires ffmpeg on PATH")
    path = output_dir / f"inference-{policy}-{scene}.mp4"
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning", "-y"]
    for camera in cameras:
        command += [
            "-thread_queue_size",
            "512",
            "-use_wallclock_as_timestamps",
            "1",
            "-i",
            f"{url.rstrip('/')}/cameras/{camera_map.get(camera, camera)}/mjpeg",
        ]
    if len(cameras) == 1:
        command += ["-vf", f"fps={fps},setsar=1"]
    else:
        streams = ";".join(f"[{i}:v]fps={fps},setsar=1[v{i}]" for i in range(len(cameras)))
        stack = "".join(f"[v{i}]" for i in range(len(cameras)))
        command += ["-filter_complex", f"{streams};{stack}hstack=inputs={len(cameras)}[v]", "-map", "[v]"]
    command += [
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
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
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


@pytest.mark.parametrize("scene", SCENES)
def test_train_export_and_run_in_simulation(scene: str, tmp_path: Path) -> None:
    """Train on genuine scene data, then send exported policy actions to MuJoCo."""
    policy_name = os.environ.get("SIM_E2E_POLICY", "act")
    if policy_name not in FIRST_PARTY_POLICIES:
        raise ValueError(f"Unknown first-party policy: {policy_name}")
    dataset_dir = Path(os.environ["SIM_E2E_DATASET_DIR"])
    url = os.environ["SIM_E2E_HTTP_URL"]
    owner = os.environ["SIM_E2E_OWNER"]
    if not (dataset_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing recorded LeRobot dataset: {dataset_dir}")
    provenance = json.loads((dataset_dir / "simulation.json").read_text())
    assert provenance["scene"] == scene, "Dataset and simulator scenes differ"
    metadata = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert metadata["total_episodes"] > 0
    assert metadata["features"]["action"]["shape"] == [6]
    assert metadata["features"]["observation.state"]["shape"] == [6]
    cameras = sorted(
        key.removeprefix("observation.images.") for key in metadata["features"] if key.startswith("observation.images.")
    )
    assert cameras, "Recorded observations must include camera images"
    camera_map = json.loads(os.environ.get("SIM_E2E_CAMERA_MAP", "{}"))
    health = _health(url)
    assert health["scene"] == scene
    assert health["autopilot"]["mode"] == "off", "Disable simulator autopilot before policy evaluation"
    if scene == "conveyor_sort":
        assert health["episode"]["belt_speed"] == pytest.approx(provenance["belt_speed"])

    output_dir = Path(os.environ.get("SIM_E2E_OUTPUT_DIR", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    data = LeRobotDataModule(repo_id="sim-e2e", root=dataset_dir, train_batch_size=2, num_workers=0)
    # Full fine-tuning of these models exceeds one 40 GB A100 at batch size 2.
    policy_options = {
        "molmoact2": {"lora_enabled": True, "gradient_checkpointing": True},
        "xr0": {"freeze_vision_encoder": True, "scheduler_decay_steps": None},
    }.get(policy_name, {})
    policy = get_policy(policy_name, source="physicalai", **policy_options)
    trainer = Trainer(
        default_root_dir=output_dir,
        max_epochs=3,
        max_steps=int(os.environ.get("SIM_E2E_TRAIN_STEPS", "-1")),
        accelerator=os.environ.get("SIM_E2E_ACCELERATOR", "auto"),
        devices=1,
        logger=False,
        callbacks=[ModelCheckpoint(dirpath=output_dir / "checkpoints", save_top_k=0, save_last=True)],
        enable_checkpointing=True,
        enable_progress_bar=False,
    )
    training_start = time.monotonic()
    trainer.fit(policy, datamodule=data)
    training_seconds = time.monotonic() - training_start
    train_loss = next(
        (trainer.callback_metrics[key] for key in ("train/loss", "train/loss_step", "train/loss_epoch")
         if key in trainer.callback_metrics),
        None,
    )
    assert trainer.global_step > 0
    assert (output_dir / "checkpoints" / "last.ckpt").is_file()
    export_dir = output_dir / "export"
    policy.eval().export(export_dir, backend="openvino")
    assert (export_dir / "manifest.json").is_file()

    record_video = os.environ.get("SIM_E2E_RECORD_VIDEO", "false").lower() in {"1", "true"}
    recorder = (
        _record_video(output_dir, url, cameras, camera_map, int(metadata["fps"]), policy_name, scene)
        if record_video
        else nullcontext(None)
    )
    rollout_count = int(os.environ.get("SIM_E2E_ROLLOUTS", "1"))
    if rollout_count < 1:
        raise ValueError("SIM_E2E_ROLLOUTS must be positive")
    rollouts = []
    with recorder as video_path:
        for _ in range(rollout_count):
            # Reset between trials so idle training and previous rollouts cannot affect the score.
            _reset(url)
            time.sleep(0.5)  # Reset is queued on the simulation control thread.
            monitor = _RolloutMonitor(url, _health(url))
            model = InferenceModel(export_dir, backend="openvino", device="CPU")
            runtime = RobotRuntime(
                robot=SharedRobot.attach(name=owner),
                action_source=PolicySource(model=model, execution=SyncExecution(request_threshold=0.5)),
                cameras={
                    key: IPCamera(url=f"{url.rstrip('/')}/cameras/{camera_map.get(key, key)}/mjpeg") for key in cameras
                },
                fps=float(metadata["fps"]),
                callbacks=[monitor],
            )
            stop = threading.Event()
            timer = threading.Timer(float(os.environ.get("SIM_E2E_ROLLOUT_SECONDS", "120")), stop.set)
            try:
                with runtime:
                    timer.start()
                    try:
                        steps = runtime.run(stop_event=stop)
                    finally:
                        timer.cancel()
                monitor.observe(_health(url))
                assert steps > 0
                monitor.check_actions()
                rollouts.append({
                    "score": _score(scene, monitor.episodes),
                    "steps": steps,
                    "episodes": monitor.episodes,
                    "actions_sent": monitor.actions_sent,
                    "max_joint_change": monitor.max_joint_change,
                })
            finally:
                timer.cancel()
                runtime.disconnect()
    score = sum(rollout["score"] for rollout in rollouts) / len(rollouts)
    result = {
        "policy": policy_name,
        "scene": scene,
        "score": score,
        "rollouts": rollouts,
        "training_steps": trainer.global_step,
        "training_seconds": training_seconds,
        "train_loss": float(train_loss) if train_loss is not None else None,
    }
    if video_path is not None:
        result["video"] = video_path.name
    (output_dir / "result.json").write_text(json.dumps(result))
    assert score >= float(os.environ.get("SIM_E2E_MIN_SUCCESS", "0"))


@pytest.mark.parametrize("train_steps", [None, 2])
@pytest.mark.parametrize("record_video", [False, True])
@pytest.mark.parametrize("policy_name", FIRST_PARTY_POLICIES)
@pytest.mark.parametrize("scene", SCENES)
def test_library_orchestration_with_mock_owner(
    scene: str, policy_name: str, record_video: bool, train_steps: int | None, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check train/export/Runtime wiring without claiming a simulated quality result."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    module = sys.modules[__name__]
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "simulation.json").write_text(json.dumps({"scene": scene, "belt_speed": 0.04}))
    features = {
        "action": {"shape": [6]},
        "observation.state": {"shape": [6]},
        "observation.images.overview": {"shape": [64, 64, 3]},
    }
    if scene == "conveyor_sort":
        features["observation.images.wrist"] = {"shape": [64, 64, 3]}
    (dataset / "meta/info.json").write_text(json.dumps({"total_episodes": 2, "fps": 30, "features": features}))
    policy = MagicMock()
    policy.eval.return_value = policy

    def export(path: Path, *, backend: str) -> None:
        assert backend == "openvino"
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("{}")

    policy.export.side_effect = export
    loss_key = "train/loss_step" if policy_name == "rldx1" else "train/loss"
    trainer = MagicMock(global_step=2, callback_metrics={loss_key: 0.25})

    def save_checkpoint(*args: object, **kwargs: object) -> None:
        checkpoint = tmp_path / "results/checkpoints/last.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()

    trainer.fit.side_effect = save_checkpoint
    runtime = MagicMock()
    video_process = MagicMock()
    video_process.poll.return_value = None

    def finish_video(*, timeout: int) -> int:
        video = tmp_path / f"results/inference-{policy_name}-{scene}.mp4"
        video.write_bytes(b"mock video")
        return 0

    video_process.wait.side_effect = finish_video
    popen = MagicMock(return_value=video_process)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    get_policy = MagicMock(return_value=policy)
    monkeypatch.setattr(module, "get_policy", get_policy)
    make_trainer = MagicMock(return_value=trainer)
    monkeypatch.setattr(module, "Trainer", make_trainer)
    make_data = MagicMock(return_value=object())
    monkeypatch.setattr(module, "LeRobotDataModule", make_data)
    make_model = MagicMock(side_effect=lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "InferenceModel", make_model)
    monkeypatch.setattr(module, "SharedRobot", SimpleNamespace(attach=lambda **kwargs: object()))
    monkeypatch.setattr(module, "IPCamera", lambda **kwargs: object())
    monkeypatch.setattr(module, "PolicySource", lambda **kwargs: object())
    monkeypatch.setattr(module, "SyncExecution", lambda **kwargs: object())
    make_runtime = MagicMock(return_value=runtime)
    monkeypatch.setattr(module, "RobotRuntime", make_runtime)

    def run(*, stop_event: threading.Event) -> int:
        assert stop_event.wait(0.5), "Rollout did not stop at the wall-clock deadline"
        monitor = make_runtime.call_args.kwargs["callbacks"][0]
        for position in (0.0, 2.0):
            monitor.on_tick(
                SimpleNamespace(
                    robot_state=SimpleNamespace(joint_positions=np.full(6, position)), action_sent=np.zeros(6)
                )
            )
        return 30

    runtime.run.side_effect = run
    health = [
        {"scene": scene, "autopilot": {"mode": "off"}, "episode": {"belt_speed": 0.04}},
        {"episode": {"episode_count": 0}},
        {"episode": {"episode_count": 1, "last_episode": {"correct": 7, "wrong": 2, "missed": 1}}},
    ]
    if scene == "conveyor_sort":
        health += [
            {"episode": {"episode_count": 0}},
            {"episode": {"episode_count": 1, "last_episode": {"correct": 3, "wrong": 2, "missed": 5}}},
        ]
    monkeypatch.setattr(module, "_health", lambda url: health.pop(0))
    reset = MagicMock()
    monkeypatch.setattr(module, "_reset", reset)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setenv("SIM_E2E_POLICY", policy_name)
    monkeypatch.setenv("SIM_E2E_DATASET_DIR", str(dataset))
    monkeypatch.setenv("SIM_E2E_HTTP_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("SIM_E2E_OWNER", "mock-owner")
    monkeypatch.setenv("SIM_E2E_OUTPUT_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("SIM_E2E_RECORD_VIDEO", "true" if record_video else "false")
    monkeypatch.setenv("SIM_E2E_ROLLOUT_SECONDS", "0.01")
    monkeypatch.setenv("SIM_E2E_ROLLOUTS", "2" if scene == "conveyor_sort" else "1")
    if train_steps is None:
        monkeypatch.delenv("SIM_E2E_TRAIN_STEPS", raising=False)
    else:
        monkeypatch.setenv("SIM_E2E_TRAIN_STEPS", str(train_steps))
    test_train_export_and_run_in_simulation(scene, tmp_path)

    trainer.fit.assert_called_once()
    assert make_data.call_args.kwargs["train_batch_size"] == 2
    assert make_trainer.call_args.kwargs["max_epochs"] == 3
    assert make_trainer.call_args.kwargs["max_steps"] == (-1 if train_steps is None else train_steps)
    assert make_trainer.call_args.kwargs["callbacks"][0].save_last
    assert reset.call_count == (2 if scene == "conveyor_sort" else 1)
    assert runtime.run.call_count == reset.call_count
    assert runtime.disconnect.call_count == reset.call_count
    assert make_model.call_count == reset.call_count
    if policy_name == "molmoact2":
        get_policy.assert_called_once_with(
            policy_name, source="physicalai", lora_enabled=True, gradient_checkpointing=True
        )
    elif policy_name == "xr0":
        get_policy.assert_called_once_with(
            policy_name, source="physicalai", freeze_vision_encoder=True, scheduler_decay_steps=None
        )
    else:
        get_policy.assert_called_once_with(policy_name, source="physicalai")
    expected = 0.5 if scene == "conveyor_sort" else 1.0
    result = json.loads((tmp_path / "results/result.json").read_text())
    assert result["score"] == pytest.approx(expected)
    assert [r["score"] for r in result["rollouts"]] == ([0.7, 0.3] if scene == "conveyor_sort" else [1.0])
    assert all(len(r["episodes"]) == 1 for r in result["rollouts"])
    assert all(r["actions_sent"] == 2 for r in result["rollouts"])
    assert all(r["max_joint_change"] == pytest.approx(2.0) for r in result["rollouts"])
    assert result["training_steps"] == 2
    assert result["training_seconds"] >= 0
    assert result["train_loss"] == pytest.approx(0.25)
    if record_video:
        assert result["video"] == f"inference-{policy_name}-{scene}.mp4"
        command = popen.call_args.args[0]
        assert "cameras/overview/mjpeg" in " ".join(command)
        if scene == "conveyor_sort":
            assert "cameras/wrist/mjpeg" in " ".join(command)
            assert "hstack=inputs=2" in command[command.index("-filter_complex") + 1]
        else:
            assert command[command.index("-vf") + 1] == "fps=30,setsar=1"
        video_process.send_signal.assert_called_once_with(signal.SIGINT)
    else:
        assert "video" not in result
        popen.assert_not_called()


def test_video_recording_requires_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit video requests fail clearly if the encoder is unavailable."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="requires ffmpeg"):
        with _record_video(tmp_path, "http://127.0.0.1:8080", ["overview"], {}, 30, "act", "conveyor_sort"):
            pass


def test_simulation_scoring_contract() -> None:
    """Every completed episode contributes to the score, including failures."""
    assert _score("conveyor_sort", [{"correct": 7, "wrong": 2, "missed": 1}]) == pytest.approx(0.7)
    assert _score(
        "conveyor_sort", [{"correct": 7, "wrong": 2, "missed": 1}, {"correct": 0, "wrong": 0, "missed": 10}]
    ) == pytest.approx(0.35)
    assert _score("single_pick_place", []) == 0
    assert _score("single_pick_place", [{}]) == 1
    with pytest.raises(AssertionError, match="did not finish"):
        _score("conveyor_sort", [])


def test_rollout_monitor_rejects_no_motion() -> None:
    """Sent commands alone do not prove that policy actions drove the arm."""
    from types import SimpleNamespace

    monitor = _RolloutMonitor("http://127.0.0.1:8080", {"episode": {"episode_count": 0}})
    monitor.on_tick(SimpleNamespace(robot_state=SimpleNamespace(joint_positions=np.zeros(6)), action_sent=np.zeros(6)))
    monitor.on_tick(SimpleNamespace(robot_state=SimpleNamespace(joint_positions=np.zeros(6)), action_sent=np.zeros(6)))
    with pytest.raises(AssertionError, match="did not move"):
        monitor.check_actions()
    with pytest.raises(AssertionError, match="completed episodes between health polls"):
        monitor.observe({"episode": {"episode_count": 2}})
