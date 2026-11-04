# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Controls and provenance for the opt-in simulation E2E suite."""

from __future__ import annotations

import json
import threading
import time
from importlib.metadata import distribution
from typing import TYPE_CHECKING
from urllib.request import Request, urlopen

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

DATASET = {
    "repo_id": "physicalai-studio/conveyor-sort-sim-demos-100-episodes",
    "revision": "17db8f758b32141d795a71a064cfa9f5c6341b5b",
    "episodes": 100,
    "belt_speeds": [0.04],
}
SEEDS = (104729, 104743, 104759, 104761, 104773)


def health(url: str) -> dict:
    with urlopen(f"{url.rstrip('/')}/health", timeout=5) as response:
        return json.load(response)


def command(url: str, route: str, payload: dict) -> None:
    request = Request(
        f"{url.rstrip('/')}/{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5):
        pass


def wait_health(url: str, predicate, timeout: float = 15) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = health(url)
        if predicate(snapshot):
            return snapshot
        time.sleep(0.1)
    raise TimeoutError("Simulator did not acknowledge the requested state")


def prepare_episode(url: str, seed: int, mode: str, belt_speed: float) -> dict:
    """Pause setup; use the autopilot mode transition as a FIFO command barrier.

    The conveyor must be active while scoring: disabling auto-reset pauses its
    belt, rather than preventing the next episode from starting.
    """
    command(url, "episode/auto-reset", {"enabled": False})
    wait_health(url, lambda h: not h["episode"]["active"])
    command(url, "autopilot", {"mode": "leader"})
    wait_health(url, lambda h: h["autopilot"]["mode"] == "leader")
    command(url, "seed", {"seed": seed})
    wait_health(url, lambda h: h["seed"] == seed)
    command(url, "conveyor/belt-speed", {"speed": belt_speed})
    command(url, "reset", {})
    command(url, "autopilot", {"mode": mode})
    return wait_health(
        url,
        lambda h: (
            h["autopilot"]["mode"] == mode and h["episode"]["spawned"] == 0 and h["episode"]["belt_speed"] == belt_speed
        ),
    )


def score(episodes: list[dict]) -> float:
    if not episodes:
        raise AssertionError("No complete scored episode")
    total = sum(e[key] for e in episodes for key in ("correct", "wrong", "missed"))
    if total <= 0:
        raise AssertionError("Episode has no scored items")
    return sum(e["correct"] for e in episodes) / total


def timing_passes(trials: list[dict], fps: float, cameras: list[str], camera_map: dict[str, str] | None = None) -> bool:
    """Require control throughput and valid simulator timing throughout every scored trial."""
    simulator_cameras = [(camera_map or {}).get(camera, camera) for camera in cameras]
    return (
        bool(trials)
        and bool(cameras)
        and all(
            trial["control_hz"] is not None
            and trial["control_hz"] >= fps * 0.8
            and all(
                trial["camera_hz"].get(camera, 0) >= fps * 0.8
                and trial["camera_max_gap_s"].get(camera, float("inf")) <= 0.25
                and camera in trial["changed_cameras"]
                for camera in cameras
            )
            and bool(trial["simulator_timing_samples"])
            and all(
                0.9 <= (timing.get("real_time_factor") or 0) <= 1.1
                and all(
                    ((timing.get("camera_fps") or {}).get(camera) or 0) >= fps * 0.8 for camera in simulator_cameras
                )
                for timing in trial["simulator_timing_samples"]
            )
            for trial in trials
        )
    )


class EpisodeMonitor:
    """Poll simulator HTTP outside Runtime's synchronous tick callbacks."""

    def __init__(self, url: str, before: dict, stop: threading.Event, timeout: float) -> None:
        self.url, self.stop, self.timeout = url, stop, timeout
        self.count = before["episode"]["episode_count"]
        self.items = before["episode"]["items_per_episode"]
        self.episodes: list[dict] = []
        self.error: Exception | None = None
        self.recording = True
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.actions_sent = 0
        self.max_joint_change = 0.0
        self.initial_joints: np.ndarray | None = None
        self.ticks: list[float] = []
        self.latencies: list[float] = []
        self.frame_times: dict[str, set[float]] = {}
        self.frame_updates: dict[str, float] = {}
        self.frame_max_gaps: dict[str, float] = {}
        self.initial_images: dict[str, np.ndarray] = {}
        self.changed_cameras: set[str] = set()
        self.snapshot = before
        self.timing_samples: list[dict] = []
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def observe(self, snapshot: dict) -> None:
        count = snapshot["episode"]["episode_count"]
        if snapshot.get("seed") != self.snapshot.get("seed"):
            raise AssertionError("Simulator seed changed during a trial")
        if snapshot.get("autopilot", {}).get("mode") != self.snapshot.get("autopilot", {}).get("mode"):
            raise AssertionError("Simulator autopilot changed during a trial")
        if count < self.count or count > self.count + 1:
            raise AssertionError("Lost simulator episodes or unexpected reset")
        self.snapshot = snapshot
        self.timing_samples.append(snapshot.get("timing", {}))
        if count > self.count:
            episode = snapshot["episode"]["last_episode"]
            if sum(episode[k] for k in ("correct", "wrong", "missed")) != self.items:
                raise AssertionError("Incomplete episode item accounting")
            self.completed_at = time.monotonic()
            self.episodes.append(episode)
            self.stop.set()
        self.count = count

    def _poll(self) -> None:
        deadline = time.monotonic() + self.timeout
        try:
            while not self.finished.wait(0.1):
                self.observe(health(self.url))
                if self.episodes:
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("Episode did not finish before the wall-clock deadline")
        except Exception as exc:
            self.error = exc
            self.stop.set()

    def begin(self, before: dict) -> None:
        if self.error is not None:
            raise self.error
        self.count = before["episode"]["episode_count"]
        self.snapshot = before
        self.timing_samples.clear()
        self.recording = True
        self.started_at = time.monotonic()
        self.thread.start()

    def on_tick(self, event) -> None:
        if event.action_sent is not None:
            action = np.asarray(event.action_sent)
            if action.shape != (6,) or action.dtype.kind not in "fiu" or not np.isfinite(action).all():
                # Runtime logs callback exceptions; retain the failure for close()/begin().
                self.error = AssertionError("Policy actions must contain six finite joint values")
                self.stop.set()
                return
        if not self.recording:
            self.stop.set()  # A paused one-tick warmup must not enter scored metrics.
            return
        if self.completed_at is not None:
            return
        self.ticks.append(time.monotonic())
        self.actions_sent += int(event.action_sent is not None)
        joints = np.asarray(event.robot_state.joint_positions)
        if self.initial_joints is None:
            self.initial_joints = joints.copy()
        self.max_joint_change = max(self.max_joint_change, float(np.max(np.abs(joints - self.initial_joints))))
        for key, frame in event.camera_frames.items():
            # Conveyor views must change during an episode; static streaks are not dropped frames.
            if key not in self.changed_cameras:
                if key not in self.initial_images:
                    self.initial_images[key] = frame.data.copy()
                elif not np.array_equal(self.initial_images[key], frame.data):
                    self.changed_cameras.add(key)
                    del self.initial_images[key]
            timestamps = self.frame_times.setdefault(key, set())
            if frame.timestamp not in timestamps:
                previous = self.frame_updates.get(
                    key, self.started_at if self.started_at is not None else self.ticks[0]
                )
                self.frame_max_gaps[key] = max(self.frame_max_gaps.get(key, 0), self.ticks[-1] - previous)
                self.frame_updates[key] = self.ticks[-1]
                timestamps.add(frame.timestamp)

    def on_inference(self, event) -> None:
        if self.recording:
            self.latencies.append(event.latency_s)

    def close(self) -> None:
        self.finished.set()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise RuntimeError("Simulator monitor did not stop")
        if self.error is not None:
            raise self.error
        if len(self.episodes) != 1:
            raise AssertionError("Trial must score exactly one completed episode")

    def result(self, seed: int) -> dict:
        assert self.completed_at is not None, "Timing requires a completed episode"
        start = self.started_at if self.started_at is not None else (self.ticks[0] if self.ticks else self.completed_at)
        elapsed = self.completed_at - start
        return {
            "seed": seed,
            "score": score(self.episodes),
            "episodes": self.episodes,
            "actions_sent": self.actions_sent,
            "max_joint_change": self.max_joint_change,
            "steps": len(self.ticks),
            "control_hz": (len(self.ticks) if self.started_at is not None else len(self.ticks) - 1) / elapsed
            if elapsed
            else None,
            "first_action_delay_s": self.ticks[0] - self.started_at
            if self.ticks and self.started_at is not None
            else None,
            "inference_p95_s": float(np.percentile(self.latencies, 95)) if self.latencies else None,
            "unique_camera_frames": {key: len(times) for key, times in self.frame_times.items()},
            "changed_cameras": sorted(self.changed_cameras),
            "camera_hz": {key: len(times) / elapsed if elapsed else 0 for key, times in self.frame_times.items()},
            "camera_max_gap_s": {
                key: max(self.frame_max_gaps[key], self.completed_at - updated)
                for key, updated in self.frame_updates.items()
            },
            "simulator_timing": self.snapshot.get("timing", {}),
            "simulator_timing_samples": self.timing_samples,
        }


def runtime_versions() -> dict[str, str]:
    """Report installed dependency versions without enforcing E2E-specific pins."""
    return {name: distribution(name).version for name in ("physicalai", "physicalai-mujoco-plugin")}


def control_scores(path: Path) -> dict[str, float]:
    """Require successful expert/no-action controls on the rollout's exact seed protocol."""
    report = json.loads(path.read_text())
    if report.get("status") != "passed":
        raise ValueError("Simulator controls must pass before training and inference")
    scores = {}
    for mode in ("off", "drive"):
        trials = report[mode]["trials"]
        if [trial["seed"] for trial in trials] != list(SEEDS):
            raise ValueError("Simulator controls must use the rollout seeds")
        scores[mode] = score([episode for trial in trials for episode in trial["episodes"]])
    if not 0 <= scores["off"] < scores["drive"] <= 1:
        raise ValueError("Scripted expert must beat a valid no-action score")
    return scores
