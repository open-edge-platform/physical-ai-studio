# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Controls and provenance for the opt-in simulation E2E suite."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from importlib.metadata import distribution
from typing import TYPE_CHECKING
from urllib.request import Request, urlopen

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

POLICIES = ("act", "molmoact2", "pi05", "rldx1", "smolvla", "xr0")
RUNTIME_REVISION = "1cc910b451508ce40cd7319ece2a29283fadfcfb"
DATASETS = {
    "50": {
        "repo_id": "Albertvh97/conveyor-sort-sim-demos",
        "revision": "6a949360761fee17089fbaaf8bc53f59715cb575",
        "episodes": 50,
    },
    "100": {
        "repo_id": "Albertvh97/conveyor-sort-sim-demos-100-episodes",
        "revision": "17db8f758b32141d795a71a064cfa9f5c6341b5b",
        "episodes": 100,
    },
}
SEEDS = (104729, 104743, 104759, 104761, 104773)


def dataset_entry() -> dict:
    """Select a repository-reviewed recording, defaulting to the original 50 episodes."""
    choice = os.environ.get("SIM_E2E_DATASET", "50")
    if choice not in DATASETS:
        raise ValueError(f"Unknown simulation dataset: {choice!r}; choose 50 or 100")
    return DATASETS[choice]


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
        self.actions_sent = 0
        self.max_joint_change = 0.0
        self.initial_joints = None
        self.ticks: list[float] = []
        self.latencies: list[float] = []
        self.frame_times: dict[str, set[float]] = {}
        self.snapshot = before
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
        if count > self.count:
            episode = snapshot["episode"]["last_episode"]
            if sum(episode[k] for k in ("correct", "wrong", "missed")) != self.items:
                raise AssertionError("Incomplete episode item accounting")
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
        self.count = before["episode"]["episode_count"]
        self.snapshot = before
        self.recording = True
        self.started_at = time.monotonic()
        self.thread.start()

    def on_tick(self, event) -> None:
        if not self.recording:
            self.stop.set()  # A paused one-tick warmup must not enter scored metrics.
            return
        self.ticks.append(time.monotonic())
        self.actions_sent += int(event.action_sent is not None)
        joints = np.asarray(event.robot_state.joint_positions)
        if self.initial_joints is None:
            self.initial_joints = joints.copy()
        self.max_joint_change = max(self.max_joint_change, float(np.max(np.abs(joints - self.initial_joints))))
        for key, frame in event.camera_frames.items():
            self.frame_times.setdefault(key, set()).add(frame.timestamp)

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
        elapsed = (
            self.ticks[-1] - (self.started_at if self.started_at is not None else self.ticks[0]) if self.ticks else 0
        )
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
            "simulator_timing": self.snapshot.get("timing", {}),
        }


def attest_runtime() -> dict:
    """Fail before expensive work if installed VCS dependencies differ from the pins."""
    commits = {}
    for name in ("physicalai", "physicalai-mujoco-plugin"):
        direct_url = json.loads(distribution(name).read_text("direct_url.json") or "{}")
        commit = direct_url.get("vcs_info", {}).get("commit_id")
        if commit != RUNTIME_REVISION:
            raise RuntimeError(f"Installed {name} does not match the pinned Runtime revision")
        commits[name] = commit
    return commits


def checkpoint_entry(path: Path, policy: str) -> dict | None:
    entry = json.loads(path.read_text())[policy]
    if entry is None:
        return None
    if not isinstance(entry, dict) or set(entry) != {"repo_id", "revision", "filename", "sha256", "min_score"}:
        raise ValueError("Invalid qualified checkpoint entry")
    if not isinstance(entry["repo_id"], str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", entry["repo_id"]):
        raise ValueError("Invalid checkpoint repository")
    if not isinstance(entry["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", entry["revision"]):
        raise ValueError("Checkpoint revision must be immutable")
    filename = entry["filename"]
    if (
        not isinstance(filename, str)
        or "/" in filename
        or "\\" in filename
        or ".." in filename
        or not filename.endswith(".ckpt")
    ):
        raise ValueError("Invalid checkpoint filename")
    if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise ValueError("Invalid checkpoint checksum")
    if type(entry["min_score"]) not in (int, float) or not 0 < entry["min_score"] <= 1:
        raise ValueError("A qualified checkpoint needs a positive measured score threshold")
    return entry


def verify_checkpoint(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError("Checkpoint checksum mismatch")
