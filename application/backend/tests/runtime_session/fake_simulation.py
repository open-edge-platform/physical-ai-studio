"""A simulation that speaks the autopilot protocol in memory, for the autopilot recording tests.

Every request and answer goes through the plugin SDK's models. An episode is ready ``ready_after_s``
after it was asked for, and ends ``ended_after_s`` after it began, as ``outcomes`` says in order (the
last one repeats). A test can make the scene unsupported, or the simulation stop answering (``down``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
from physicalai_studio_plugin.autopilot import (
    AUTOPILOT_PATH,
    AutopilotError,
    AutopilotInfo,
    Episode,
    EpisodeRequest,
    Outcome,
)
from pydantic import ValidationError

TASK = "Put the red cube in the blue bucket."
REASONS: dict[str, str] = {
    "failure": "The cube fell off the table.",
    "stalled": "The episode ran past 120 s of simulated time.",
}


@dataclass
class _Episode:
    id: str
    phase: str = "preparing"
    task: str | None = None
    outcome: str | None = None
    reason: str | None = None
    clean: bool | None = None
    since: float = field(default_factory=time.monotonic)

    def model(self) -> Episode:
        return Episode.model_validate(
            {
                "id": self.id,
                "phase": self.phase,
                "task": self.task,
                "outcome": self.outcome,
                "reason": self.reason,
                "clean": self.clean,
            }
        )

    def end(self, outcome: str, reason: str | None = None, clean: bool | None = None) -> None:
        if self.phase != "ended":
            self.phase, self.outcome, self.reason, self.clean = "ended", outcome, reason, clean


@dataclass
class FakeSimulation:
    joint_names: list[str]
    scene: str = "bucket_sort"
    supported: bool = True
    options: list[str] = field(default_factory=lambda: ["max_attempts", "recovery_fraction"])
    outcomes: list[tuple[Outcome, bool | None]] = field(default_factory=lambda: [("success", True)])
    ready_after_s: float = 0.05
    ended_after_s: float = 0.8
    down: bool = False
    calls: list[str] = field(default_factory=list)
    """``info``, ``start``, ``episode``, ``begin`` and ``abort``, in the order they came."""
    episodes: list[_Episode] = field(default_factory=list)
    """Every episode asked for, the current or last one last."""
    _ended: int = 0

    @property
    def current(self) -> _Episode | None:
        return self.episodes[-1] if self.episodes else None

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("the simulation is down", request=request)
        self._advance()
        parts = request.url.path.removeprefix(AUTOPILOT_PATH).strip("/").split("/")
        match request.method, parts:
            case "GET", [""]:
                self.calls.append("info")
                return _answer(200, self._info())
            case "POST", ["episodes"]:
                self.calls.append("start")
                return self._start(json.loads(request.content or b"{}"))
            case "GET", ["episodes", episode_id]:
                self.calls.append("episode")
                return self._episode(episode_id, lambda episode: None)
            case "POST", ["episodes", episode_id, "begin"]:
                self.calls.append("begin")
                return self._episode(episode_id, _begin)
            case "POST", ["episodes", episode_id, "abort"]:
                self.calls.append("abort")
                return self._episode(episode_id, lambda episode: episode.end("aborted", "Aborted by Studio."))
        return httpx.Response(404, json={"detail": "Not Found"})

    def _info(self) -> AutopilotInfo:
        return AutopilotInfo(
            protocol=1,
            scene=self.scene,
            supported=self.supported,
            reason=None if self.supported else f"Scene {self.scene!r} has no autopilot.",
            options=self.options if self.supported else [],
            joint_names=self.joint_names,
            episode=self.current.model() if self.current else None,
        )

    def _start(self, body: Any) -> httpx.Response:
        try:
            request = EpisodeRequest.model_validate(body)
        except ValidationError as error:
            return _error(422, "bad_option", str(error))
        if not self.supported:
            return _error(409, "unsupported", f"Scene {self.scene!r} has no autopilot.")
        if unknown := request.options.set_fields() - set(self.options):
            return _error(422, "bad_option", f"Scene {self.scene!r} does not take {sorted(unknown)}.")
        if (previous := self.current) is not None:
            previous.end("aborted", "Replaced by a new episode.")
        episode = _Episode(f"e-{len(self.episodes) + 1:04d}")
        self.episodes.append(episode)
        return _answer(201, episode.model())

    def _episode(self, episode_id: str, change: Any) -> httpx.Response:
        episode = self.current
        if episode is None or episode.id != episode_id:
            return _error(404, "not_found", f"No episode {episode_id!r}.")
        if (refused := change(episode)) is not None:
            return refused
        return _answer(200, episode.model())

    def _advance(self) -> None:
        episode = self.current
        if episode is None:
            return
        now = time.monotonic()
        if episode.phase == "preparing" and now - episode.since >= self.ready_after_s:
            episode.phase, episode.task, episode.since = "ready", TASK, now
        if episode.phase == "running" and now - episode.since >= self.ended_after_s:
            outcome, clean = self.outcomes[min(self._ended, len(self.outcomes) - 1)]
            self._ended += 1
            reason = REASONS.get(outcome) or (None if clean is not False else "Success after a regrasp.")
            episode.end(outcome, reason, clean if outcome == "success" else None)


def _begin(episode: _Episode) -> httpx.Response | None:
    if episode.phase == "ready":
        episode.phase, episode.since = "running", time.monotonic()
    elif episode.phase != "running":
        return _error(409, "wrong_phase", f"Episode {episode.id!r} is not ready.")
    return None


def _answer(status: int, model: Any) -> httpx.Response:
    return httpx.Response(status, json=model.model_dump(mode="json"))


def _error(status: int, code: str, message: str) -> httpx.Response:
    detail = AutopilotError.model_validate({"code": code, "message": message})
    return httpx.Response(status, json={"detail": detail.model_dump(mode="json")})


class FakeLeader:
    """The simulation's leader: a new target every read, unless ``frozen``.

    Like the MuJoCo plugin's virtual leader, it says which episode its last sample is of (``episode_id``):
    the simulation's current one, or ``publish_episode`` when a test sets it.
    """

    def __init__(self, sim: FakeSimulation) -> None:
        self.sim = sim
        self.joint_names = list(sim.joint_names)
        self.frozen = False
        self.connected = False
        self.publish_episode: str | None = None
        self.episode_id: str | None = None
        self._last: tuple[np.ndarray, float] | None = None
        self._t0 = time.monotonic()

    @property
    def device_ids(self) -> tuple[str, ...]:
        return ()

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def get_observation(self) -> Any:
        if self._last is None or not self.frozen:
            now = time.monotonic()
            # Targets that move, so a target and a hold can be told apart.
            self._last = (np.array([1.0, 2.0]) + 0.5 * (now - self._t0), now)
            current = self.sim.current
            self.episode_id = self.publish_episode or (current.id if current is not None else None)
        positions, timestamp = self._last
        return type("Observation", (), {"joint_positions": positions, "timestamp": timestamp, "sensor_data": None})()

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        _ = action, goal_time
