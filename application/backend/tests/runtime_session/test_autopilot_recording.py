"""Autopilot recording in a session, against a simulation that speaks the autopilot protocol in memory."""

from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
import pytest
from physicalai_studio_plugin.autopilot import AutopilotSpec

from api.runtime_ws import _request
from exceptions import DeviceInUseError, RuntimeSessionCommandError
from runtime_session import autopilot_recording
from runtime_session.bus import command_key
from runtime_session.session import Resolvers, RuntimeSession, SimulationEndpoint
from runtime_session.store import ACTION
from schemas.dataset import Dataset
from schemas.environment import EnvironmentWithRelations
from schemas.runtime_session import AutopilotRecordingOptions, StartAutopilotRecordingCommand, StartEpisodeCommand
from tests.workers.fake_dataset import FakeDatasetWorker, recording_log
from tests.workers.fake_robot import FakeRobot
from workers.dataset_worker import RecordingStatus

from .fake_simulation import TASK, FakeLeader, FakeSimulation
from .fixtures import FOLLOWER_ID, JOINTS, next_status, subscribe_statuses

SIM_URL = "http://127.0.0.1:50488"
RUN_1 = "101@2026-11-17T00:00:00+00:00"
RUN_2 = "202@2026-11-17T00:05:00+00:00"


def _environment(follower_id=FOLLOWER_ID, cameras: int = 1, *, autopilot: bool = True) -> EnvironmentWithRelations:
    return EnvironmentWithRelations.model_validate(
        {
            "id": str(uuid4()),
            "name": "Sim bench",
            "robots": [
                {
                    "robot": {
                        "id": str(follower_id),
                        "name": f"Sim {str(follower_id)[:4]}",
                        "type": "MuJoCo_SO101_Follower",
                        "payload": {"http_url": SIM_URL},
                    },
                    "tele_operator": {"type": "autopilot"} if autopilot else {"type": "none"},
                }
            ],
            "cameras": [
                {
                    "id": str(uuid4()),
                    "driver": "usb_camera",
                    "name": f"Cam {index}",
                    "fingerprint": {"path": f"/dev/video{index}"},
                    "hardware_name": None,
                    "payload": {"width": 160, "height": 120, "fps": 30},
                }
                for index in range(cameras)
            ],
        }
    )


def _dataset() -> Dataset:
    dataset = Dataset(id=uuid4(), name="Auto", default_task="pick", project_id=uuid4(), environment_id=uuid4())
    Path(dataset.path).mkdir(parents=True)
    return dataset


class Rig:
    """One simulation, the leaders built for it, and which run of it Studio's supervisor reports."""

    def __init__(self) -> None:
        self.sim = FakeSimulation(joint_names=JOINTS)
        self.leaders: list[FakeLeader] = []
        self.run: str | None = RUN_1
        self.has_autopilot = True

    def leader_for(self, _payload, http_url: str) -> FakeLeader:
        assert http_url == SIM_URL
        leader = FakeLeader(self.sim)
        self.leaders.append(leader)
        return leader

    def endpoint(self, _robot_id) -> SimulationEndpoint | None:
        return SimulationEndpoint(http_url=SIM_URL, run=self.run) if self.run is not None else None

    @property
    def leader(self) -> FakeLeader:
        return self.leaders[-1]


@pytest.fixture
def rig() -> Rig:
    return Rig()


@pytest.fixture
def sim_factory(mock_robot_client_factory, tmp_path, rig):
    """Builds a FakeRobot follower whose type has a simulation and, unless the rig says not, an autopilot."""

    def definition():
        autopilot = AutopilotSpec(leader_for=rig.leader_for) if rig.has_autopilot else None
        return SimpleNamespace(simulation=SimpleNamespace(owner_name=lambda _payload: "sim-owner", autopilot=autopilot))

    async def build(robot, _port_finder):
        return FakeRobot(str(tmp_path / f"{robot.id}.jsonl"), JOINTS, [1.0, 2.0]), definition()

    mock_robot_client_factory.build_robot_driver = AsyncMock(side_effect=build)
    return mock_robot_client_factory


@pytest.fixture
async def sim_session(registry, camera_claims, sim_factory, rig, fake_cameras):
    with patch("runtime_session.dataset.DatasetWorker", FakeDatasetWorker):
        async with AsyncExitStack() as stack:

            async def create() -> RuntimeSession:
                session = RuntimeSession(
                    project_id=uuid4(),
                    project_name="Alpha",
                    registry=registry,
                    camera_claims=camera_claims,
                    robot_factory=sim_factory,
                    resolve=Resolvers(environment=_never, dataset=_never, model=_never, simulation=rig.endpoint),
                    autopilot_transport=rig.sim.transport(),
                )
                return await stack.enter_async_context(session)

            yield create


async def _never(_what):
    raise AssertionError("not looked up")


async def _ready(sim_session, rig: Rig, **environment) -> tuple[RuntimeSession, Dataset]:
    session = await sim_session()
    await session.load_environment(_environment(**environment))
    dataset = _dataset()
    await session.load_dataset(dataset, fps=30)
    return session, dataset


async def _start(session: RuntimeSession, dataset: Dataset, **options) -> None:
    command = StartAutopilotRecordingCommand(
        request_id="", dataset_id=dataset.id, options=AutopilotRecordingOptions(**options)
    )
    await session.start_autopilot_recording(command)


async def _until(condition, within: float = 15.0) -> None:
    deadline = time.monotonic() + within
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        await asyncio.sleep(0.02)


def _state(session: RuntimeSession):
    state = session.state().autopilot_recording
    assert state is not None
    return state


def _phase(session: RuntimeSession) -> str | None:
    state = session.state().autopilot_recording
    return state.phase if state is not None else None


def _events(dataset: Dataset) -> list[str]:
    return [event["event"] for event in recording_log(Path(dataset.path) / "recording.jsonl")]


async def _manual_start(session: RuntimeSession) -> None:
    command = StartEpisodeCommand(request_id="", task="mine")
    await session.bus.request(command_key("start_episode"), command, RecordingStatus, timeout=10)


# ----------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------


async def test_it_keeps_successes_only_a_regrasp_included(sim_session, rig) -> None:
    rig.sim.outcomes = [("failure", None), ("success", False), ("stalled", None), ("success", True)]
    session, dataset = await _ready(sim_session, rig)

    await _start(session, dataset, max_episodes=2, max_attempts=2)
    assert session.state().control.type == "autopilot"
    await _until(lambda: _phase(session) == "done", within=30)

    state = _state(session)
    assert (state.saved, state.discarded, state.discards) == (2, 2, {"failure": 1, "stalled": 1})
    assert state.scene == "bucket_sort" and state.message == "Saved 2 episode(s)."
    assert state.last.kept and state.last.clean is True and state.last.frames >= autopilot_recording.MIN_FRAMES
    assert state.last.task == TASK
    assert _events(dataset).count("save") == 2 and _events(dataset).count("discard") == 2
    assert all(event["task"] == TASK for event in recording_log(Path(dataset.path) / "recording.jsonl")[:1])
    assert session.state().dataset.episodes_recorded == 2
    await _until(lambda: session.state().control is None)  # the follower holds once it is done
    await _until(lambda: not rig.leader.connected)  # the control's thread disconnects it after the session lets go
    await session.stop()


async def test_a_short_success_is_discarded(sim_session, rig) -> None:
    rig.sim.ended_after_s = 0.0
    session, dataset = await _ready(sim_session, rig)

    await _start(session, dataset)
    await _until(lambda: _state(session).discarded >= 1)

    assert _state(session).last.discard_reason == "too_short" and not _state(session).last.kept
    assert "save" not in _events(dataset)
    await session.stop()


async def test_a_stale_camera_discards_the_episode_and_the_recording_goes_on(sim_session, rig, fake_cameras) -> None:
    rig.sim.ended_after_s = 2.0
    session, dataset = await _ready(sim_session, rig)
    await _start(session, dataset, max_episodes=1)
    await _until(lambda: _phase(session) == "recording")

    frozen = SimpleNamespace(data=np.zeros((120, 160, 3), dtype=np.uint8), timestamp=time.monotonic())
    fresh = fake_cameras.return_value.read_latest.side_effect
    fake_cameras.return_value.read_latest.side_effect = lambda: frozen
    await _until(lambda: _state(session).discarded == 1, within=10)
    fake_cameras.return_value.read_latest.side_effect = fresh

    # The camera worker may report the camera lost first: then the dataset discards the episode itself.
    assert _state(session).last.discard_reason in {"stale_stream", "dataset"}
    await _until(lambda: _phase(session) in {"done", "failed"}, within=30)
    assert _phase(session) == "done" and _state(session).saved == 1
    await session.stop()


@pytest.mark.parametrize("stream", ["leader", "follower"])
async def test_a_stale_leader_or_follower_holds_and_fails_the_recording(sim_session, rig, tmp_path, stream) -> None:
    rig.sim.ended_after_s = 30.0
    session, dataset = await _ready(sim_session, rig)
    await _start(session, dataset)
    await _until(lambda: _phase(session) == "recording")
    control = session._control.value

    if stream == "leader":
        rig.leader.frozen = True
    else:
        (tmp_path / f"{FOLLOWER_ID}.jsonl.freeze").touch()  # its SharedRobot keeps returning its last state
    await _until(lambda: control.halted, within=3)
    held = session.store.read(ACTION)
    await asyncio.sleep(0.3)

    assert np.allclose(session.store.read(ACTION).value, held.value)  # no target after the hold
    await _until(lambda: _phase(session) == "failed", within=5)
    assert stream in _state(session).message and "holds" in _state(session).message
    assert _events(dataset)[-1] == "discard" and _state(session).discards == {"aborted": 1}
    (tmp_path / f"{FOLLOWER_ID}.jsonl.freeze").unlink(missing_ok=True)
    await session.stop()


async def test_a_leader_sample_of_another_episode_discards_it(sim_session, rig) -> None:
    rig.sim.ended_after_s = 1.5
    session, dataset = await _ready(sim_session, rig)
    await _start(session, dataset, max_episodes=1)
    await _until(lambda: _phase(session) == "recording")

    rig.leader.publish_episode = "e-9999"
    await _until(lambda: _state(session).discarded == 1, within=5)
    rig.leader.publish_episode = None

    last = _state(session).last
    assert last.discard_reason == "aborted" and "e-9999" in last.reason
    assert rig.sim.episodes[0].outcome == "aborted"  # aborted on the simulation too
    await _until(lambda: _phase(session) == "done", within=15)
    assert _state(session).saved == 1
    await session.stop()


# ----------------------------------------------------------------------
# Stopping
# ----------------------------------------------------------------------


async def test_stop_mid_episode_holds_aborts_and_discards(sim_session, rig) -> None:
    rig.sim.ended_after_s = 30.0
    session, dataset = await _ready(sim_session, rig)
    await _start(session, dataset)
    await _until(lambda: _phase(session) == "recording")
    control = session._control.value

    await session.stop_autopilot_recording()

    assert control.halted and _phase(session) == "stopped" and _state(session).message == "Stopped."
    assert session.state().control is None
    assert rig.sim.calls[-1] == "abort" and rig.sim.current.outcome == "aborted"
    assert _events(dataset)[-1] == "discard" and _state(session).discards == {"aborted": 1}
    assert not rig.leader.connected
    await _manual_start(session)  # manual recording works again
    await session.stop()


async def test_stop_mid_prepare_aborts_and_records_nothing(sim_session, rig) -> None:
    rig.sim.ready_after_s = 30.0
    session, dataset = await _ready(sim_session, rig)
    await _start(session, dataset)
    await _until(lambda: _phase(session) == "preparing" and rig.sim.episodes)

    await session.stop_autopilot_recording()

    assert _phase(session) == "stopped" and _state(session).discarded == 0
    assert rig.sim.current.outcome == "aborted"
    assert "start" not in _events(dataset)
    await session.stop()


async def test_stop_mid_save_lets_the_save_finish(sim_session, rig) -> None:
    session, dataset = await _ready(sim_session, rig)
    (Path(dataset.path) / "slow_save").write_text("1.5")
    await _start(session, dataset)
    await _until(lambda: _phase(session) == "saving")

    await session.stop_autopilot_recording()

    assert _phase(session) == "stopped"
    assert (_state(session).saved, _state(session).discarded) == (1, 0)
    assert _events(dataset)[-1] == "save" and session.state().dataset.episodes_recorded == 1
    await session.stop()


async def test_a_control_switch_stops_it_first(sim_session, rig) -> None:
    rig.sim.ended_after_s = 30.0
    session, dataset = await _ready(sim_session, rig)
    await _start(session, dataset)
    await _until(lambda: _phase(session) == "recording")

    await session.set_control(None)

    assert _phase(session) == "stopped" and _state(session).message == "The control was switched."
    assert session.state().control is None and _events(dataset)[-1] == "discard"
    await session.stop()


async def test_manual_episode_commands_are_refused_while_it_runs(sim_session, rig) -> None:
    rig.sim.ended_after_s = 30.0
    session, dataset = await _ready(sim_session, rig)
    await _start(session, dataset)
    await _until(lambda: _phase(session) == "recording")

    for command in ("start_episode", "save_episode", "discard_episode"):
        ack = await _request(
            session.bus, f'{{"command": "{command}", "request_id": "r", "task": "t"}}', session.check_command
        )
        assert not ack.ok and ack.error.error_code == "autopilot_recording", command
        assert ack.error.message == "Stop automatic recording first."
    assert session.state().dataset.is_recording and "save" not in _events(dataset)
    session.check_command("set_control")  # anything else goes through

    await session.stop_autopilot_recording()
    session.check_command("start_episode")
    await session.stop()


# ----------------------------------------------------------------------
# Refusals at start
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "code", "match"),
    [
        ("no autopilot teleoperator", "autopilot_refused", "teleoperator"),
        ("no AutopilotSpec", "autopilot_refused", "update its plugin"),
        ("not started by Studio", "simulation_not_running", "from Studio first"),
        ("unsupported scene", "autopilot_refused", "has no autopilot"),
        ("an option the scene does not take", "autopilot_refused", "recovery_fraction"),
        ("another dataset", "dataset_changed", "Another dataset"),
        ("a simulation that does not answer", "autopilot_refused", "does not answer"),
    ],
)
async def test_a_refused_start_changes_nothing(sim_session, rig, case, code, match) -> None:
    rig.has_autopilot = case != "no AutopilotSpec"
    rig.sim.supported = case != "unsupported scene"
    rig.sim.options = [] if case == "an option the scene does not take" else rig.sim.options
    rig.run = None if case == "not started by Studio" else RUN_1
    rig.sim.down = case == "a simulation that does not answer"
    session, dataset = await _ready(sim_session, rig, autopilot=case != "no autopilot teleoperator")
    target = _dataset() if case == "another dataset" else dataset

    with pytest.raises(RuntimeSessionCommandError, match=match) as refused:
        await _start(session, target, recovery_fraction=0.2)

    assert refused.value.error_code == code
    assert session.state().control is None and "start" not in rig.sim.calls and not rig.leaders
    await _manual_start(session)  # the dataset is free
    await session.stop()


# ----------------------------------------------------------------------
# Restarts, and one simulation for one session
# ----------------------------------------------------------------------


async def test_a_simulation_restart_ends_the_recording(sim_session, rig) -> None:
    rig.sim.ended_after_s = 30.0
    session, dataset = await _ready(sim_session, rig)
    statuses = subscribe_statuses(session)
    await _start(session, dataset)
    await _until(lambda: _phase(session) == "recording")
    control = session._control.value

    await session.follow_simulation(FOLLOWER_ID, SIM_URL, run=RUN_2)

    assert control.halted and _phase(session) == "failed" and "started again" in _state(session).message
    assert session.state().control is None and _events(dataset)[-1] == "discard"
    status = await next_status(statuses, "environment")
    assert status.state == "failed" and "reload the environment" in status.error
    rig.run = RUN_2
    with pytest.raises(RuntimeSessionCommandError) as refused:  # only after a reload
        await _start(session, dataset)
    assert refused.value.error_code == "environment_stale"
    await session.reload_environment()
    rig.sim.ended_after_s = 0.8
    await _start(session, dataset, max_episodes=1)
    await _until(lambda: _phase(session) == "done", within=15)
    await session.stop()


async def test_an_idle_session_reloads_on_a_restart(sim_session, rig) -> None:
    session, _dataset_loaded = await _ready(sim_session, rig)
    reloads = []
    original = session._reload

    async def counted(**kwargs):
        reloads.append(kwargs)
        return await original(**kwargs)

    session._reload = counted  # type: ignore[method-assign]
    rig.run = RUN_2  # it started again on the same port
    await session.follow_simulation(FOLLOWER_ID, SIM_URL, run=RUN_2)

    assert len(reloads) == 1 and session._connected_runs[FOLLOWER_ID] == RUN_2
    await session.stop()


async def test_two_robot_rows_for_one_simulation_cannot_both_be_loaded(sim_session) -> None:
    first = await sim_session()
    await first.load_environment(_environment(cameras=0))
    second = await sim_session()

    with pytest.raises(DeviceInUseError, match="sim-owner"):
        await second.load_environment(_environment(follower_id=uuid4(), cameras=0))

    await first.stop()
    await second.load_environment(_environment(follower_id=uuid4(), cameras=0))  # free once the first let go
    await second.stop()
