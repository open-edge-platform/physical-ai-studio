from __future__ import annotations

import asyncio
import threading
import time

import anyio
import pytest

from exceptions import DeviceInUseError, RuntimeSessionCommandError, SimulationNotRunningError
from runtime_session.session import Resolvers
from runtime_session.store import ACTION, STATE
from schemas.environment import EnvironmentWithRelations
from schemas.runtime_session import EnvironmentDraft

from .fixtures import FOLLOWER_ID, LEADER_ID, make_environment, next_status, subscribe_events, subscribe_statuses


async def _until(condition, within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        await asyncio.sleep(0.02)


async def test_loading_connects_everything_and_publishes_state(new_session, registry, fake_cameras) -> None:
    session = await new_session()
    events = subscribe_events(session)
    try:
        await session.load_environment(make_environment())

        environment = session.environment
        assert environment is not None
        assert environment.joint_names == ["shoulder_pan.pos", "gripper.pos"]
        assert [camera.key for camera in environment.cameras] == ["cam 0"]
        assert session.leader is not None and session.leader.is_connected()
        assert registry.holder_of_robot(FOLLOWER_ID) is session
        assert registry.holder_of_robot(LEADER_ID) is session
        assert (await asyncio.wait_for(events.next(), 1)).data.environment == environment

        store = session.store
        await _until(lambda: store.read(STATE) is not None and store.read_observation("cam 0") is not None)
        assert store.read(ACTION).value.tolist() == [1.0, 2.0]  # seeded: the arm holds still
    finally:
        await session.stop()


async def test_unloading_releases_everything(new_session, registry, camera_claims, fake_cameras) -> None:
    session = await new_session()
    environment = make_environment()
    await session.load_environment(environment)
    leader = session.leader
    store = session.store
    await session.unload_environment()

    assert session.environment is None
    assert session.store is store  # kept, until an environment with another spec needs a new one
    assert leader is not None and not leader.is_connected()
    assert registry.holder_of_robot(FOLLOWER_ID) is None
    assert camera_claims.holder_of(environment.cameras[0].fingerprint) is None
    # And it can load again.
    await session.load_environment(environment)
    await session.stop()
    assert registry.get(session.id) is None


async def test_loading_another_environment_replaces_the_first(new_session, fake_cameras) -> None:
    session = await new_session()
    await session.load_environment(make_environment(cameras=1))
    first_store = session.store
    await session.load_environment(make_environment(cameras=0))
    environment = session.environment
    assert environment is not None and environment.cameras == []
    assert session.store is not first_store
    await session.stop()


async def test_another_session_cannot_take_the_robot(new_session, fake_cameras) -> None:
    first, second = await new_session(), await new_session()
    await first.load_environment(make_environment())
    try:
        with pytest.raises(DeviceInUseError, match="Bench"):
            await second.load_environment(make_environment(cameras=0))
        assert second.environment is None
    finally:
        await first.stop()
    await second.load_environment(make_environment(cameras=0))
    await second.stop()


async def test_a_load_waits_for_a_closing_session_to_let_go_of_its_devices(new_session, fake_cameras) -> None:
    # A page that opens a session has usually just closed one on the same devices.
    first, second = await new_session(), await new_session()
    await first.load_environment(make_environment())
    closing = asyncio.create_task(first.stop())
    await asyncio.sleep(0)  # the teardown has started, the devices are not let go yet
    assert first.closed.is_set() and not first.finished.is_set()
    await second.load_environment(make_environment())
    assert closing.done()
    assert second.environment is not None
    await second.stop()


async def test_a_failed_connect_loads_the_environment_failed_to_reload(
    new_session, registry, robot_factory, tmp_path
) -> None:
    from tests.workers.fake_robot import FakeRobot

    failing = {FOLLOWER_ID}

    async def build(robot, _port_finder):
        connect = robot.id in failing
        return FakeRobot(str(tmp_path / f"{robot.id}"), ["shoulder_pan"], [0.0], fail_connect=connect), object()

    robot_factory.build_robot_driver.side_effect = build
    session = await new_session()
    statuses = subscribe_statuses(session)
    await session.load_environment(make_environment(cameras=0))
    for component in ("environment/follower", "environment"):
        status = await next_status(statuses, component, within=1)
        assert status.state == "failed" and "Failed to open serial port" in status.error
    # Loaded, with nothing connected: the leader's device is free again.
    assert session.environment is not None and session.leader is None
    await session.stop()


async def test_an_unplugged_leader_leaves_the_environment_running_without_teleop(
    new_session, registry, robot_factory, tmp_path
) -> None:
    from schemas.runtime_session import TeleopControlConfig
    from tests.workers.fake_robot import FakeRobot

    async def build(robot, _port_finder):
        connect = robot.id == LEADER_ID
        return FakeRobot(str(tmp_path / f"{robot.id}"), ["shoulder_pan"], [0.0], fail_connect=connect), object()

    robot_factory.build_robot_driver.side_effect = build
    session = await new_session()
    statuses = subscribe_statuses(session)
    await session.load_environment(make_environment(cameras=0))
    leader = await next_status(statuses, "environment/leader", within=1)
    assert leader.state == "failed" and "Failed to open serial port" in leader.error
    with pytest.raises(TimeoutError):  # the environment does not fail with it
        await next_status(statuses, "environment", within=1)
    assert session.environment is not None and session.leader is None
    try:
        with pytest.raises(ValueError, match="connected leader arm"):
            await session.set_control(TeleopControlConfig())
    finally:
        await session.stop()


async def test_a_dead_worker_fails_the_environment_until_it_is_loaded_again(
    new_session, registry, fake_cameras
) -> None:
    session = await new_session()
    events, statuses = subscribe_events(session), subscribe_statuses(session)
    environment = make_environment()
    await session.load_environment(environment)
    await asyncio.wait_for(events.next(), 1)  # the loaded state
    fake_cameras.return_value.read_latest.side_effect = RuntimeError("camera unplugged")

    camera = await next_status(statuses, "environment/cam 0")
    assert camera.state == "failed" and "camera unplugged" in camera.error
    failed = await next_status(statuses, "environment")
    assert failed.state == "failed" and failed.error.startswith("cam 0: ")
    state = await asyncio.wait_for(events.next(), 15)
    assert state.type == "state" and state.data.environment is not None
    # Still loaded, holding its devices, until it is loaded again.
    assert registry.holder_of_robot(FOLLOWER_ID) is session
    store = session.store
    assert not session.closed.is_set()

    fake_cameras.return_value.read_latest.side_effect = None
    await session.load_environment(environment)
    assert session.environment is not None and session.store is store
    await session.stop()


async def test_reloading_a_failed_environment_retries_in_place(new_session, fake_cameras) -> None:
    session = await new_session()
    statuses = subscribe_statuses(session)
    await session.load_environment(make_environment())
    store = session.store
    fake_cameras.return_value.read_latest.side_effect = RuntimeError("camera unplugged")
    assert (await next_status(statuses, "environment")).state == "failed"

    fake_cameras.return_value.connect.side_effect = RuntimeError("no camera")
    with pytest.raises(RuntimeSessionCommandError, match="no camera") as failed:
        await session.reload_environment()
    assert failed.value.error_code == "device_connection_failed"
    assert session.environment is not None and session.leader is None  # still loaded, to try again

    fake_cameras.return_value.connect.side_effect = None
    fake_cameras.return_value.read_latest.side_effect = None
    await session.reload_environment()
    assert session.store is store
    assert session.leader is not None and session.leader.is_connected()
    await session.stop()


SIM_A = "http://127.0.0.1:8080"
SIM_B = "http://127.0.0.1:50123"


def _simulated(http_url: str, *, cameras: int = 1):
    """An environment whose follower is simulated, its cameras resolved to the simulation at *http_url*.

    As ``resolve_cameras`` resolves them: IP cameras on the simulation's streams, whose fingerprint
    (what claims go by) stays the same wherever the simulation serves.
    """
    data = make_environment(leader=False, cameras=cameras).model_dump(mode="json")
    data["robots"][0]["robot"] |= {"type": "MuJoCo_SO101_Follower", "payload": {"http_url": http_url}}
    for index, camera in enumerate(data["cameras"]):
        camera |= {
            "id": f"00000000-0000-0000-0000-00000000000{index}",
            "driver": "ipcam",
            "fingerprint": {"robot_id": str(FOLLOWER_ID), "camera": f"cam{index}"},
            "payload": {"url": f"{http_url}/cameras/cam{index}/mjpeg", "width": 160, "height": 120, "fps": 30},
        }
    return EnvironmentWithRelations.model_validate(data)


DRAFT = EnvironmentDraft(name="Bench", robots=[{"robot_id": FOLLOWER_ID, "tele_operator": {"type": "none"}}])


class Simulations:
    """Resolves the draft to where the simulation serves now; ``gate`` holds a lookup until set."""

    def __init__(self) -> None:
        self.url = SIM_A
        self.cameras = 1
        self.error: Exception | None = None
        self.gate: asyncio.Event | None = None
        self.looking_up = asyncio.Event()
        self.lookups = 0

    async def resolve(self, _draft):
        self.lookups += 1
        url = self.url  # the address as the lookup starts
        self.looking_up.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return _simulated(url, cameras=self.cameras)


async def _session_on(new_session, simulations: Simulations):
    session = await new_session(Resolvers(environment=simulations.resolve, dataset=_unresolved, model=_unresolved))
    await session._load_draft(DRAFT, unload_incompatible=False)
    return session


async def _unresolved(_what):
    raise AssertionError("not looked up")


def _streamed_from(fake_cameras) -> str:
    """The stream the camera connected to last."""
    return fake_cameras.call_args.kwargs["config"].payload.url


async def test_reloading_looks_the_environment_up_again(new_session, fake_cameras) -> None:
    simulations = Simulations()
    session = await _session_on(new_session, simulations)
    store = session.store
    assert _streamed_from(fake_cameras) == f"{SIM_A}/cameras/cam0/mjpeg"

    simulations.url = SIM_B  # the simulation started again elsewhere; a reload connects there
    await session.reload_environment()

    assert _streamed_from(fake_cameras) == f"{SIM_B}/cameras/cam0/mjpeg"
    assert session.store is store
    await session.stop()


async def test_a_reload_refuses_an_environment_that_changed_its_spec(new_session, fake_cameras) -> None:
    simulations = Simulations()
    session = await _session_on(new_session, simulations)
    simulations.cameras = 2

    with pytest.raises(RuntimeSessionCommandError) as refused:
        await session.reload_environment()

    assert refused.value.error_code == "environment_changed"
    assert session.environment is not None
    await session.stop()


async def test_the_session_follows_its_simulation_to_a_new_address(new_session, fake_cameras) -> None:
    simulations = Simulations()
    session = await _session_on(new_session, simulations)
    statuses = subscribe_statuses(session)

    # Another robot's simulation, or this one on the same address: nothing to do.
    await session.follow_simulation(LEADER_ID, SIM_B)
    await session.follow_simulation(FOLLOWER_ID, SIM_A)
    assert simulations.lookups == 1

    simulations.url = SIM_B
    await session.follow_simulation(FOLLOWER_ID, SIM_B)

    assert _streamed_from(fake_cameras) == f"{SIM_B}/cameras/cam0/mjpeg"
    assert (await asyncio.wait_for(statuses.next(), 1)).component == "environment"  # the UI sees it reload
    await session.follow_simulation(FOLLOWER_ID, SIM_B)
    assert simulations.lookups == 2  # once per address
    await session.stop()


@pytest.mark.parametrize(
    ("change", "error_code"),
    [
        ({"error": SimulationNotRunningError("Start the simulation", robot_name="Follower")}, "simulation_not_running"),
        ({"cameras": 2}, "environment_changed"),
    ],
    ids=["stopped again", "other cameras"],
)
async def test_a_simulation_the_environment_cannot_follow_fails_it_and_says_why(
    new_session, fake_cameras, change, error_code
) -> None:
    simulations = Simulations()
    session = await _session_on(new_session, simulations)
    events, statuses = subscribe_events(session), subscribe_statuses(session)
    simulations.url = SIM_B
    for name, value in change.items():
        setattr(simulations, name, value)

    await session.follow_simulation(FOLLOWER_ID, SIM_B)

    failed = await next_status(statuses, "environment")
    assert failed.state == "failed" and SIM_B in (failed.error or "")
    error = await asyncio.wait_for(events.next(), 1)
    while error.type != "error":
        error = await asyncio.wait_for(events.next(), 1)
    assert error.error_code == error_code and error.session_alive
    assert session.environment is not None  # still loaded, to reload
    await session.stop()


async def test_a_stale_follow_leaves_a_newly_loaded_environment_alone(new_session, fake_cameras) -> None:
    simulations = Simulations()
    session = await _session_on(new_session, simulations)
    simulations.url, simulations.gate = SIM_B, asyncio.Event()
    following = asyncio.create_task(session.follow_simulation(FOLLOWER_ID, SIM_B))
    await asyncio.wait_for(simulations.looking_up.wait(), 1)

    # The user loads an environment (here, the same draft afresh) while the follow looks it up.
    simulations.gate.set()
    simulations.gate = None
    await session.load_environment(_simulated(SIM_B), draft=DRAFT)
    connects = fake_cameras.call_count
    await asyncio.wait_for(following, 5)

    assert fake_cameras.call_count == connects  # the old follow did not reload the new environment
    await session.stop()


async def test_a_restart_reported_while_loading_is_followed_once_loaded(new_session, fake_cameras) -> None:
    simulations = Simulations()
    session = await new_session(Resolvers(environment=simulations.resolve, dataset=_unresolved, model=_unresolved))
    simulations.gate = asyncio.Event()
    loading = asyncio.create_task(session._load_draft(DRAFT, unload_incompatible=False))
    await asyncio.wait_for(simulations.looking_up.wait(), 1)  # resolved at SIM_A, not loaded yet

    simulations.url = SIM_B
    await session.follow_simulation(FOLLOWER_ID, SIM_B)  # nothing loaded to follow yet
    simulations.gate.set()
    simulations.gate = None
    await asyncio.wait_for(loading, 10)

    assert _streamed_from(fake_cameras) == f"{SIM_B}/cameras/cam0/mjpeg"
    await session.stop()


async def test_a_follow_the_socket_leaves_while_reconnecting_finishes_before_teardown(
    new_session, fake_cameras, registry
) -> None:
    simulations = Simulations()
    session = await _session_on(new_session, simulations)
    simulations.url = SIM_B
    reconnecting, release = threading.Event(), threading.Event()
    build = fake_cameras.side_effect

    def slow_to_open(**kwargs):
        if kwargs["config"].payload.url.startswith(SIM_B):
            reconnecting.set()
            release.wait(5)
        return fake_cameras.return_value

    fake_cameras.side_effect = slow_to_open
    finished_early: list[bool] = []

    async def release_later() -> None:
        await asyncio.sleep(0.2)
        finished_early.append(session.finished.is_set())
        release.set()

    # As the socket runs it: in a task group, cancelled when the socket closes, then the session stops.
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(session.follow_simulation, FOLLOWER_ID, SIM_B)
        await asyncio.to_thread(reconnecting.wait, 5)
        releasing = asyncio.create_task(release_later())
        stopping = asyncio.create_task(session.stop())
        tasks.cancel_scope.cancel()
    await asyncio.wait_for(stopping, 15)
    await releasing
    fake_cameras.side_effect = build

    assert finished_early == [False]  # teardown waited for the reload instead of overtaking it
    assert session.environment is None and session.store is None
    assert registry.holder_of_robot(FOLLOWER_ID) is None


async def test_a_follow_still_looking_up_when_the_session_stops_does_nothing(new_session, fake_cameras) -> None:
    simulations = Simulations()
    session = await _session_on(new_session, simulations)
    simulations.url, simulations.gate = SIM_B, asyncio.Event()
    following = asyncio.create_task(session.follow_simulation(FOLLOWER_ID, SIM_B))
    await asyncio.wait_for(simulations.looking_up.wait(), 1)
    await session.stop()
    connects = fake_cameras.call_count

    simulations.gate.set()
    await asyncio.wait_for(following, 5)

    assert fake_cameras.call_count == connects
    assert session.environment is None


async def test_stopping_tells_the_socket(new_session) -> None:
    session = await new_session()
    events = subscribe_events(session)
    await session.stop()
    event = await asyncio.wait_for(events.next(), 1)
    assert event.error_code == "session_closed"
    assert not event.session_alive


async def _leader_at(robot_factory, tmp_path, position: list[float]) -> None:
    from tests.workers.fake_robot import FakeRobot

    async def build(robot, _port_finder):
        start = position if robot.id == LEADER_ID else [1.0, 2.0]
        return FakeRobot(str(tmp_path / f"{robot.id}.jsonl"), ["shoulder_pan", "gripper"], start), object()

    robot_factory.build_robot_driver.side_effect = build


async def test_teleop_drives_the_follower_and_none_holds_it(new_session, robot_factory, tmp_path) -> None:
    from schemas.runtime_session import TeleopControlConfig

    await _leader_at(robot_factory, tmp_path, [5.0, 6.0])
    session = await new_session()
    await session.load_environment(make_environment(cameras=0))
    store = session.store
    try:
        await session.set_control(TeleopControlConfig(hz=50, grace_s=0.2))
        assert session.state().control == TeleopControlConfig(hz=50, grace_s=0.2)
        await _until(lambda: store.read(STATE).value.tolist() == [5.0, 6.0])

        await session.set_control(None)
        assert session.state().control is None
        assert store.read(ACTION).value.tolist() == [5.0, 6.0]  # held where it got to
    finally:
        await session.stop()


async def test_set_control_needs_an_environment(new_session) -> None:
    from schemas.runtime_session import TeleopControlConfig

    session = await new_session()
    with pytest.raises(RuntimeSessionCommandError, match="Load an environment"):
        await session.set_control(TeleopControlConfig())
    await session.stop()


async def test_teleop_without_a_leader_is_refused_and_holds(new_session) -> None:
    from schemas.runtime_session import TeleopControlConfig

    session = await new_session()
    await session.load_environment(make_environment(leader=False, cameras=0))
    try:
        with pytest.raises(ValueError, match="leader arm"):
            await session.set_control(TeleopControlConfig())
        assert session.state().control is None
    finally:
        await session.stop()


async def test_a_failing_control_reports_and_falls_back_to_hold(new_session) -> None:
    from schemas.runtime_session import TeleopControlConfig

    session = await new_session()
    statuses = subscribe_statuses(session)
    await session.load_environment(make_environment(cameras=0))
    try:

        def unplugged():
            raise ConnectionError("leader unplugged")

        session.leader.get_observation = unplugged
        await session.set_control(TeleopControlConfig())
        status = await next_status(statuses, "control", within=5)
        assert status.state == "failed" and "leader unplugged" in status.error
        await _until(lambda: session.state().control is None)
        assert session.environment is not None  # the environment stays loaded
    finally:
        await session.stop()


async def test_a_stopped_session_loads_nothing(new_session, registry) -> None:
    session = await new_session()
    await session.stop()
    assert registry.get(session.id) is None
    with pytest.raises(RuntimeSessionCommandError, match="closed"):
        await session.load_environment(make_environment(cameras=0))
