from __future__ import annotations

import asyncio
import time

import pytest

from exceptions import DeviceInUseError, RuntimeSessionCommandError
from runtime_session.store import ACTION, STATE

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


async def test_teleop_switches_to_relative_over_the_bus_without_stopping(new_session, robot_factory, tmp_path) -> None:
    from runtime_session.bus import command_key
    from schemas.runtime_session import CommandDone, SetTeleopSlowdownCommand, StateMessage, TeleopControlConfig

    await _leader_at(robot_factory, tmp_path, [5.0, 6.0])
    session = await new_session()
    await session.load_environment(make_environment(cameras=0))
    events = subscribe_events(session)
    try:
        await session.set_control(TeleopControlConfig(hz=50, grace_s=0))
        control = session._control.value
        command = SetTeleopSlowdownCommand(request_id="1", slowdown=2.0)
        await session.bus.request(command_key(command.command), command, CommandDone, timeout=2.0)

        assert session._control.value is control  # the same control, still running
        assert session.state().control == TeleopControlConfig(hz=50, grace_s=0, slowdown=2.0)
        async with asyncio.timeout(2.0):  # and the socket is told
            while not isinstance(event := await events.next(), StateMessage) or event.data.control != control.config:
                pass
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
