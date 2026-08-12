import numpy as np
from physicalai.runtime import LifecycleEvent, TickEvent

from runtime.callbacks.stream import StreamCallback
from runtime.contract import ErrorEvent, ObservationEvent, QueueEventSink, StateData, StateEvent

from .fakes import FakeObservation


def test_lifecycle_start_emits_connected_state_once() -> None:
    events = QueueEventSink()
    callback = StreamCallback(event_sink=events, follower_source=lambda: "hold")

    callback.on_lifecycle(
        LifecycleEvent(session_id="test", timestamp=1, event="start", metadata={"joint_names": ["joint"]})
    )

    event = events.get_nowait()
    assert event.model_dump() == {"event": "state", "data": {"connected": True, "follower_source": "hold"}}


def test_tick_emits_exact_observation_shape() -> None:
    events = QueueEventSink()
    callback = StreamCallback(event_sink=events, follower_source=lambda: "hold")
    callback.on_lifecycle(
        LifecycleEvent(session_id="test", timestamp=1, event="start", metadata={"joint_names": ["joint"]})
    )
    events.get_nowait()

    callback.on_tick(
        TickEvent(
            session_id="test",
            step=0,
            timestamp=1,
            robot_state=FakeObservation(np.array([2]), 1),
            camera_frames={},
            action_sent=np.array([2]),
            loop_duration_s=0,
            sleep_time_s=0,
            stale_obs=False,
        )
    )

    assert events.get_nowait().model_dump() == {"event": "observation", "data": {"joint.pos": 2.0}}


def test_state_and_error_events_keep_the_websocket_contract() -> None:
    assert StateEvent(data=StateData(connected=True, follower_source="teleop")).model_dump() == {
        "event": "state",
        "data": {"connected": True, "follower_source": "teleop"},
    }
    assert ErrorEvent(message="lost", error_code="leader_connection_lost").model_dump() == {
        "event": "error",
        "message": "lost",
        "error_code": "leader_connection_lost",
    }


def test_event_sink_coalesces_observations_without_dropping_state() -> None:
    events = QueueEventSink()
    events.emit(ObservationEvent(data={"joint.pos": 1}))
    events.emit(StateEvent(data=StateData(connected=True, follower_source="hold")))
    events.emit(ObservationEvent(data={"joint.pos": 2}))

    assert events.get_nowait().event == "state"
    assert events.get_nowait().model_dump() == {"event": "observation", "data": {"joint.pos": 2.0}}
