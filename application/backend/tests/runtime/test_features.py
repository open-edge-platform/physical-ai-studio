import numpy as np

from runtime.features import feature_names, observation_to_dict, sanitize_camera_name

from .fakes import FakeObservation


def test_feature_names_and_observation_values_have_the_same_order() -> None:
    observation = FakeObservation(
        joint_positions=np.array([1, 2]),
        timestamp=1,
        sensor_data={"velocities": np.array([3, 4])},
    )

    assert feature_names(["a", "b"], include_velocities=True) == ["a.pos", "b.pos", "a.vel", "b.vel"]
    assert observation_to_dict(["a", "b"], observation, include_velocities=True) == {
        "a.pos": 1.0,
        "b.pos": 2.0,
        "a.vel": 3.0,
        "b.vel": 4.0,
    }


def test_camera_name_sanitization_remains_stable() -> None:
    assert sanitize_camera_name("Front/Left Camera") == "front_left camera"
