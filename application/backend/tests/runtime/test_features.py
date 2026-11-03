import numpy as np

from runtime.features import feature_names, observation_to_dict, sanitize_camera_name

from .fakes import FakeObservation


def test_feature_names_and_observation_values_are_joint_positions() -> None:
    observation = FakeObservation(
        joint_positions=np.array([1, 2]),
        timestamp=1,
        sensor_data={"velocities": np.array([3, 4])},
    )

    assert feature_names(["a", "b"]) == ["a.pos", "b.pos"]
    assert observation_to_dict(["a", "b"], observation) == {
        "a.pos": 1.0,
        "b.pos": 2.0,
    }


def test_camera_name_sanitization_remains_stable() -> None:
    assert sanitize_camera_name("Front/Left Camera") == "front_left camera"
    assert sanitize_camera_name("Bob's view") == "bob_s view"
    assert sanitize_camera_name("global view") == "global view"
    assert sanitize_camera_name("Left Camera") == "left camera"
    assert sanitize_camera_name("Front") == "front"
    assert sanitize_camera_name("  -front-  ") == "  -front-  "
    assert sanitize_camera_name("grabber") == "grabber"
    assert sanitize_camera_name("cámara-1") == "c_mara-1"
    assert sanitize_camera_name("bob\\view") == "bob_view"
    assert sanitize_camera_name("my\ncam") == "my_cam"
    assert sanitize_camera_name(sanitize_camera_name("Bob's view")) == sanitize_camera_name("Bob's view")
