from __future__ import annotations

from runtime import config_builder
from runtime.dataset_features import build_lerobot_dataset_features
from runtime.features import feature_names, sanitize_camera_name


def test_features_come_from_joint_names() -> None:
    names = ["shoulder_pan", "wrist"]
    features = build_lerobot_dataset_features(joint_names=names, camera_specs={})
    expected = feature_names(names, include_velocities=False)

    assert features["observation.state"]["names"] == expected
    assert features["action"]["names"] == expected


def test_camera_features_use_frame_shape() -> None:
    features = build_lerobot_dataset_features(
        joint_names=["joint"],
        camera_specs={"front": (720, 1280, 3)},
    )

    assert features["observation.images.front"]["shape"] == (720, 1280, 3)


def test_camera_keys_are_sanitized() -> None:
    assert config_builder.sanitize_camera_name is sanitize_camera_name
    key = sanitize_camera_name("Front/Left Camera")
    features = build_lerobot_dataset_features(joint_names=["joint"], camera_specs={key: (480, 640, 3)})

    assert f"observation.images.{key}" in features
