import asyncio
import ctypes
import multiprocessing as mp
from multiprocessing import Array as MPArray
from unittest.mock import MagicMock, patch

import pytest

from control.data_registry import CameraRegistryEntry, EnvironmentDataRegistry, RobotRegistryEntry
from control.environment_integration import EnvironmentIntegration
from control.utils import format_observation_for_model, format_observation_for_reporting, get_observation_from_manifest

FEATURES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CAM_ID_1 = "3ed60255-04ae-407b-8e2c-c3281847a4e0"
CAM_ID_2 = "4629e172-2aa7-4fde-86b1-e19eb1d210ff"


def _make_manifest():
    n = len(FEATURES)
    robot = RobotRegistryEntry(
        name="Khaos",
        type="SO101_Follower",
        features=list(FEATURES),
        state=MPArray(ctypes.c_double, n),
        actions=MPArray(ctypes.c_double, n),
        action_read_state=mp.Value(ctypes.c_int, 0),
    )
    cameras = [
        CameraRegistryEntry(
            id=CAM_ID_1, name="grabber", width=640, height=480, frame_data=MPArray(ctypes.c_uint8, 640 * 480 * 3)
        ),
        CameraRegistryEntry(
            id=CAM_ID_2, name="front", width=640, height=480, frame_data=MPArray(ctypes.c_uint8, 640 * 480 * 3)
        ),
    ]
    return EnvironmentDataRegistry(robot=robot, cameras=cameras)


def _make_worker_mock():
    worker = MagicMock()
    worker.loaded_event = mp.Event()
    worker.loaded_event.set()
    return worker


@pytest.fixture
def stop_event():
    return mp.Event()


@pytest.fixture
def manifest():
    return _make_manifest()


class TestEnvironmentIntegration:
    def _make_env(self, test_environment, mock_robot_client_factory, stop_event):
        teleop_mock = _make_worker_mock()
        teleop_mock._output_state = MagicMock()
        teleop_mock._output_actions = MagicMock()
        teleop_mock._action_read_state = MagicMock()

        cam_mock = _make_worker_mock()
        cam_mock._width = 640
        cam_mock._height = 480
        cam_mock._frame_data = MagicMock()

        subject = EnvironmentIntegration(test_environment, mock_robot_client_factory, stop_event)

        with (
            patch("control.environment_integration.TeleoperateWorker", return_value=teleop_mock),
            patch("control.environment_integration.CameraWorker", return_value=cam_mock),
        ):
            asyncio.run(subject.setup_environment())

        return subject, teleop_mock, cam_mock

    def test_manifest_is_none_before_setup(self, mock_robot_client_factory, test_environment, stop_event):
        subject = EnvironmentIntegration(test_environment, mock_robot_client_factory, stop_event)
        assert subject.manifest is None

    def test_manifest_created_after_setup(self, mock_robot_client_factory, test_environment, stop_event):
        subject, _, _ = self._make_env(test_environment, mock_robot_client_factory, stop_event)
        assert subject.manifest is not None

    def test_manifest_has_robot_features(self, mock_robot_client_factory, test_environment, stop_event):
        subject, _, _ = self._make_env(test_environment, mock_robot_client_factory, stop_event)
        assert "shoulder_pan.pos" in subject.manifest.robot.features

    def test_manifest_has_two_cameras(self, mock_robot_client_factory, test_environment, stop_event):
        subject, _, _ = self._make_env(test_environment, mock_robot_client_factory, stop_event)
        assert len(subject.manifest.cameras) == 2

    def test_teardown_stops_workers(self, mock_robot_client_factory, test_environment, stop_event):
        subject, teleop_mock, cam_mock = self._make_env(test_environment, mock_robot_client_factory, stop_event)
        subject.teardown()
        teleop_mock.stop.assert_called()
        cam_mock.stop.assert_called()


class TestGetObservationFromRegistry:
    def test_returns_expected_keys(self, manifest):
        obs = get_observation_from_manifest(manifest)
        assert "state" in obs
        assert "action" in obs
        assert "images" in obs
        assert "timestamp" in obs

    def test_images_keyed_by_camera_id(self, manifest):
        obs = get_observation_from_manifest(manifest)
        assert CAM_ID_1 in obs["images"]
        assert CAM_ID_2 in obs["images"]

    def test_state_length_matches_features(self, manifest):
        obs = get_observation_from_manifest(manifest)
        assert len(obs["state"]) == len(FEATURES)


class TestFormatObservationForModel:
    def test_state_shape(self, manifest):
        obs = get_observation_from_manifest(manifest)
        result = format_observation_for_model(obs, manifest, task="")
        assert result.state is not None
        assert result.state.shape == (1, len(FEATURES))

    def test_images_keyed_by_camera_name(self, manifest):
        obs = get_observation_from_manifest(manifest)
        result = format_observation_for_model(obs, manifest, task="")
        assert result.images is not None
        assert "grabber" in result.images
        assert "front" in result.images


class TestFormatObservationForReporting:
    def test_returns_expected_keys(self, manifest):
        obs = get_observation_from_manifest(manifest)
        result = format_observation_for_reporting(obs, manifest)
        assert "state" in result
        assert "actions" in result
        assert "cameras" in result
        assert "timestamp" in result

    def test_state_and_actions_keyed_by_feature_names(self, manifest):
        obs = get_observation_from_manifest(manifest)
        result = format_observation_for_reporting(obs, manifest)
        for feature in FEATURES:
            assert feature in result["state"]
            assert feature in result["actions"]

    def test_cameras_keyed_by_camera_id(self, manifest):
        obs = get_observation_from_manifest(manifest)
        result = format_observation_for_reporting(obs, manifest)
        assert CAM_ID_1 in result["cameras"]
        assert CAM_ID_2 in result["cameras"]
