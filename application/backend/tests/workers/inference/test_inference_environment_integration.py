from frame_source.video_capture_base import VideoCaptureBase
from robots.robot_client import RobotClient
import asyncio
from unittest.mock import patch, MagicMock, Mock, AsyncMock

import numpy as np
import pytest
import torch

from robots.robot_client_factory import RobotClientFactory
from schemas.environment import EnvironmentWithRelations
from workers.inference.inference_environment_integration import InferenceEnvironmentIntegration

test_environment = {
    "id": "7656679b-25fe-4af5-a19d-73e7df16f384",
    "name": "Home Setup",
    "robots": [
        {
            "robot": {
                "id": "c3f3f886-8813-4b3b-ba48-165cdaa39995",
                "name": "Khaos",
                "connection_string": "",
                "serial_number": "5AA9017083",
                "type": "SO101_Follower",
            },
            "tele_operator": {"type": "none"},
        }
    ],
    "cameras": [
        {
            "id": "3ed60255-04ae-407b-8e2c-c3281847a4e0",
            "driver": "usb_camera",
            "name": "grabber",
            "fingerprint": "/dev/video0:0",
            "hardware_name": None,
            "payload": {"width": 640, "height": 480, "fps": 30},
        },
        {
            "id": "4629e172-2aa7-4fde-86b1-e19eb1d210ff",
            "driver": "usb_camera",
            "name": "front",
            "fingerprint": "/dev/video6:6",
            "hardware_name": None,
            "payload": {"width": 640, "height": 480, "fps": 30},
        },
    ],
}


@pytest.fixture
def mock_robot_client():
    client = MagicMock(spec=RobotClient)
    client.features.return_value = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos"
    ]
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.read_state = AsyncMock(
        return_value={
            "state": {
                "shoulder_pan.pos": -8.705526116578355,
                "shoulder_lift.pos": -98.16753926701571,
                "elbow_flex.pos": 95.98393574297188,
                "wrist_flex.pos": 73.85993485342019,
                "wrist_roll.pos": -13.84615384615384,
                "gripper.pos": 26.885644768856448,
            }
        }
    )
    return client


@pytest.fixture
def mock_robot_client_factory(mock_robot_client):
    factory = MagicMock(spec=RobotClientFactory)
    factory.build = AsyncMock(return_value=mock_robot_client)
    return factory


@pytest.fixture
def mock_camera():
    camera = MagicMock(spec=VideoCaptureBase)
    camera.connect = Mock()
    camera.start_async = Mock
    camera.stop = Mock()
    camera.disconnect = Mock()
    camera.get_latest_frame.return_value = (True, np.zeros([480, 640, 3], dtype=np.uint8))
    return camera


@pytest.fixture
def inference_environment_integration(mock_robot_client_factory, mock_camera):
    environment = EnvironmentWithRelations.model_validate(test_environment)
    factory = mock_robot_client_factory

    with patch(
        "workers.inference.inference_environment_integration.create_frames_source_from_camera",
        return_value=mock_camera,
    ):
        subject = InferenceEnvironmentIntegration(environment, factory)
        asyncio.run(subject.setup())
        yield subject
        asyncio.run(subject.teardown())


class TestInferenceEnvironmentIntegration:
    def test_get_observation(self, inference_environment_integration: InferenceEnvironmentIntegration):
        observation = asyncio.run(inference_environment_integration.get_observation())
        assert observation is not None
        assert "shoulder_pan.pos" in observation
        assert "3ed60255-04ae-407b-8e2c-c3281847a4e0" in observation  # camera id 1
        assert "4629e172-2aa7-4fde-86b1-e19eb1d210ff" in observation  # camera id 2

    def test_transform_observation_to_model_input(
        self, inference_environment_integration: InferenceEnvironmentIntegration
    ):
        observation = asyncio.run(inference_environment_integration.get_observation())
        assert observation is not None
        phy_ai_obs = inference_environment_integration.format_model_input_observation(observation)
        assert phy_ai_obs.state is not None
        assert phy_ai_obs.state.shape == torch.Size([1, 6])
        assert phy_ai_obs.images is not None
        assert "front" in phy_ai_obs.images
        assert "grabber" in phy_ai_obs.images

    def test_transform_observation_to_report_to_ui(
        self, inference_environment_integration: InferenceEnvironmentIntegration
    ):
        observation = asyncio.run(inference_environment_integration.get_observation())
        assert observation is not None
        report_obs = inference_environment_integration.format_observation_for_reporting(observation, 0)
        assert "shoulder_pan.pos" in report_obs["state"]
        assert "3ed60255-04ae-407b-8e2c-c3281847a4e0" in report_obs["cameras"]  # camera id 1
        assert "4629e172-2aa7-4fde-86b1-e19eb1d210ff" in report_obs["cameras"]  # camera id 2

    def test_teardown_disconnects_robot_and_stops_cameras(
        self,
        inference_environment_integration,
        mock_robot_client,
        mock_camera,
    ):
        asyncio.run(inference_environment_integration.teardown())
        mock_robot_client.disconnect.assert_awaited_once()
        assert mock_camera.stop.call_count == 2  # one per camera
