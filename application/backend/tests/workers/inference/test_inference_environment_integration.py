import asyncio
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from frame_source.video_capture_base import VideoCaptureBase

from workers.inference.inference_environment_integration import InferenceEnvironmentIntegration


@pytest.fixture
def mock_camera():
    camera = MagicMock(spec=VideoCaptureBase)
    camera.connect = Mock()
    camera.start_async = Mock
    camera.stop = Mock()
    camera.disconnect = Mock()
    camera.read.return_value = (True, np.zeros([480, 640, 3], dtype=np.uint8))
    return camera


@pytest.fixture
def inference_environment_integration(mock_robot_client_factory, mock_camera, test_environment):
    factory = mock_robot_client_factory

    with patch(
        "workers.inference.inference_environment_integration.create_frames_source_from_camera",
        return_value=mock_camera,
    ):
        subject = InferenceEnvironmentIntegration(test_environment, factory)
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
