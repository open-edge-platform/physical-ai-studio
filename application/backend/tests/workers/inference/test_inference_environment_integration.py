import torch
import numpy as np
from unittest.mock import patch
import asyncio
from workers.inference.inference_environment_integration import InferenceEnvironmentIntegration
from uuid import UUID
import datetime
from schemas.environment import EnvironmentWithRelations
from robots.robot_client_factory import RobotClientFactory
import pytest
from settings import get_settings

from utils.serial_robot_tools import RobotConnectionManager
from services.robot_calibration_service import RobotCalibrationService

test_environment = {
    'id': UUID('7656679b-25fe-4af5-a19d-73e7df16f384'),
    'created_at': datetime.datetime(2026, 3, 3, 9, 36, 13),
    'updated_at': datetime.datetime(2026, 3, 3, 9, 36, 13),
    'name': 'Home Setup',
    'robots': [
        {
            'robot': {
                'id': UUID('c3f3f886-8813-4b3b-ba48-165cdaa39995'),
                'created_at': datetime.datetime(2026, 3, 3, 9, 35, 9),
                'updated_at': datetime.datetime(2026, 3, 3, 9, 35, 9),
                'name': 'Khaos',
                'connection_string': '',
                'serial_number': '5AA9017083',
                'type': 'SO101_Follower',
                'active_calibration_id': UUID('877e5a03-47a6-4383-b15f-807259cd9691')
            },
            'tele_operator': {
                'type': 'robot',
                'robot_id': UUID('9da8143e-ea83-4811-88a8-5b4b02ee234d'),
                'robot': {
                    'id': UUID('9da8143e-ea83-4811-88a8-5b4b02ee234d'),
                    'created_at': datetime.datetime(2026, 3, 3, 9, 35, 19),
                    'updated_at': datetime.datetime(2026, 3, 3, 9, 35, 19),
                    'name': 'Khronos',
                    'connection_string': '',
                    'serial_number': '5A7A016060',
                    'type': 'SO101_Leader',
                    'active_calibration_id': UUID('40399827-95bd-4151-bc20-893a5f51db8b')
                }
            }
        }
    ], 'cameras': [
        {
            'id': '3ed60255-04ae-407b-8e2c-c3281847a4e0',
            'driver': 'usb_camera',
            'created_at': datetime.datetime(2026, 3, 3, 9, 35, 53),
            'updated_at': datetime.datetime(2026, 3, 3, 9, 35, 53),
            'name': 'grabber',
            'fingerprint': '/dev/video0:0',
            'hardware_name': 'Innomaker-U20CAM-1080p-S1: Inno',
            'payload': {'width': 640, 'height': 480, 'fps': 30, 'exposure': None, 'gain': None}
        },
        {
            'id': '4629e172-2aa7-4fde-86b1-e19eb1d210ff',
            'driver': 'usb_camera',
            'created_at': datetime.datetime(2026, 3, 3, 9, 35, 46),
            'updated_at': datetime.datetime(2026, 3, 3, 9, 35, 46),
            'name': 'front',
            'fingerprint': '/dev/video6:6',
            'hardware_name': 'Intel(R) RealSense(TM) Depth Ca',
            'payload': {'width': 640, 'height': 480, 'fps': 30, 'exposure': None, 'gain': None}
        }
    ]
}

class FakeFrameSourceCamera:
    def connect(self):
        pass

    def start_async(self):
        pass

    def get_latest_frame(self):
        return True, np.zeros([480,640,3],dtype=np.uint8)

    def stop(self):
        pass

    def disconnect(self):
        pass


@pytest.fixture
def inference_environment_integration():

    robot_manager = RobotConnectionManager()
    asyncio.run(robot_manager.find_robots())
    settings = get_settings()
    calibration_service = RobotCalibrationService(robot_manager, settings)
    factory = RobotClientFactory(robot_manager, calibration_service)
    environment = EnvironmentWithRelations.model_validate(test_environment)

    with patch("workers.inference.inference_environment_integration.create_frames_source_from_camera",
               return_value=FakeFrameSourceCamera()):
        subject = InferenceEnvironmentIntegration(environment, factory)
        asyncio.run(subject.setup())
        yield subject
        asyncio.run(subject.teardown())

class TestInferenceEnvironmentIntegration:
    # TODO: Hardware in the loop. FIX
    def test_get_observation(self, inference_environment_integration: InferenceEnvironmentIntegration):
        observation = asyncio.run(inference_environment_integration.get_observation())
        assert observation is not None
        assert "shoulder_pan.pos" in observation
        assert "3ed60255-04ae-407b-8e2c-c3281847a4e0" in observation  #camera id 1
        assert "4629e172-2aa7-4fde-86b1-e19eb1d210ff" in observation  #camera id 2

    def test_transform_observation_to_model_input(self, inference_environment_integration: InferenceEnvironmentIntegration):
        observation = asyncio.run(inference_environment_integration.get_observation())
        assert observation is not None
        phy_ai_obs = inference_environment_integration.format_model_input_observation(observation)
        assert phy_ai_obs.state is not None
        assert phy_ai_obs.state.shape == torch.Size([1, 6])
        assert phy_ai_obs.images is not None
        assert "front" in phy_ai_obs.images
        assert "grabber" in phy_ai_obs.images

    def test_transform_observation_to_report_to_ui(self, inference_environment_integration: InferenceEnvironmentIntegration):
        observation = asyncio.run(inference_environment_integration.get_observation())
        assert observation is not None
        report_obs = inference_environment_integration.format_observation_for_reporting(observation, 0)
        assert "shoulder_pan.pos" in report_obs["state"]
        assert "3ed60255-04ae-407b-8e2c-c3281847a4e0" in report_obs["cameras"]  #camera id 1
        assert "4629e172-2aa7-4fde-86b1-e19eb1d210ff" in report_obs["cameras"]  #camera id 2
