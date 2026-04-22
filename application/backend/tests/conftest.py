from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from robots.robot_client import RobotClient
from robots.robot_client_factory import RobotClientFactory
from schemas.dataset import Dataset
from schemas.environment import EnvironmentWithRelations
from schemas.model import Model


@pytest.fixture
def mock_robot_client():
    client = MagicMock(spec=RobotClient)
    client.features.return_value = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
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
def test_environment():
    return EnvironmentWithRelations.model_validate(
        {
            "id": "7656679b-25fe-4af5-a19d-73e7df16f384",
            "name": "Home Setup",
            "robots": [
                {
                    "robot": {
                        "id": "c3f3f886-8813-4b3b-ba48-165cdaa39995",
                        "name": "Khaos",
                        "type": "SO101_Follower",
                        "payload": {
                            "connection_string": "",
                            "serial_number": "5AA9017083",
                        },
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
    )


@pytest.fixture
def test_actions():
    return {
        "shoulder_pan.pos": -11.076923076923077,
        "shoulder_lift.pos": 56.043956043956044,
        "elbow_flex.pos": -10.197802197802197,
        "wrist_flex.pos": 69.45054945054945,
        "wrist_roll.pos": -24.791208791208792,
        "gripper.pos": 12.364425162689804,
    }


@pytest.fixture
def test_observation():
    return {
        "shoulder_pan.pos": -11.076923076923077,
        "shoulder_lift.pos": 56.043956043956044,
        "elbow_flex.pos": -10.197802197802197,
        "wrist_flex.pos": 69.45054945054945,
        "wrist_roll.pos": -24.791208791208792,
        "gripper.pos": 12.364425162689804,
        "3ed60255-04ae-407b-8e2c-c3281847a4e0": np.zeros([480, 640, 3], dtype=np.uint8),
        "4629e172-2aa7-4fde-86b1-e19eb1d210ff": np.zeros([480, 640, 3], dtype=np.uint8),
    }


@pytest.fixture
def test_model():
    return Model.model_validate(
        {
            "name": "foo",
            "policy": "act",
            "path": "/dev/null",
            "project_id": "35b48dc9-31df-40be-b295-08ae1d5378b1",
            "dataset_id": "93cffdc2-db6d-47bf-ac0c-4e5a727cbf0d",
            "properties": {},
            "snapshot_id": "f5e2cb67-3df2-4f16-bdfd-8b0782dd9e02",
        }
    )


@pytest.fixture
def test_dataset():
    return Dataset.model_validate(
        {
            "name": "Collect blocks",
            "path": "/some/path/to/dataset",
            "default_task": "Collect blocks",
            "project_id": "35b48dc9-31df-40be-b295-08ae1d5378b1",
            "environment_id": "7656679b-25fe-4af5-a19d-73e7df16f384",
        }
    )
