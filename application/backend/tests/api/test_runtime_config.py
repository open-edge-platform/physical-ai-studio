from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from physicalai.config import Config

from api.environments import get_runtime_config
from schemas.environment import EnvironmentWithRelations


def _environment(*, with_leader: bool) -> EnvironmentWithRelations:
    teleoperator = (
        {
            "type": "robot",
            "robot_id": str(uuid4()),
            "robot": {
                "id": str(uuid4()),
                "name": "Leader",
                "type": "SO101_Leader",
                "payload": {"connection_string": "/dev/ttyACM1", "serial_number": "leader"},
            },
        }
        if with_leader
        else {"type": "none"}
    )
    return EnvironmentWithRelations.model_validate(
        {
            "id": str(uuid4()),
            "name": "Test",
            "robots": [
                {
                    "robot": {
                        "id": str(uuid4()),
                        "name": "Follower",
                        "type": "SO101_Follower",
                        "payload": {"connection_string": "/dev/ttyACM0", "serial_number": "follower"},
                    },
                    "tele_operator": teleoperator,
                }
            ],
            "cameras": [],
        }
    )


async def test_runtime_config_download_is_yaml_attachment(mocker) -> None:
    environment = _environment(with_leader=True)
    environment_service = SimpleNamespace(get_environment_by_id=mocker.AsyncMock(return_value=environment))
    document = Config(
        "physicalai.runtime.RobotRuntime",
        {"robot": {"class_path": "tests.runtime.fakes.FakeRobot", "init_args": {}}, "fps": 30.0},
    ).to_dict()
    build = mocker.patch("api.environments.build_runtime_config", return_value=document)

    response = await get_runtime_config(
        project_id=uuid4(),
        environment_id=environment.id,
        environment_service=environment_service,
        robot_client_factory=mocker.MagicMock(),
    )

    assert response.media_type == "application/yaml"
    assert response.headers["content-disposition"] == 'attachment; filename="runtime.yaml"'
    assert b"physicalai.runtime.RobotRuntime" in response.body
    # A config can be exported for a rig that is not plugged in right now.
    assert build.call_args.kwargs["allow_stored_port"] is True


async def test_runtime_config_download_requires_leader(mocker) -> None:
    environment = _environment(with_leader=False)
    environment_service = SimpleNamespace(get_environment_by_id=mocker.AsyncMock(return_value=environment))

    with pytest.raises(HTTPException, match="leader robot"):
        await get_runtime_config(
            project_id=uuid4(),
            environment_id=environment.id,
            environment_service=environment_service,
            robot_client_factory=mocker.MagicMock(),
        )
