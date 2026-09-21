# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from schemas.job import SshTrainJobPayload
from schemas.remote_server import RemoteServer
from services.training_targets.ssh import SshTrainingTargetHandler


def _remote_server(**overrides) -> RemoteServer:
    defaults = {
        "id": uuid4(),
        "name": "lab-gpu-box",
        "ssh_host_alias": "gpu-box",
        "device_type": "cuda",
        "last_check_status": "healthy",
    }
    defaults.update(overrides)
    return RemoteServer(**defaults)


def _payload(remote_server_id) -> SshTrainJobPayload:
    return SshTrainJobPayload(
        dataset_id=uuid4(),
        project_id=uuid4(),
        model_name="test-model",
        policy="act",
        remote_server_id=remote_server_id,
    )


class TestSshTrainingTargetHandlerPrepare:
    @pytest.mark.asyncio
    async def test_pins_the_remote_server_name_onto_the_payload(self, monkeypatch):
        # A job's badge/label must stay traceable to the server it actually ran
        # on even after that server is later deleted, since `remote_server_id`
        # alone resolves to nothing once the row is gone.
        remote_server = _remote_server()
        remote_server_service = AsyncMock()
        remote_server_service.get_remote_server.return_value = remote_server
        monkeypatch.setattr(
            "services.training_targets.ssh.resolve_alias",
            lambda *_args, **_kwargs: type("Resolved", (), {"found": True})(),
        )

        handler = SshTrainingTargetHandler(remote_server_service)
        prepared = await handler.prepare(_payload(remote_server.id))

        assert prepared.remote_server_name == remote_server.name
