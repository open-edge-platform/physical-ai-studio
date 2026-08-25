# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SSH-provisioned training target: dials a remote server registered by SSH host alias."""

from __future__ import annotations

from exceptions import RemoteResumeUnsupportedError, RemoteServerAliasNotFoundError, RemoteServerNotReadyError
from schemas.job import SshTrainJobPayload, TrainingTarget, TrainJobPayload
from services.remote_server_service import RemoteServerService
from services.ssh_config_reader import resolve_alias
from settings import get_settings


class SshTrainingTargetHandler:
    """Validates and keys jobs that train on an SSH-provisioned remote server."""

    def __init__(self, remote_server_service: RemoteServerService) -> None:
        self.remote_server_service = remote_server_service

    async def prepare(self, payload: TrainJobPayload) -> TrainJobPayload:
        """Verify the selected server's last preflight result and current SSH-config alias.

        Studio never dials SSH from job submission, so this only consults the
        persisted last-check summary from the explicit save/verify actions
        (`last_check_status != "healthy"` also catches a server that has never
        been checked at all). A renamed/removed Host entry is caught by
        re-parsing the config file directly (no SSH dial), so it fails closed
        even if the server's last preflight happened before the alias
        disappeared.

        `get_training_target_handler` only ever routes an `SshTrainJobPayload`
        here (selected by `payload.training_target`), so this narrows via
        `isinstance` before touching `remote_server_id`.
        """
        if not isinstance(payload, SshTrainJobPayload):
            raise TypeError("SshTrainingTargetHandler.prepare requires an SshTrainJobPayload")
        if payload.base_model_id is not None:
            raise RemoteResumeUnsupportedError
        remote_server = await self.remote_server_service.get_remote_server(payload.remote_server_id)
        if remote_server.last_check_status != "healthy":
            raise RemoteServerNotReadyError(remote_server.name, remote_server.last_check_status)
        resolved = resolve_alias(get_settings().ssh_config_path, remote_server.ssh_host_alias)
        if not resolved.found:
            raise RemoteServerAliasNotFoundError(remote_server.name, remote_server.ssh_host_alias)
        return payload

    @staticmethod
    def target_key(payload: TrainJobPayload) -> str:
        if not isinstance(payload, SshTrainJobPayload):
            raise TypeError("SshTrainingTargetHandler.target_key requires an SshTrainJobPayload")
        return f"{TrainingTarget.SSH.value}:{payload.remote_server_id}"
