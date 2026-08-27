# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training backends selected by each submitted training job.

`TrainingBackend` abstracts where training runs. `LocalTrainingBackend` trains
in-process with torch/Lightning; `RemoteTrainingBackend` offloads to a trainer
service. The active backend is selected per job from its persisted execution
target.
"""

from uuid import UUID

from schemas.job import RemoteTrainJobPayload, TrainJobPayload
from services.training_backends.base import (
    ProgressReporter,
    TrainingBackend,
    TrainingCanceledError,
    TrainingContext,
    TrainingSuspendedError,
)


async def get_training_backend(payload: TrainJobPayload, job_id: UUID) -> TrainingBackend:
    """Return the backend selected by a job's persisted execution target.

    ``job_id`` is only used by the SSH target, to key its provisioning record;
    the local and direct-URL remote targets ignore it.
    """
    from schemas.job import TrainingTarget

    if payload.training_target is TrainingTarget.SSH:
        from db import get_async_db_session_ctx
        from services.remote_server_service import RemoteServerService
        from services.training_backends.ssh import SshTrainingBackend

        if payload.remote_server_id is None:
            raise ValueError("SSH training job is missing its selected remote server")
        async with get_async_db_session_ctx() as session:
            server = await RemoteServerService(session).get_remote_server(payload.remote_server_id)
        return SshTrainingBackend(job_id, server)

    if isinstance(payload, RemoteTrainJobPayload):
        from services.training_backends.remote import RemoteTrainingBackend

        if payload.remote_trainer_url is None:
            raise ValueError("Remote training job is missing its pinned trainer URL")
        return RemoteTrainingBackend(payload.remote_trainer_url, trainer_name=payload.remote_trainer_name)

    from services.training_backends.local import LocalTrainingBackend

    return LocalTrainingBackend()


__all__ = [
    "ProgressReporter",
    "TrainingBackend",
    "TrainingCanceledError",
    "TrainingContext",
    "TrainingSuspendedError",
    "get_training_backend",
]
