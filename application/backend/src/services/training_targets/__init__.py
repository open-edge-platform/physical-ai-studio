# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training targets selected by each submitted training job.

`TrainingTargetHandler` validates a job payload for one execution target
(local or direct-URL remote, which also covers an SSH-tunneled trainer - see
`services.ssh.persistent_trainer`) and computes the worker key that keeps two
jobs on the same target from running concurrently. `JobService.submit_train_job`
uses `get_training_target_handler(...).prepare` at submission time;
`TrainingWorker` uses `target_key` to serialize scheduling. Adding a target
means adding one handler class here, not another branch in each caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from schemas.job import TrainingTarget, TrainJobPayload
from services.training_targets.base import TrainingTargetHandler
from services.training_targets.local import LocalTrainingTargetHandler
from services.training_targets.remote import RemoteTrainingTargetHandler

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from services.remote_trainer_service import RemoteTrainerService

_KEY_HANDLERS: dict[TrainingTarget, type[TrainingTargetHandler]] = {
    TrainingTarget.LOCAL: LocalTrainingTargetHandler,
    TrainingTarget.REMOTE: RemoteTrainingTargetHandler,
}


def get_training_target_handler(
    payload: TrainJobPayload,
    session: AsyncSession,
    remote_trainer_service: RemoteTrainerService | None = None,
) -> TrainingTargetHandler:
    """Return the handler selected by a payload's execution target.

    Falls back to constructing a request-scoped service from `session` when
    the caller (typically `JobService`) was not given one.
    """
    if payload.training_target is TrainingTarget.LOCAL:
        return LocalTrainingTargetHandler()
    from services.remote_trainer_service import RemoteTrainerService as _RemoteTrainerService

    return RemoteTrainingTargetHandler(remote_trainer_service or _RemoteTrainerService(session))


def target_key(payload: TrainJobPayload) -> str:
    """Return the exclusive worker-scheduling key for a payload.

    Pure lookup on `payload`: no service instance or DB session is needed,
    so a worker can compute this for every pending job without touching the
    database.
    """
    return _KEY_HANDLERS[payload.training_target].target_key(payload)


__all__ = [
    "LocalTrainingTargetHandler",
    "RemoteTrainingTargetHandler",
    "TrainingTargetHandler",
    "get_training_target_handler",
    "target_key",
]
