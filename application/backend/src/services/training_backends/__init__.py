# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training backends selected by each submitted training job."""

from schemas.job import TrainJobPayload
from services.training_backends.base import (
    ProgressReporter,
    TrainingBackend,
    TrainingCanceledError,
    TrainingContext,
    TrainingSuspendedError,
)


def get_training_backend(payload: TrainJobPayload) -> TrainingBackend:
    """Return the backend selected by a job's persisted execution target."""
    from schemas.job import TrainingTarget

    if payload.training_target is TrainingTarget.REMOTE:
        from services.training_backends.remote import RemoteTrainingBackend

        if payload.remote_trainer_url is None:
            raise ValueError("Remote training job is missing its pinned trainer URL")
        return RemoteTrainingBackend(payload.remote_trainer_url)

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
