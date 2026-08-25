# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training backends selected by each submitted training job.

`TrainingBackend` abstracts where training runs. `LocalTrainingBackend` trains
in-process with torch/Lightning; `RemoteTrainingBackend` offloads to a trainer
service. The active backend is selected per job from its persisted execution
target.
"""

from schemas.job import RemoteTrainJobPayload, SshTrainJobPayload, TrainJobPayload
from services.training_backends.base import (
    ProgressReporter,
    TrainingBackend,
    TrainingCanceledError,
    TrainingContext,
    TrainingSuspendedError,
)


def get_training_backend(payload: TrainJobPayload) -> TrainingBackend:
    """Return the backend selected by a job's persisted execution target."""
    if isinstance(payload, SshTrainJobPayload):
        # SSH provisioning (PR7/PR8) is not wired in yet. JobService rejects SSH
        # submissions before a job reaches this factory (see submit_train_job),
        # so this should be unreachable; fail loudly rather than silently
        # falling through to local training if it ever is reached.
        raise NotImplementedError("SSH-provisioned training backend is not yet implemented")

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
