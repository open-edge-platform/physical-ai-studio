# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training backends.

`TrainingBackend` abstracts where training runs. `LocalTrainingBackend` trains
in-process with torch/Lightning; `RemoteTrainingBackend` offloads to a trainer
service. The active backend is selected from settings by the training worker.
"""

from services.training_backends.base import (
    ProgressReporter,
    TrainingBackend,
    TrainingCanceledError,
    TrainingContext,
    TrainingSuspendedError,
)


def get_training_backend() -> TrainingBackend:
    """Return the training backend selected by application settings.

    Heavy imports are deferred to the chosen backend so a recording-only
    install (TRAINING_MODE=remote) never imports torch.
    """
    from settings import get_settings

    settings = get_settings()
    if settings.training_mode == "remote":
        from services.training_backends.remote import RemoteTrainingBackend

        return RemoteTrainingBackend()

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
