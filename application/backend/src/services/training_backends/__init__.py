# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training backends.

`TrainingBackend` abstracts where training runs. `LocalTrainingBackend` trains
in-process with torch/Lightning. The active backend is selected by the training
worker.
"""

from services.training_backends.base import ProgressReporter, TrainingBackend, TrainingCanceledError, TrainingContext


def get_training_backend() -> TrainingBackend:
    """Return the training backend used to run jobs.

    Heavy imports are deferred to the chosen backend so a recording-only install
    never imports torch at module load.
    """
    from services.training_backends.local import LocalTrainingBackend

    return LocalTrainingBackend()


__all__ = [
    "ProgressReporter",
    "TrainingBackend",
    "TrainingCanceledError",
    "TrainingContext",
    "get_training_backend",
]
