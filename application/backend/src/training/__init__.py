# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""One training run, described by a spec and executed by one function.

Shared by the local training backend (in-process) and the trainer service
(remote), so a policy trains identically from either path. See :mod:`training.job` for
the full contract.
"""

from .device import resolve_accelerator, resolve_devices, resolve_strategy
from .job import TrainingJobSpec, run_training_job

__all__ = [
    "TrainingJobSpec",
    "resolve_accelerator",
    "resolve_devices",
    "resolve_strategy",
    "run_training_job",
]
