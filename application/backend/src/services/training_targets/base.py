# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training target contract shared by local, remote, and SSH implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from schemas.job import TrainJobPayload


@runtime_checkable
class TrainingTargetHandler(Protocol):
    """Validates and keys a job payload for one training execution target.

    `JobService.submit_train_job` calls `prepare` once, at submission time, to
    reject an invalid target selection and pin any target-specific fields
    (e.g. a resolved remote trainer URL) onto the payload. `TrainingWorker`
    calls `target_key` for every pending job to derive the key that keeps two
    jobs on the same target (the same remote trainer, the same SSH server)
    from running concurrently. Adding a target (e.g. a future
    AWS-provisioned trainer) means adding one handler class, not another
    branch in each caller.

    Each concrete handler is only ever invoked with the payload variant
    matching its target (`get_training_target_handler` routes on
    `payload.training_target`), so it narrows via `isinstance` internally
    before touching a target-specific field. The signature stays the full
    `TrainJobPayload` union so every handler is substitutable for this
    Protocol regardless of which variant it actually handles.
    """

    async def prepare(self, payload: TrainJobPayload) -> TrainJobPayload:
        """Validate `payload` for this target and return it, pinning target-specific fields."""
        ...

    @staticmethod
    def target_key(payload: TrainJobPayload) -> str:
        """Return the exclusive worker-scheduling key for a payload of this target."""
        ...
