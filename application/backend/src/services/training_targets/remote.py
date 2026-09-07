# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Remote (direct-URL) training target: offloads to a configured remote trainer."""

from __future__ import annotations

from exceptions import RemoteResumeUnsupportedError
from schemas.job import RemoteTrainJobPayload, TrainingTarget, TrainJobPayload
from services.remote_trainer_service import RemoteTrainerService


class RemoteTrainingTargetHandler:
    """Validates and keys jobs that offload to a directly-configured remote trainer."""

    def __init__(self, remote_trainer_service: RemoteTrainerService) -> None:
        self.remote_trainer_service = remote_trainer_service

    async def prepare(self, payload: TrainJobPayload) -> TrainJobPayload:
        """Resolve the configured trainer and pin its url/name onto the payload.

        Rejects resuming from a base model: the trainer protocol can receive a
        dataset but has no way to upload a base checkpoint.

        `get_training_target_handler` only ever routes a `RemoteTrainJobPayload`
        here (selected by `payload.training_target`), so this narrows via
        `isinstance` before touching `remote_trainer_id`.
        """
        if not isinstance(payload, RemoteTrainJobPayload):
            raise TypeError("RemoteTrainingTargetHandler.prepare requires a RemoteTrainJobPayload")
        if payload.base_model_id is not None:
            raise RemoteResumeUnsupportedError
        remote_trainer = await self.remote_trainer_service.get_remote_trainer(payload.remote_trainer_id)
        return payload.model_copy(
            update={"remote_trainer_url": str(remote_trainer.url), "remote_trainer_name": remote_trainer.name}
        )

    @staticmethod
    def target_key(payload: TrainJobPayload) -> str:
        if not isinstance(payload, RemoteTrainJobPayload):
            raise TypeError("RemoteTrainingTargetHandler.target_key requires a RemoteTrainJobPayload")
        return f"{TrainingTarget.REMOTE.value}:{payload.remote_trainer_id}"
