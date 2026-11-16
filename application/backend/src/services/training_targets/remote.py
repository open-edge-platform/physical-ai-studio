# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Remote training target: offloads to a configured direct or managed trainer."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from botocore.exceptions import BotoCoreError, ClientError

from exceptions import InvalidResourceError, RemoteAllDevicesUnsupportedError, RemoteResumeUnsupportedError
from schemas.job import RemoteTrainJobPayload, TrainingTarget, TrainJobPayload
from schemas.remote_trainer import AwsBatchConnection
from services.aws_auth_service import AwsAuthenticationRequiredError, get_credential_provider
from services.aws_batch import _make_session, aws_batch_trainer_url
from services.remote_trainer_service import RemoteTrainerService

if TYPE_CHECKING:
    from uuid import UUID


class RemoteTrainingTargetHandler:
    """Validates and keys jobs that offload to a configured remote trainer."""

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
        is_batch = isinstance(remote_trainer.connection, AwsBatchConnection)
        if payload.device is not None and payload.device.all_devices:
            await self._require_all_devices_support(payload.remote_trainer_id, is_batch=is_batch)
        if isinstance(remote_trainer.connection, AwsBatchConnection):
            try:
                await asyncio.to_thread(_make_session, remote_trainer.connection)
            except (BotoCoreError, ClientError) as error:
                if issue := get_credential_provider(remote_trainer.connection).diagnose_error(error):
                    raise AwsAuthenticationRequiredError(issue) from error
                raise
        if remote_trainer.url is None and not is_batch:
            raise ValueError(f"Training on {remote_trainer.connection_mode.value} trainers is not available yet")
        catalog = await self.remote_trainer_service.get_policy_catalog(payload.remote_trainer_id)
        if payload.policy not in {policy.id for policy in catalog.available}:
            raise InvalidResourceError("policy", f"{payload.policy!r} is not available on the selected trainer")
        return payload.model_copy(
            update={
                "remote_trainer_url": (
                    aws_batch_trainer_url(remote_trainer.connection)
                    if isinstance(remote_trainer.connection, AwsBatchConnection)
                    else str(remote_trainer.url)
                ),
                "remote_trainer_name": remote_trainer.name,
                "authentication_issue": None,
                "cancel_requested": False,
                **({"device": None} if is_batch else {}),
            }
        )

    async def _require_all_devices_support(self, remote_trainer_id: UUID, *, is_batch: bool) -> None:
        """Reject an all-GPU job the trainer cannot run, before it is queued.

        Raises:
            RemoteAllDevicesUnsupportedError: On AWS Batch (which drops the
                device and would train on one GPU), when the trainer's image does
                not list the ``all_devices`` capability, or when its health
                cannot be confirmed right now.
        """
        if is_batch:
            raise RemoteAllDevicesUnsupportedError("Training on all GPUs is not available on AWS Batch trainers.")
        health = await self.remote_trainer_service.check_remote_trainer(remote_trainer_id)
        if health.status not in {"healthy", "degraded"}:
            raise RemoteAllDevicesUnsupportedError(
                "Can't confirm that this trainer supports training on all GPUs because it isn't ready."
            )
        if "all_devices" not in health.capabilities:
            raise RemoteAllDevicesUnsupportedError(
                "This trainer's image doesn't support training on all GPUs yet; update the trainer image."
            )
        if sum(1 for device in health.devices if device.type == "cuda") < 2:
            raise RemoteAllDevicesUnsupportedError("This trainer doesn't have several CUDA GPUs.")

    @staticmethod
    def target_key(payload: TrainJobPayload) -> str:
        if not isinstance(payload, RemoteTrainJobPayload):
            raise TypeError("RemoteTrainingTargetHandler.target_key requires a RemoteTrainJobPayload")
        trainer_key = f"{TrainingTarget.REMOTE.value}:{payload.remote_trainer_id}"
        if payload.remote_trainer_url and payload.remote_trainer_url.startswith("aws-batch://"):
            return f"batch:{payload.remote_trainer_id}:{payload.policy}"
        # The bare trainer key conflicts with every device key on that trainer (see
        # `TrainingJobHandler.resource_is_busy`), so an all-device job takes the whole trainer.
        if payload.device is None or payload.device.all_devices:
            return trainer_key
        return f"{trainer_key}:{payload.device.type}:{payload.device.index}"
