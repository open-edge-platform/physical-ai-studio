# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local training target: runs in the Studio process on a local device."""

from __future__ import annotations

from exceptions import InvalidResourceError, UnsupportedDeviceError
from schemas.job import TrainingTarget, TrainJobPayload
from services.system_service import SystemService
from training.policy_catalog import available_policy_classes


class LocalTrainingTargetHandler:
    """Validates and keys jobs that train in the Studio process."""

    async def prepare(self, payload: TrainJobPayload) -> TrainJobPayload:
        """Reject unsupported local devices and policies before saving a job."""
        if payload.device is not None and not SystemService.is_device_supported_for_training(payload.device.type):
            raise UnsupportedDeviceError(
                device_type=payload.device.type,
                supported=SystemService.supported_training_device_types(),
            )
        if payload.policy not in available_policy_classes():
            raise InvalidResourceError("policy", f"{payload.policy!r} is not available on this machine")
        return payload

    @staticmethod
    def target_key(payload: TrainJobPayload) -> str:  # noqa: ARG004 - shared key, no per-job fields
        return TrainingTarget.LOCAL.value
