# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local training target: runs in the Studio process on a local device."""

from __future__ import annotations

from exceptions import UnsupportedDeviceError
from schemas.job import TrainingTarget, TrainJobPayload
from services.system_service import SystemService


class LocalTrainingTargetHandler:
    """Validates and keys jobs that train in the Studio process."""

    async def prepare(self, payload: TrainJobPayload) -> TrainJobPayload:
        """Reject a local device SystemService reports as unsupported for training.

        A remote or SSH-provisioned trainer validates its own devices, so only
        local device choices are checked here. `device` is a shared field
        (declared on `TrainJobPayloadBase`), so no narrowing to
        `LocalTrainJobPayload` is needed to read it.
        """
        if payload.device is not None and not SystemService.is_device_supported_for_training(payload.device.type):
            raise UnsupportedDeviceError(
                device_type=payload.device.type,
                supported=SystemService.supported_training_device_types(),
            )
        return payload

    @staticmethod
    def target_key(payload: TrainJobPayload) -> str:  # noqa: ARG004 - shared key, no per-job fields
        return TrainingTarget.LOCAL.value
