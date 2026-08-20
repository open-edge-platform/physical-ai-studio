# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import TYPE_CHECKING

from schemas import InferenceDevice, Model

if TYPE_CHECKING:
    from physicalai.inference import InferenceModel


def load_inference_model(model: Model, inference_device: InferenceDevice) -> "InferenceModel":
    """Loads inference model."""
    from physicalai.inference import InferenceModel

    backend = inference_device.backend.value
    export_dir = Path(model.path) / "exports" / backend
    return InferenceModel(
        export_dir=export_dir,
        policy_name=model.policy,
        backend=backend,
        device=inference_device.device,
    )
