# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cross-implementation consistency tests for the SmolVLA preprocessor.

The numpy ``ResizeSmolVLA`` preprocessor lives in core ``physicalai`` and
is exercised by tests under ``physicalai/tests/unit/inference/preprocessors``.
This file stays alongside ``physicalai-train`` because it pins the numpy
output against the reference torch ``SmolVLAPreprocessor`` implementation,
which lives in this distribution.
"""

from __future__ import annotations

import numpy as np
import torch

from physicalai.inference.preprocessors import ResizeSmolVLA
from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor


def test_consistency_vs_ref() -> None:
    input_batch = {
        "images": np.random.rand(1, 3, 640, 640).astype(np.float32),
        "state": np.random.rand(1, 2).astype(np.float32),
        "task": ["sample_task"],
    }

    input_batch_torch = {
        "images": torch.from_numpy(input_batch["images"]),
        "state": torch.from_numpy(input_batch["state"]),
        "task": input_batch["task"],
    }
    resolution = (512, 512)
    np_resize = ResizeSmolVLA(image_resolution=resolution)
    torch_resize = SmolVLAPreprocessor(image_resolution=resolution)

    np_result = np_resize(input_batch)
    torch_result = torch_resize(input_batch_torch)

    assert np.linalg.norm(np_result["images"] - torch_result["images"].numpy()) < 1e-1
