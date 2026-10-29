# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Camera-byte and float-input parity for ACT export recipes."""

import numpy as np
import pytest
import torch
from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.preprocessors import Preprocessor

from physicalai.policies import ACT
from physicalai.policies.act.preprocessor import ACTPreprocessor


@pytest.mark.parametrize("keys", [("images",), ("images.overview", "images.wrist")])
@pytest.mark.parametrize("backend", ["onnx", "openvino", "executorch"])
@pytest.mark.parametrize("layout", ["BHWC", "BCHW"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_export_recipe_matches_native_resize_for_camera_inputs(backend, layout, dtype, keys):
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, size=(1, 37, 53, 3), dtype=np.uint8)
    native_input = torch.from_numpy(pixels.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
    image = pixels if dtype == np.uint8 else pixels.astype(np.float32) / 255.0
    if layout == "BCHW":
        image = image.transpose(0, 3, 1, 2)
    inputs = {key: image.copy() for key in keys}
    inputs["state"] = np.ones((1, 6), dtype=np.float32)
    expected = ACTPreprocessor(image_resolution=(32, 32))({"images": native_input})["images"].numpy()
    policy = ACT(image_size=(32, 32), pretrained_backbone_weights=None)
    specs = policy.extra_export_args[backend].preprocessors_specs
    actual = inputs
    for spec in specs:
        actual = instantiate_component(Preprocessor, spec)(actual)
    for key in keys:
        np.testing.assert_allclose(actual[key], expected, rtol=0, atol=5e-5)
    np.testing.assert_array_equal(actual["state"], inputs["state"])
    np.testing.assert_array_equal(inputs[keys[0]], image)
