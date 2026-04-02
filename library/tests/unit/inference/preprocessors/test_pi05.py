# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import IMAGE_MASKS, IMAGES, TASK
from physicalai.inference.preprocessors import Preprocessor
from physicalai.inference.preprocessors.pi05 import Pi05Preprocessor


@pytest.fixture()
def preprocessor():
    return Pi05Preprocessor(image_resolution=(64, 64))


def _make_inputs(
    batch: int = 1,
    channels: int = 3,
    h: int = 64,
    w: int = 64,
    state_dim: int = 4,
    n_cameras: int = 1,
) -> dict[str, np.ndarray]:
    """Build a minimal Pi0.5 observation dict."""
    inputs: dict = {
        "observation.state": np.random.rand(batch, state_dim).astype(np.float32),
        TASK: ["pick up the cup"] * batch,
    }
    if n_cameras == 1:
        inputs[IMAGES] = np.random.rand(batch, channels, h, w).astype(np.float32)
    else:
        for i in range(n_cameras):
            inputs[f"{IMAGES}.{i}"] = np.random.rand(batch, channels, h, w).astype(np.float32)
    return inputs


class TestPi05PreprocessorInit:
    def test_is_preprocessor(self, preprocessor) -> None:
        assert isinstance(preprocessor, Preprocessor)

    def test_default_resolution(self) -> None:
        prep = Pi05Preprocessor()
        assert prep._image_resolution == (224, 224)

    def test_custom_resolution(self, preprocessor) -> None:
        assert preprocessor.image_resolution == (64, 64)


class TestPi05PreprocessorImages:
    def test_output_has_image_keys(self, preprocessor) -> None:
        inputs = _make_inputs()
        result = preprocessor(inputs)
        assert IMAGES in result
        assert IMAGE_MASKS in result

    def test_output_image_shape(self, preprocessor) -> None:
        inputs = _make_inputs(batch=2, h=32, w=32)
        result = preprocessor(inputs)
        # (n_cameras, batch, C, H, W)
        assert result[IMAGES].shape == (1, 2, 3, 64, 64)

    def test_pixel_range_ones(self, preprocessor) -> None:
        inputs = _make_inputs()
        inputs[IMAGES] = np.ones((1, 3, 64, 64), dtype=np.float32)
        result = preprocessor(inputs)
        np.testing.assert_allclose(result[IMAGES].max(), 1.0, atol=1e-5)

    def test_pixel_range_zeros(self, preprocessor) -> None:
        inputs = _make_inputs()
        inputs[IMAGES] = np.zeros((1, 3, 64, 64), dtype=np.float32)
        result = preprocessor(inputs)
        np.testing.assert_allclose(result[IMAGES].min(), -1.0, atol=1e-5)

    def test_masks_are_boolean_ones(self, preprocessor) -> None:
        inputs = _make_inputs(batch=2)
        result = preprocessor(inputs)
        assert result[IMAGE_MASKS].dtype == np.bool_
        assert result[IMAGE_MASKS].all()

    def test_multiple_cameras_stacked(self, preprocessor) -> None:
        inputs = _make_inputs(n_cameras=2)
        result = preprocessor(inputs)
        assert result[IMAGES].shape[0] == 2

    def test_non_square_image_padded(self, preprocessor) -> None:
        inputs = _make_inputs(h=32, w=64)
        result = preprocessor(inputs)
        assert result[IMAGES].shape[3] == 64
        assert result[IMAGES].shape[4] == 64

    def test_empty_cameras_appended(self) -> None:
        prep = Pi05Preprocessor(image_resolution=(64, 64), empty_cameras=1)
        inputs = _make_inputs()
        result = prep(inputs)
        # 1 real + 1 empty
        assert result[IMAGES].shape[0] == 2
        # empty camera mask should be zeros
        assert not result[IMAGE_MASKS][1].any()


class TestPi05PreprocessorText:
    def test_output_has_task_key(self, preprocessor) -> None:
        inputs = _make_inputs()
        result = preprocessor(inputs)
        assert TASK in result
        assert isinstance(result[TASK], list)

    def test_prompt_format(self, preprocessor) -> None:
        inputs = _make_inputs(batch=1, state_dim=2)
        inputs["observation.state"] = np.array([[0.0, 0.0]], dtype=np.float32)
        inputs[TASK] = ["test task"]

        result = preprocessor(inputs)

        prompt = result[TASK][0]
        assert prompt.startswith("Task: test task, State: ")
        assert prompt.endswith(";\nAction: ")

    def test_batch_prompt_count(self, preprocessor) -> None:
        inputs = _make_inputs(batch=3)
        result = preprocessor(inputs)
        assert len(result[TASK]) == 3

    def test_task_none_defaults_to_empty(self, preprocessor) -> None:
        inputs = _make_inputs()
        inputs.pop(TASK)
        result = preprocessor(inputs)
        assert TASK in result
        assert result[TASK][0].startswith("Task: , State: ")

    def test_task_single_string(self, preprocessor) -> None:
        inputs = _make_inputs()
        inputs[TASK] = "single task"
        result = preprocessor(inputs)
        assert "single task" in result[TASK][0]

    def test_underscores_replaced(self, preprocessor) -> None:
        inputs = _make_inputs(batch=1)
        inputs[TASK] = ["pick_up_the_cup"]

        result = preprocessor(inputs)

        prompt = result[TASK][0]
        assert "pick up the cup" in prompt
        assert "_" not in prompt


class TestPi05PreprocessorState:
    def test_discretization_range(self, preprocessor) -> None:
        inputs = _make_inputs(batch=1, state_dim=3)
        inputs["observation.state"] = np.array([[0.5, -0.5, 0.0]], dtype=np.float32)

        result = preprocessor(inputs)

        prompt = result[TASK][0]
        state_part = prompt.split("State: ")[1].split(";")[0]
        bins = [int(x) for x in state_part.split()]
        assert len(bins) == 3
        for b in bins:
            assert 0 <= b <= 255

    def test_3d_state_uses_last_timestep(self, preprocessor) -> None:
        inputs = _make_inputs(batch=1, state_dim=2)
        # (batch, timesteps, state_dim)
        inputs["observation.state"] = np.random.rand(1, 5, 2).astype(np.float32)

        result = preprocessor(inputs)

        prompt = result[TASK][0]
        state_part = prompt.split("State: ")[1].split(";")[0]
        bins = state_part.split()
        assert len(bins) == 2

    def test_custom_state_key(self) -> None:
        prep = Pi05Preprocessor(
            image_resolution=(64, 64),
            state_key="my_state",
        )
        inputs = {
            "my_state": np.random.rand(1, 4).astype(np.float32),
            TASK: ["task"],
            IMAGES: np.random.rand(1, 3, 64, 64).astype(np.float32),
        }
        result = prep(inputs)
        assert TASK in result


class TestPi05ResizeWithPad:
    def test_invalid_ndim_raises(self) -> None:
        from physicalai.inference.preprocessors.pi05 import Pi05Preprocessor

        with pytest.raises(ValueError, match="expected"):
            Pi05Preprocessor._resize_with_pad(np.zeros((3, 64, 64)), 64, 64)

    def test_exact_size_noop(self) -> None:
        from physicalai.inference.preprocessors.pi05 import Pi05Preprocessor

        img = np.random.rand(1, 64, 64, 3).astype(np.float32)
        result = Pi05Preprocessor._resize_with_pad(img, 64, 64)
        assert result.shape == (1, 64, 64, 3)

    def test_pads_to_target_size(self) -> None:
        from physicalai.inference.preprocessors.pi05 import Pi05Preprocessor

        img = np.random.rand(1, 32, 64, 3).astype(np.float32)
        result = Pi05Preprocessor._resize_with_pad(img, 64, 64)
        assert result.shape == (1, 64, 64, 3)

    def test_preserves_other_keys(self, preprocessor) -> None:
        inputs = _make_inputs()
        inputs["extra_key"] = "hello"
        result = preprocessor(inputs)
        assert result["extra_key"] == "hello"
