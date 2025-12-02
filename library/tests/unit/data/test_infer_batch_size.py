# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test utility functions."""

import pytest
import torch

from getiaction.data.observation import Observation
from getiaction.data.utils import infer_batch_size


class TestInferBatchSize:
    """Tests for inferring batch size."""
    def test_infer_from_action(self):
        """Infers batch size using priority field: action."""
        obs = Observation(action=torch.randn(5, 3))
        assert infer_batch_size(obs) == 5

    def test_infer_from_state(self):
        """Infers batch size using priority field: state."""
        obs = Observation(state=torch.zeros(4, 2))
        assert infer_batch_size(obs) == 4

    def test_infer_from_image_nested(self):
        """Infers batch size from nested image tensor."""
        obs = Observation(images={"rgb": torch.zeros(3, 3, 10, 10)})
        assert infer_batch_size(obs) == 3

    def test_infer_from_dict_action(self):
        """Infers batch size when batch is a plain dict with action tensor."""
        batch = {"action": torch.randn(7, 2)}
        assert infer_batch_size(batch) == 7

    def test_infer_fallback_to_other_tensor(self):
        """Falls back to non-priority tensors if no action/state/image exists."""
        batch = {"meta": "ignore_me", "aux": torch.randn(6, 10)}
        assert infer_batch_size(batch) == 6

    def test_nested_dict_fallback(self):
        """Falls back to nested dictionary tensor when priority keys are absent."""
        batch = {"info": {"state": torch.zeros(2, 5)}}  # not a priority key
        assert infer_batch_size(batch) == 2

    def test_raises_if_no_tensor(self):
        """Raises ValueError when batch contains no tensors."""
        batch = {"info": "nothing", "meta": None}
        with pytest.raises(ValueError):
            infer_batch_size(batch)
