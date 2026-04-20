# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OVTokenizer preprocessor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.inference.constants import TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.preprocessors import Preprocessor


@pytest.fixture
def mock_adapter():
    """Create a mock OpenVINOAdapter."""
    adapter = MagicMock()
    adapter.input_names = ["string_input"]
    adapter.output_names = ["input_ids", "attention_mask"]

    def _predict(inputs):
        # Simulate tokenizer output based on batch size
        batch_size = len(inputs["string_input"])
        return {
            "input_ids": np.ones((batch_size, 64), dtype=np.int64),
            "attention_mask": np.ones((batch_size, 64), dtype=np.int64),
        }

    adapter.predict.side_effect = _predict
    return adapter


@pytest.fixture
def ov_tokenizer(mock_adapter):
    """Create an OVTokenizer with mocked adapter."""
    with patch(
        "physicalai.inference.preprocessors.ov_tokenizer.OpenVINOAdapter",
        return_value=mock_adapter,
    ):
        from physicalai.inference.preprocessors.ov_tokenizer import OVTokenizer

        return OVTokenizer(artifact="tokenizer.xml")


class TestOVTokenizerInit:
    """Tests for OVTokenizer initialization."""

    def test_is_preprocessor(self, ov_tokenizer) -> None:
        """OVTokenizer should be a Preprocessor instance."""
        assert isinstance(ov_tokenizer, Preprocessor)

    def test_loads_adapter_on_init(self, mock_adapter) -> None:
        """Adapter should be loaded with the artifact path during init."""
        with patch(
            "physicalai.inference.preprocessors.ov_tokenizer.OpenVINOAdapter",
            return_value=mock_adapter,
        ):
            from physicalai.inference.preprocessors.ov_tokenizer import OVTokenizer

            OVTokenizer(artifact=Path("my_tokenizer.xml"))

        mock_adapter.load.assert_called_once_with(Path("my_tokenizer.xml"))

    def test_raises_when_multiple_inputs(self) -> None:
        """Should raise ValueError if adapter has more than one input."""
        bad_adapter = MagicMock()
        bad_adapter.input_names = ["input1", "input2"]

        with patch(
            "physicalai.inference.preprocessors.ov_tokenizer.OpenVINOAdapter",
            return_value=bad_adapter,
        ):
            from physicalai.inference.preprocessors.ov_tokenizer import OVTokenizer

            with pytest.raises(ValueError, match="Expected exactly one input"):
                OVTokenizer(artifact="tokenizer.xml")


class TestOVTokenizerCall:
    """Tests for OVTokenizer __call__ method."""

    def test_tokenizes_batch(self, ov_tokenizer) -> None:
        """Should tokenize a batch of tasks and return correct keys."""
        inputs = {TASK: ["task one", "task two"]}
        result = ov_tokenizer(inputs)

        assert TOKENIZED_PROMPT in result
        assert TOKENIZED_PROMPT_MASK in result
        assert TASK not in result
        assert result[TOKENIZED_PROMPT].shape == (2, 64)
        assert result[TOKENIZED_PROMPT_MASK].dtype == np.bool_

    def test_raises_when_task_not_list(self, ov_tokenizer) -> None:
        """Should raise TypeError if TASK is not a list."""
        inputs = {TASK: "not a list"}

        with pytest.raises(TypeError, match="Expected TASK to be a list"):
            ov_tokenizer(inputs)
