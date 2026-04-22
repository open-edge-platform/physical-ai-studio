# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.inference.constants import TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.preprocessors import Preprocessor


@pytest.fixture()
def mock_tokenizer():
    tok = MagicMock()
    tok.name_or_path = "mock-tokenizer"
    tok.config.revision = "main"

    def _encode(texts, **kwargs):
        max_length = kwargs.get("max_length", 512)
        batch_size = len(texts)
        result = MagicMock()
        result.__getitem__ = lambda self, key: {
            "input_ids": np.ones((batch_size, max_length), dtype=np.int64),
            "attention_mask": np.ones((batch_size, max_length), dtype=np.int64),
        }[key]
        return result

    tok.side_effect = _encode
    tok.__call__ = _encode
    return tok


@pytest.fixture()
def mock_transformers(mock_tokenizer):
    mock_mod = MagicMock()
    mock_mod.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    return mock_mod


@pytest.fixture()
def hf_tokenizer(mock_transformers):
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        from physicalai.inference.preprocessors.hf_tokenizer import HFTokenizer

        prep = HFTokenizer(tokenizer_name="mock-tokenizer", revision="main", max_token_len=64)
    return prep


class TestHFTokenizerInit:
    def test_is_preprocessor(self, hf_tokenizer) -> None:
        assert isinstance(hf_tokenizer, Preprocessor)

    def test_stores_max_token_len(self, hf_tokenizer) -> None:
        assert hf_tokenizer._max_token_len == 64

    def test_default_max_token_len(self, mock_transformers) -> None:
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            from physicalai.inference.preprocessors.hf_tokenizer import HFTokenizer

            prep = HFTokenizer(tokenizer_name="mock-tokenizer", revision="main")
        assert prep._max_token_len == 512

    def test_import_error_when_transformers_missing(self) -> None:
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                from physicalai.inference.preprocessors.hf_tokenizer import HFTokenizer

                HFTokenizer(tokenizer_name="mock-tokenizer", revision="main")

    def test_from_pretrained_called_with_args(self, mock_transformers) -> None:
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            from physicalai.inference.preprocessors.hf_tokenizer import HFTokenizer

            HFTokenizer(tokenizer_name="my-model", revision="v2", max_token_len=128)
            mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with("my-model", revision="v2", use_fast=True, padding_side="right")


class TestHFTokenizerCall:
    def test_tokenizes_single_task(self, hf_tokenizer) -> None:
        inputs = {TASK: ["pick up the cup"]}
        result = hf_tokenizer(inputs)

        assert TOKENIZED_PROMPT in result
        assert TOKENIZED_PROMPT_MASK in result
        assert TASK not in result
        assert result[TOKENIZED_PROMPT].shape == (1, 64)
        assert result[TOKENIZED_PROMPT_MASK].dtype == np.bool_

    def test_tokenizes_batch(self, hf_tokenizer) -> None:
        inputs = {TASK: ["task one", "task two", "task three"]}
        result = hf_tokenizer(inputs)

        assert result[TOKENIZED_PROMPT].shape[0] == 3
        assert result[TOKENIZED_PROMPT_MASK].shape[0] == 3

    def test_removes_task_key(self, hf_tokenizer) -> None:
        inputs = {TASK: ["hello"]}
        result = hf_tokenizer(inputs)
        assert TASK not in result

    def test_preserves_other_keys(self, hf_tokenizer) -> None:
        images = np.zeros((1, 3, 64, 64))
        inputs = {TASK: ["hello"], "images": images}
        result = hf_tokenizer(inputs)
        assert "images" in result
        np.testing.assert_array_equal(result["images"], images)

    def test_non_list_task_raises(self, hf_tokenizer) -> None:
        inputs = {TASK: "not a list"}
        with pytest.raises(TypeError, match="Expected TASK to be a list"):
            hf_tokenizer(inputs)

    def test_does_not_mutate_original(self, hf_tokenizer) -> None:
        inputs = {TASK: ["test"], "other": 42}
        original_keys = set(inputs.keys())
        hf_tokenizer(inputs)
        assert set(inputs.keys()) == original_keys


class TestHFTokenizerRepr:
    def test_repr_contains_class_name(self, hf_tokenizer) -> None:
        r = repr(hf_tokenizer)
        assert "HFTokenizer" in r

    def test_repr_contains_tokenizer_name(self, hf_tokenizer) -> None:
        r = repr(hf_tokenizer)
        assert "mock-tokenizer" in r
