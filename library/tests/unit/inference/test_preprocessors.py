# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Preprocessor ABC."""

from __future__ import annotations

from typing import override

import numpy as np
import pytest

from physicalai.inference.preprocessors import Preprocessor


class _ScalePreprocessor(Preprocessor):
    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    @override
    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: v * self.factor if isinstance(v, np.ndarray) else v for k, v in inputs.items()}

    @override
    def __repr__(self) -> str:
        return f"_ScalePreprocessor(factor={self.factor})"


class _AddKeyPreprocessor(Preprocessor):
    @override
    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        inputs["added_key"] = np.array([1.0])
        return inputs


class TestPreprocessorABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Preprocessor()  # type: ignore[abstract]

    def test_subclass_call(self) -> None:
        pre = _ScalePreprocessor(factor=3.0)
        inputs = {"state": np.array([1.0, 2.0])}
        result = pre(inputs)
        np.testing.assert_array_equal(result["state"], np.array([3.0, 6.0]))

    def test_subclass_preserves_non_array_values(self) -> None:
        pre = _ScalePreprocessor()
        inputs: dict[str, np.ndarray] = {"state": np.array([1.0]), "metadata": "keep_me"}  # type: ignore[dict-item]
        result = pre(inputs)
        assert result["metadata"] == "keep_me"

    def test_subclass_can_add_keys(self) -> None:
        pre = _AddKeyPreprocessor()
        inputs = {"state": np.array([1.0])}
        result = pre(inputs)
        assert "added_key" in result
        np.testing.assert_array_equal(result["added_key"], np.array([1.0]))

    def test_repr_default(self) -> None:
        pre = _AddKeyPreprocessor()
        assert repr(pre) == "_AddKeyPreprocessor()"

    def test_repr_custom(self) -> None:
        pre = _ScalePreprocessor(factor=5.0)
        assert repr(pre) == "_ScalePreprocessor(factor=5.0)"

    def test_isinstance_check(self) -> None:
        pre = _ScalePreprocessor()
        assert isinstance(pre, Preprocessor)

    def test_chaining_preserves_order(self) -> None:
        chain: list[Preprocessor] = [_ScalePreprocessor(factor=2.0), _AddKeyPreprocessor()]
        data: dict[str, np.ndarray] = {"state": np.array([1.0])}
        for pre in chain:
            data = pre(data)
        np.testing.assert_array_equal(data["state"], np.array([2.0]))
        assert "added_key" in data
