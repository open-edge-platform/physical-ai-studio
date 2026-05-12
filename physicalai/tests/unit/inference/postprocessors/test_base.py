# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Postprocessor ABC."""

from __future__ import annotations

from typing import override

import numpy as np
import pytest

from physicalai.inference.postprocessors import Postprocessor


class _ClampPostprocessor(Postprocessor):
    def __init__(self, low: float = -1.0, high: float = 1.0) -> None:
        self.low = low
        self.high = high

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs["action"] = np.clip(outputs["action"], self.low, self.high)
        return outputs

    @override
    def __repr__(self) -> str:
        return f"_ClampPostprocessor(low={self.low}, high={self.high})"


class _ScaleActionPostprocessor(Postprocessor):
    def __init__(self, factor: float = 0.5) -> None:
        self.factor = factor

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs["action"] = outputs["action"] * self.factor
        return outputs


class TestPostprocessorABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Postprocessor()  # type: ignore[abstract]

    def test_subclass_call(self) -> None:
        post = _ClampPostprocessor(low=-0.5, high=0.5)
        outputs = {"action": np.array([2.0, -3.0, 0.1])}
        result = post(outputs)
        np.testing.assert_array_equal(result["action"], np.array([0.5, -0.5, 0.1]))

    def test_repr_default(self) -> None:
        post = _ClampPostprocessor()
        assert repr(post) == "_ClampPostprocessor(low=-1.0, high=1.0)"

    def test_repr_from_base(self) -> None:
        post = _ScaleActionPostprocessor()
        assert repr(post) == "_ScaleActionPostprocessor()"

    def test_isinstance_check(self) -> None:
        post = _ClampPostprocessor()
        assert isinstance(post, Postprocessor)

    def test_chaining_preserves_order(self) -> None:
        chain: list[Postprocessor] = [
            _ScaleActionPostprocessor(factor=0.5),
            _ClampPostprocessor(low=-0.2, high=0.2),
        ]
        data: dict[str, np.ndarray] = {"action": np.array([1.0, -1.0])}
        for post in chain:
            data = post(data)
        # 1.0 * 0.5 = 0.5 → clamp to 0.2
        # -1.0 * 0.5 = -0.5 → clamp to -0.2
        np.testing.assert_array_equal(data["action"], np.array([0.2, -0.2]))

    def test_preserves_non_action_keys(self) -> None:
        post = _ClampPostprocessor()
        outputs = {"action": np.array([5.0]), "metadata": np.array([99.0])}
        result = post(outputs)
        np.testing.assert_array_equal(result["metadata"], np.array([99.0]))
