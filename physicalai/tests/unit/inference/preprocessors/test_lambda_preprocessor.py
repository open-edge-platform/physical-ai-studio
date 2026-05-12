# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from physicalai.inference.preprocessors import LambdaPreprocessor, Preprocessor


class TestLambdaPreprocessorInit:
    def test_is_preprocessor(self) -> None:
        prep = LambdaPreprocessor(fn=lambda x: x)
        assert isinstance(prep, Preprocessor)

    def test_stores_callable(self) -> None:
        fn = lambda x: x  # noqa: E731
        prep = LambdaPreprocessor(fn=fn)
        assert prep._fn is fn


class TestLambdaPreprocessorCall:
    def test_identity(self) -> None:
        prep = LambdaPreprocessor(fn=lambda x: x)
        inputs = {"image": np.array([1.0, 2.0, 3.0])}
        result = prep(inputs)
        np.testing.assert_array_equal(result["image"], inputs["image"])

    def test_transforms_values(self) -> None:
        prep = LambdaPreprocessor(fn=lambda x: {k: v * 2 for k, v in x.items()})
        inputs = {"obs": np.array([1.0, 2.0])}
        result = prep(inputs)
        np.testing.assert_array_equal(result["obs"], np.array([2.0, 4.0]))

    def test_normalise_to_unit_range(self) -> None:
        prep = LambdaPreprocessor(fn=lambda x: {k: v / 255.0 for k, v in x.items()})
        inputs = {"image": np.array([0.0, 127.5, 255.0])}
        result = prep(inputs)
        np.testing.assert_allclose(result["image"], np.array([0.0, 0.5, 1.0]))

    def test_multiple_keys(self) -> None:
        prep = LambdaPreprocessor(fn=lambda x: {k: v + 1 for k, v in x.items()})
        inputs = {"a": np.array([0.0]), "b": np.array([10.0])}
        result = prep(inputs)
        np.testing.assert_array_equal(result["a"], np.array([1.0]))
        np.testing.assert_array_equal(result["b"], np.array([11.0]))

    def test_empty_inputs(self) -> None:
        prep = LambdaPreprocessor(fn=lambda x: x)
        result = prep({})
        assert result == {}


class TestLambdaPreprocessorRepr:
    def test_repr_contains_class_name(self) -> None:
        prep = LambdaPreprocessor(fn=lambda x: x)
        assert "LambdaPreprocessor" in repr(prep)
