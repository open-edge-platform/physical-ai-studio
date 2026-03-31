# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lambda preprocessor that wraps an arbitrary callable."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.inference.preprocessors.base import Preprocessor

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


class LambdaPreprocessor(Preprocessor):
    """Preprocessor that delegates to a user-supplied callable.

    Args:
        fn: A callable that accepts and returns an observation dict.

    Examples:
        >>> prep = LambdaPreprocessor(lambda inputs: {k: v / 255 for k, v in inputs.items()})
        >>> outputs = prep({"image": np.ones((3, 224, 224))})
    """

    def __init__(self, fn: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]) -> None:
        """Initialize the LambdaPreprocessor.

        Args:
            fn: A callable that takes a dict of numpy arrays as input and returns a dict of numpy arrays as output.
        """
        super().__init__()
        self._fn = fn

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the wrapped callable to the inputs.

        Args:
            inputs: Observation dict mapping names to numpy arrays.

        Returns:
            Transformed observation dict.
        """
        return self._fn(inputs)

    def __repr__(self) -> str:
        """Return string representation of the preprocessor."""
        return f"{self.__class__.__name__}(fn={self._fn!r})"
