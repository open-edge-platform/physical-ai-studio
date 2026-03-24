# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base preprocessor interface.

Preprocessors transform observation dicts *before* the adapter bridge
(``_prepare_inputs``) flattens and filters them.  They operate on the
user-friendly nested observation structure and must return a dict of
the same shape.

Preprocessors are domain-provided — ``physicalai.inference`` ships only
the ABC.  Concrete implementations (``ObservationNormalizer``,
``ImageResize``, …) belong in domain layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Preprocessor(ABC):
    """Abstract base class for inference preprocessors.

    A preprocessor transforms an observation dict before inference.
    Preprocessors run in declared order *before*
    ``InferenceModel._prepare_inputs`` flattens/filters the dict for
    the runtime adapter.

    Subclasses must implement ``__call__``.

    Examples:
        >>> class Normalize(Preprocessor):
        ...     def __call__(self, inputs):
        ...         inputs["observation.state"] = inputs["observation.state"] / 255.0
        ...         return inputs
    """

    @abstractmethod
    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Transform observation inputs before inference.

        Args:
            inputs: Observation dict mapping names to numpy arrays.
                May contain nested dicts (e.g. ``{"obs": {"image": …}}``).

        Returns:
            Transformed observation dict.  Must preserve the key
            structure expected by downstream preprocessors and
            ``_prepare_inputs``.
        """

    def __repr__(self) -> str:
        """Return string representation of the preprocessor."""
        return f"{self.__class__.__name__}()"
