# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base postprocessor interface.

Postprocessors transform inference outputs *after* the runner returns.
They receive a dict wrapping the runner's output and must return a dict
of the same shape.

Postprocessors are domain-provided — ``physicalai.inference`` ships only
the ABC.  Concrete implementations (``ActionUnnormalizer``,
``ActionClamp``, …) belong in domain layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Postprocessor(ABC):
    """Abstract base class for inference postprocessors.

    A postprocessor transforms inference outputs after the runner
    returns.  Postprocessors run in declared order and receive a
    dict wrapping the runner's action output.

    Subclasses must implement ``__call__``.

    Examples:
        >>> class ClampAction(Postprocessor):
        ...     def __call__(self, outputs):
        ...         outputs["action"] = np.clip(outputs["action"], -1.0, 1.0)
        ...         return outputs
    """

    @abstractmethod
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Transform inference outputs after the runner.

        Args:
            outputs: Dict wrapping runner output.  The action array
                is stored under the ``"action"`` key.

        Returns:
            Transformed output dict.  Must preserve the ``"action"``
            key for downstream consumption.
        """

    def __repr__(self) -> str:
        """Return string representation of the postprocessor."""
        return f"{self.__class__.__name__}()"
