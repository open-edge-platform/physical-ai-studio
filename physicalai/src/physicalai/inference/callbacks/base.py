# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base callback interface for inference lifecycle hooks.

Callbacks provide a way to add cross-cutting concerns (timing,
logging, safety checks, telemetry) to the inference pipeline without
modifying model or runner code.

The ``Callback`` class is a **concrete** base class — not an ABC.
All hooks are no-ops by default.  Subclass and override only the
hooks you need.  This follows the Lightning callback pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel


class Callback:
    """Base callback class.  Override hooks as needed.

    Callbacks fire in declared order during the inference lifecycle:

    1. ``on_load`` — once, after the model finishes loading
    2. ``on_predict_start`` — before each prediction
    3. ``on_predict_end`` — after each prediction
    4. ``on_reset`` — when ``InferenceModel.reset()`` is called

    The ``on_predict_start`` and ``on_predict_end`` hooks can
    optionally modify the data flowing through the pipeline by
    returning a replacement dict.  Return ``None`` (the default) to
    pass through unchanged.

    Examples:
        >>> class PrintCallback(Callback):
        ...     def on_predict_start(self, inputs: dict) -> None:
        ...         print(f"Predicting with keys: {list(inputs.keys())}")

        >>> model = InferenceModel.load("./exports/act", callbacks=[PrintCallback()])
    """

    def on_predict_start(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        """Called before prediction.  Can modify inputs.

        Args:
            inputs: Observation dict about to enter the pipeline.

        Returns:
            A replacement dict to use instead of *inputs*, or
            ``None`` to pass *inputs* through unchanged.
        """

    def on_predict_end(self, outputs: dict[str, Any]) -> dict[str, Any] | None:
        """Called after prediction.  Can modify outputs.

        Args:
            outputs: Dict wrapping the runner's result (e.g.
                ``{"action": np.ndarray}``).

        Returns:
            A replacement dict to use instead of *outputs*, or
            ``None`` to pass *outputs* through unchanged.
        """

    def on_reset(self) -> None:
        """Called when model state is reset."""

    def on_load(self, model: InferenceModel) -> None:
        """Called once after the model finishes loading.

        Args:
            model: The fully initialised ``InferenceModel`` instance.
        """

    def __repr__(self) -> str:
        """Return string representation of the callback."""
        return f"{self.__class__.__name__}()"
