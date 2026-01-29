# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getiaction package - Embodied AI training and inference.

This package supports modular installation:
- Core only: `pip install getiaction` (lightweight, inference-ready)
- With backends: `pip install getiaction[onnx]` or `getiaction[openvino]`
- Full training: `pip install getiaction[train]`

Primary APIs:
    >>> from getiaction import InferenceModel, Trainer
    >>> policy = InferenceModel.load("./exports/act_policy")
    >>> trainer = Trainer(policy, datamodule)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from getiaction.inference import InferenceModel

if TYPE_CHECKING:
    from getiaction.train import Trainer

__version__ = "0.1.0"

__all__ = [
    "InferenceModel",
    "Trainer",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name == "Trainer":
        from getiaction.train import Trainer  # noqa: PLC0415

        return Trainer

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return __all__
