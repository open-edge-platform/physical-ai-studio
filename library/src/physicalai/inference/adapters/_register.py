# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy registration of ``physicalai-train`` inference adapters.

This module is discovered through the ``physicalai.inference.adapters``
entry-point group and registers Torch and ExecuTorch as **lazy** backends
on the shared :data:`backend_registry`.  Importing this module is cheap —
it does not import torch or executorch.  The heavy adapter modules are
loaded only when a user actually requests one of those backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.inference.adapters.registry import InferenceBackendRegistry


def register(registry: InferenceBackendRegistry) -> None:
    """Register the Torch and ExecuTorch adapter modules as lazy backends.

    Args:
        registry: The shared inference backend registry to populate.
    """
    registry.register_lazy_module(
        "torch",
        "physicalai.inference.adapters.torch",
        extensions=(".ckpt", ".pt"),
    )
    registry.register_lazy_module(
        "executorch",
        "physicalai.inference.adapters.executorch",
        extensions=(".pte",),
    )
