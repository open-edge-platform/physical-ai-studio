# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy Runtime adapter registration for GR00T packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.inference.adapters.registry import RuntimeAdapterRegistry

_RUNTIME_ADAPTER_MODULE = "physicalai_gr00t_policy_plugin.runtime"


def register(registry: RuntimeAdapterRegistry) -> None:
    """Register the GR00T Runtime adapter without importing it.

    Args:
        registry: Runtime's inference-adapter registry.
    """
    registry.register_lazy_module("gr00t", _RUNTIME_ADAPTER_MODULE, extensions=(".gr00t",))
