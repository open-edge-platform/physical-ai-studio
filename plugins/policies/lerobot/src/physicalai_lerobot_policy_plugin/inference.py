# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy Runtime adapter registration for native LeRobot policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.inference.adapters.registry import RuntimeAdapterRegistry


_RUNTIME_ADAPTER_MODULE = "physicalai_lerobot_policy_plugin.runtime"


def register(registry: RuntimeAdapterRegistry) -> None:
    """Register the LeRobot Runtime adapter without importing training code.

    Args:
        registry: Runtime's inference-adapter registry.
    """
    registry.register_lazy_module(
        "lerobot",
        _RUNTIME_ADAPTER_MODULE,
        extensions=(".lerobot",),
    )
