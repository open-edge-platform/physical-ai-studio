# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - XPU Device"""


from getiaction.devices.xpu import XPUAccelerator, SingleXPUStrategy
from getiaction.devices import is_xpu_available


class TestXPUAccelerator:
    """Unit tests for XPUAccelerator class."""

    def test_is_available_returns_true_when_xpu_available(self):
        """Test is_available returns True when XPU is available."""
        assert XPUAccelerator.is_available() == is_xpu_available()


class TestSingleXPUStrategy:
    """Unit tests for SingleXPUStrategy class."""

    def test_strategy_name(self):
        """Test that the strategy name is correctly set."""
        assert SingleXPUStrategy.strategy_name == "xpu_single"
