# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - XPU Device"""

import pytest
import torch
from lightning.pytorch.strategies import StrategyRegistry

from physicalai.devices.xpu import XPUAccelerator, SingleXPUStrategy, XPUDDPStrategy


class TestXPUAccelerator:
    """Unit tests for XPUAccelerator class."""

    def test_is_available_returns_true_when_xpu_available(self):
        """Test is_available returns True when XPU is available."""
        assert XPUAccelerator.is_available() == torch.xpu.is_available()

    def test_parse_devices_returns_device_indexes_for_integer_count(self):
        """Test parse_devices expands an integer count into XPU device indexes."""
        assert XPUAccelerator.parse_devices(2) == [0, 1]

    def test_parse_devices_rejects_non_positive_integer_count(self):
        """Test parse_devices rejects non-positive integer device counts."""
        with pytest.raises(ValueError, match="at least 1"):
            XPUAccelerator.parse_devices(0)


class TestSingleXPUStrategy:
    """Unit tests for SingleXPUStrategy class."""

    def test_strategy_name(self):
        """Test that the strategy name is correctly set."""
        assert SingleXPUStrategy.strategy_name == "xpu_single"


class TestXPUDDPStrategy:
    """Unit tests for XPUDDPStrategy class."""

    def test_strategy_name(self):
        """Test that the multi-XPU strategy name is correctly set."""
        assert XPUDDPStrategy.strategy_name == "xpu_ddp"

    def test_strategy_is_registered(self):
        """Test that the multi-XPU strategy is registered in Lightning."""
        strategy = StrategyRegistry.get(XPUDDPStrategy.strategy_name)

        assert isinstance(strategy, XPUDDPStrategy)
