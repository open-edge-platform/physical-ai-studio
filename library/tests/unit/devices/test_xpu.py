# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - XPU Device"""

from unittest.mock import patch

import pytest
import torch
from lightning.pytorch.strategies import StrategyRegistry
from lightning.pytorch.utilities.exceptions import MisconfigurationException

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

    def test_get_parallel_devices_from_indexes(self):
        """Test get_parallel_devices maps integer indexes to xpu torch.device objects."""
        assert XPUAccelerator.get_parallel_devices([0, 1]) == [
            torch.device("xpu", 0),
            torch.device("xpu", 1),
        ]

    def test_get_parallel_devices_from_device_strings(self):
        """Test get_parallel_devices maps device strings to xpu torch.device objects."""
        assert XPUAccelerator.get_parallel_devices(["xpu:0", "xpu:1"]) == [
            torch.device("xpu", 0),
            torch.device("xpu", 1),
        ]

    def test_get_parallel_devices_from_torch_devices(self):
        """Test get_parallel_devices passes through torch.device objects."""
        assert XPUAccelerator.get_parallel_devices([torch.device("xpu", 0)]) == [torch.device("xpu", 0)]


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
        """Test that the multi-XPU strategy class is registered in Lightning.

        Uses the registry's internal mapping instead of ``StrategyRegistry.get``, which
        would instantiate the strategy and fail on this runner if XPU is unavailable.
        """
        assert XPUDDPStrategy.strategy_name in StrategyRegistry
        assert StrategyRegistry[XPUDDPStrategy.strategy_name]["strategy"] is XPUDDPStrategy

    def test_raises_when_xpu_unavailable(self):
        """Test that instantiating without XPU devices raises MisconfigurationException."""
        with patch("torch.xpu.is_available", return_value=False), pytest.raises(MisconfigurationException):
            XPUDDPStrategy()

    @pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU devices")
    def test_root_device_uses_parallel_devices(self):
        """Test that root_device follows parallel_devices/local_rank, not a bare rank->index mapping.

        Regression test: an earlier implementation hardcoded
        ``torch.device("xpu", self.local_rank)``, which silently trains on
        the wrong device whenever devices are non-contiguous (e.g. [2, 3]).
        """
        strategy = XPUDDPStrategy()
        strategy.parallel_devices = [torch.device("xpu", 2), torch.device("xpu", 3)]
        with patch.object(type(strategy), "local_rank", new=1):
            assert strategy.root_device == torch.device("xpu", 3)
