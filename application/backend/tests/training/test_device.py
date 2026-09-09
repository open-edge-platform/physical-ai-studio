# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Lightning accelerator/strategy/device resolution."""

from training.device import resolve_devices, resolve_strategy


class TestResolveStrategy:
    """Tests for resolve_strategy."""

    def test_non_xpu_uses_auto(self):
        """Test that non-XPU accelerators always resolve to 'auto'."""
        assert resolve_strategy("cpu") == "auto"
        assert resolve_strategy("cuda", [0, 1]) == "auto"

    def test_xpu_single_device_uses_xpu_single(self):
        """Test that a single XPU device resolves to 'xpu_single'."""
        assert resolve_strategy("xpu") == "xpu_single"
        assert resolve_strategy("xpu", 0) == "xpu_single"
        assert resolve_strategy("xpu", [0]) == "xpu_single"

    def test_xpu_multi_device_uses_xpu_ddp(self):
        """Test that more than one XPU device resolves to 'xpu_ddp'."""
        assert resolve_strategy("xpu", [0, 1]) == "xpu_ddp"


class TestResolveDevices:
    """Tests for resolve_devices."""

    def test_none_defaults_to_one_device(self):
        """Test that no index picks a single auto-selected device."""
        assert resolve_devices() == 1

    def test_int_index_wraps_in_list(self):
        """Test that a single index is wrapped in a list."""
        assert resolve_devices(2) == [2]

    def test_list_of_indices_passes_through(self):
        """Test that a list of indices is passed through unchanged."""
        assert resolve_devices([0, 1]) == [0, 1]
