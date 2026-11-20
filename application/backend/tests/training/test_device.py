# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Lightning device resolution shared by every training runner."""

from __future__ import annotations

from training.device import MULTI_DEVICE_STRATEGY, resolve_devices, resolve_strategy


class TestResolveDevices:
    def test_an_index_selects_that_single_device(self) -> None:
        assert resolve_devices(2) == [2]

    def test_no_index_lets_lightning_pick_one_device(self) -> None:
        assert resolve_devices() == 1

    def test_all_devices_selects_every_visible_device(self) -> None:
        assert resolve_devices(all_devices=True) == -1

    def test_all_devices_takes_precedence_over_an_index(self) -> None:
        assert resolve_devices(0, all_devices=True) == -1


class TestResolveStrategy:
    def test_an_all_device_run_tolerates_parameters_without_gradients(self) -> None:
        """SmolVLA and Pi0.5 freeze parts of the model; plain DDP fails on their first step."""
        assert resolve_strategy("cuda", all_devices=True) == MULTI_DEVICE_STRATEGY == "ddp_find_unused_parameters_true"

    def test_a_single_device_run_keeps_the_automatic_strategy(self) -> None:
        assert resolve_strategy("cuda") == "auto"
        assert resolve_strategy("xpu") == "xpu_single"
