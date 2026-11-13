# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Lightning device resolution shared by every training runner."""

from __future__ import annotations

from training.device import resolve_devices


class TestResolveDevices:
    def test_an_index_selects_that_single_device(self) -> None:
        assert resolve_devices(2) == [2]

    def test_no_index_lets_lightning_pick_one_device(self) -> None:
        assert resolve_devices() == 1

    def test_all_devices_selects_every_visible_device(self) -> None:
        assert resolve_devices(all_devices=True) == -1

    def test_all_devices_takes_precedence_over_an_index(self) -> None:
        assert resolve_devices(0, all_devices=True) == -1
