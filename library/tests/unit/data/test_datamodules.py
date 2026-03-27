# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DataModule."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestResolveAutoNumWorkers:
    """Tests for the resolve_auto_num_workers helper."""

    def test_returns_cpu_count_when_below_cap(self):
        from physicalai.data.datamodules import resolve_auto_num_workers

        with patch("physicalai.data.datamodules.os.cpu_count", return_value=4):
            assert resolve_auto_num_workers() == 4

    def test_caps_at_max_workers(self):
        from physicalai.data.datamodules import _AUTO_NUM_WORKERS_CAP, resolve_auto_num_workers

        with patch("physicalai.data.datamodules.os.cpu_count", return_value=64):
            assert resolve_auto_num_workers() == _AUTO_NUM_WORKERS_CAP

    def test_returns_zero_when_cpu_count_is_none(self):
        from physicalai.data.datamodules import resolve_auto_num_workers

        with patch("physicalai.data.datamodules.os.cpu_count", return_value=None):
            assert resolve_auto_num_workers() == 0


class TestDataModuleNumWorkers:
    """Tests for the DataModule num_workers parameter."""

    def test_auto_resolves_num_workers(self, dummy_dataset):
        from physicalai.data import DataModule

        dm = DataModule(train_dataset=dummy_dataset(), num_workers="auto")
        assert isinstance(dm.num_workers, int)
        assert dm.num_workers >= 0

    def test_explicit_num_workers(self, dummy_dataset):
        from physicalai.data import DataModule

        dm = DataModule(train_dataset=dummy_dataset(), num_workers=2)
        assert dm.num_workers == 2

    def test_default_is_auto(self, dummy_dataset):
        from physicalai.data import DataModule

        with patch("physicalai.data.datamodules.resolve_auto_num_workers", return_value=6) as mock:
            dm = DataModule(train_dataset=dummy_dataset())
            mock.assert_called_once()
            assert dm.num_workers == 6


class TestDataModuleBatchSizeAlias:
    """Tests for the batch_size property alias (Tuner compatibility)."""

    def test_batch_size_reads_train_batch_size(self, dummy_dataset):
        from physicalai.data import DataModule

        dm = DataModule(train_dataset=dummy_dataset(), train_batch_size=32)
        assert dm.batch_size == 32

    def test_batch_size_writes_train_batch_size(self, dummy_dataset):
        from physicalai.data import DataModule

        dm = DataModule(train_dataset=dummy_dataset(), train_batch_size=16)
        dm.batch_size = 64
        assert dm.train_batch_size == 64
        assert dm.batch_size == 64


class TestDataModuleValidation:
    """Tests for DataModule validation functionality."""

    def test_collate_gym(self):
        from physicalai.data.datamodules import _collate_gym
        from physicalai.gyms import Gym, PushTGym

        gym = PushTGym()
        batch = [gym]

        result = _collate_gym(batch)

        assert isinstance(result, Gym)
        assert result is gym

    def test_val_dataloader_structure(self, dummy_datamodule):
        from physicalai.gyms import Gym

        dummy_datamodule.setup(stage="fit")
        val_loader = dummy_datamodule.val_dataloader()

        assert len(val_loader) == 2

        batch = next(iter(val_loader))

        assert isinstance(batch, Gym)


class TestDataModuleTrainDataloader:
    """Tests for DataModule.train_dataloader num_workers."""

    def test_train_dataloader_uses_configured_num_workers(self, dummy_dataset):
        from physicalai.data import DataModule

        dm = DataModule(train_dataset=dummy_dataset(), num_workers=3)
        dl = dm.train_dataloader()
        assert dl.num_workers == 3


class TestDataModuleLogging:
    """Tests for DataModule num_workers logging."""

    def test_logs_auto_num_workers(self, dummy_dataset, caplog):
        import logging

        from physicalai.data import DataModule

        with caplog.at_level(logging.INFO, logger="physicalai.data.datamodules"):
            with patch("physicalai.data.datamodules.resolve_auto_num_workers", return_value=6):
                DataModule(train_dataset=dummy_dataset())

        assert "DataLoader workers: 6 (auto)" in caplog.text

    def test_logs_explicit_num_workers(self, dummy_dataset, caplog):
        import logging

        from physicalai.data import DataModule

        with caplog.at_level(logging.INFO, logger="physicalai.data.datamodules"):
            DataModule(train_dataset=dummy_dataset(), num_workers=2)

        assert "DataLoader workers: 2" in caplog.text
        assert "(auto)" not in caplog.text
