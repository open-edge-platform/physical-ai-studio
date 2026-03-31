# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for training callbacks."""

from unittest.mock import MagicMock

import lightning as L

from physicalai.train.callbacks import IterationTimer


class TestIterationTimer:
    """Tests for the IterationTimer callback."""

    def test_logs_iter_time_in_seconds(self):
        """Verify that iter time is logged in seconds."""
        callback = IterationTimer()
        trainer = MagicMock(spec=L.Trainer)
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
        callback.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        pl_module.log.assert_called_once()
        args, kwargs = pl_module.log.call_args
        assert args[0] == "train/iter_time_s"
        assert isinstance(args[1], float)
        assert args[1] >= 0
        assert kwargs["prog_bar"] is True

    def test_iter_time_reflects_elapsed_duration(self):
        """Verify that logged time reflects actual elapsed duration."""
        import time

        callback = IterationTimer()
        trainer = MagicMock(spec=L.Trainer)
        pl_module = MagicMock(spec=L.LightningModule)

        callback.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
        time.sleep(0.05)
        callback.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        logged_time = pl_module.log.call_args[0][1]
        assert logged_time >= 0.04  # allow small timing tolerance
        assert logged_time < 1.0  # sanity upper bound
