from unittest.mock import MagicMock, patch

import pytest
from physicalai.train.trainer import Trainer


class TestTrainer:
    """Tests for physicalai.train.Trainer (Lightning Trainer subclass)."""

    def test_trainer_is_lightning_subclass(self):
        """Verify Trainer is a subclass of Lightning Trainer."""
        import lightning

        assert issubclass(Trainer, lightning.Trainer)

    def test_trainer_defaults(self):
        """Verify physicalai-specific defaults are set."""
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)

        # physicalai default: num_sanity_val_steps=0 (instead of Lightning's 2)
        assert trainer.num_sanity_val_steps == 0

    def test_policy_dataset_interaction_callback_injected(self):
        """Verify PolicyDatasetInteraction callback is automatically added."""
        from physicalai.train.callbacks import PolicyDatasetInteraction

        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)

        # Check that PolicyDatasetInteraction callback was auto-injected
        callback_types = [type(cb) for cb in trainer.callbacks]
        assert PolicyDatasetInteraction in callback_types

    def test_user_callbacks_preserved(self):
        """Verify user callbacks are preserved alongside auto-injected callback."""
        from lightning.pytorch.callbacks import EarlyStopping
        from physicalai.train.callbacks import PolicyDatasetInteraction

        user_callback = EarlyStopping(monitor="val_loss")
        trainer = Trainer(
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            callbacks=[user_callback],
        )

        # Both user callback and auto-injected callback should be present
        callback_types = [type(cb) for cb in trainer.callbacks]
        assert EarlyStopping in callback_types
        assert PolicyDatasetInteraction in callback_types


class TestAutoScaleBatchSize:
    """Tests for the auto_scale_batch_size feature."""

    def test_default_is_disabled(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
        assert trainer._auto_scale_batch_size is False

    def test_stores_true(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, auto_scale_batch_size=True)
        assert trainer._auto_scale_batch_size is True

    def test_stores_binsearch(self):
        trainer = Trainer(
            accelerator="cpu", logger=False, enable_checkpointing=False, auto_scale_batch_size="binsearch"
        )
        assert trainer._auto_scale_batch_size == "binsearch"

    def test_stores_power(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, auto_scale_batch_size="power")
        assert trainer._auto_scale_batch_size == "power"

    @patch("physicalai.train.trainer.logger")
    def test_fit_calls_tuner_when_enabled(self, mock_logger, dummy_dataset, dummy_policy):
        trainer = Trainer(
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            max_epochs=1,
            auto_scale_batch_size=True,
        )

        mock_tuner_instance = MagicMock()

        with patch("lightning.pytorch.tuner.tuning.Tuner", return_value=mock_tuner_instance) as mock_tuner_cls:
            from physicalai.data import DataModule

            dm = DataModule(train_dataset=dummy_dataset(), train_batch_size=4)

            with patch.object(type(trainer).__bases__[0], "fit"):
                trainer.fit(model=dummy_policy, datamodule=dm)

            mock_tuner_cls.assert_called_once_with(trainer)
            mock_tuner_instance.scale_batch_size.assert_called_once_with(dummy_policy, datamodule=dm, mode="power")

    def test_fit_skips_tuner_when_disabled(self, dummy_dataset, dummy_policy):
        trainer = Trainer(
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            max_epochs=1,
            auto_scale_batch_size=False,
        )

        with patch("lightning.pytorch.tuner.tuning.Tuner") as mock_tuner_cls:
            from physicalai.data import DataModule

            dm = DataModule(train_dataset=dummy_dataset(), train_batch_size=4)

            with patch.object(type(trainer).__bases__[0], "fit"):
                trainer.fit(model=dummy_policy, datamodule=dm)

            mock_tuner_cls.assert_not_called()

    @patch("physicalai.train.trainer.logger")
    def test_fit_uses_binsearch_mode(self, mock_logger, dummy_dataset, dummy_policy):
        trainer = Trainer(
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            max_epochs=1,
            auto_scale_batch_size="binsearch",
        )

        mock_tuner_instance = MagicMock()

        with patch("lightning.pytorch.tuner.tuning.Tuner", return_value=mock_tuner_instance):
            from physicalai.data import DataModule

            dm = DataModule(train_dataset=dummy_dataset(), train_batch_size=4)

            with patch.object(type(trainer).__bases__[0], "fit"):
                trainer.fit(model=dummy_policy, datamodule=dm)

            mock_tuner_instance.scale_batch_size.assert_called_once_with(dummy_policy, datamodule=dm, mode="binsearch")
