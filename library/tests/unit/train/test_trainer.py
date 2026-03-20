from lightning.pytorch.callbacks import BatchSizeFinder
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

    def _has_batch_size_finder(self, trainer: Trainer) -> bool:
        return any(isinstance(cb, BatchSizeFinder) for cb in trainer.callbacks)

    def _get_batch_size_finder(self, trainer: Trainer) -> BatchSizeFinder:
        for cb in trainer.callbacks:
            if isinstance(cb, BatchSizeFinder):
                return cb
        raise AssertionError("BatchSizeFinder not found in callbacks")

    def test_disabled_by_default(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
        assert not self._has_batch_size_finder(trainer)

    def test_enabled_adds_callback(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, auto_scale_batch_size=True)
        assert self._has_batch_size_finder(trainer)

    def test_uses_power_mode(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, auto_scale_batch_size=True)
        finder = self._get_batch_size_finder(trainer)
        assert finder._mode == "power"

    def test_disabled_no_callback(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, auto_scale_batch_size=False)
        assert not self._has_batch_size_finder(trainer)
