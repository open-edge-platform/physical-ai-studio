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
