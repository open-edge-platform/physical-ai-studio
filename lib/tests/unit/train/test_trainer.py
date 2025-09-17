import pytest
from unittest.mock import MagicMock
from action_trainer.train.lightning import LightningActionTrainer

class TestLightningActionTrainer:
    """Simpler tests for LightningActionTrainer without testing the callback."""

    @pytest.fixture
    def dummy_datamodule(self):
        dm = MagicMock()
        dm.eval_dataset = MagicMock()
        return dm

    @pytest.fixture
    def dummy_model(self):
        return MagicMock()

    def test_fit_limits_val_batches_when_no_eval_dataset(self, dummy_model):
        """If datamodule has no eval_dataset, limit_val_batches is set to 0."""
        datamodule = MagicMock(eval_dataset=None)
        trainer_wrapper = LightningActionTrainer()
        trainer_wrapper.trainer.fit = MagicMock()

        trainer_wrapper.fit(model=dummy_model, datamodule=datamodule)

        assert trainer_wrapper.trainer.limit_val_batches == 0
        trainer_wrapper.trainer.fit.assert_called_once_with(model=dummy_model, datamodule=datamodule)

    def test_predict_validate_test_raise(self, dummy_model, dummy_datamodule):
        """predict, validate, and test methods raise NotImplementedError."""
        trainer_wrapper = LightningActionTrainer()
        for fn in (trainer_wrapper.predict, trainer_wrapper.validate, trainer_wrapper.test):
            with pytest.raises(NotImplementedError):
                fn()
