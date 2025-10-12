import pytest
from unittest.mock import MagicMock
from getiaction.train.trainer import Trainer

class TestLightningActionTrainer:
    """Tests for LightningActionTrainer without testing the callback."""

    @pytest.fixture
    def dummy_datamodule(self):
        dm = MagicMock()
        dm._val_dataset = MagicMock()
        return dm

    @pytest.fixture
    def dummy_model(self):
        return MagicMock()

    def test_fit_limits_val_batches_when_no__val_dataset(self, dummy_model):
        """If datamodule has no _val_dataset, limit_val_batches is set to 0."""
        datamodule = MagicMock(_val_dataset=None)
        trainer_wrapper = Trainer()
        trainer_wrapper.backend.fit = MagicMock()

        trainer_wrapper.fit(model=dummy_model, datamodule=datamodule)

        assert trainer_wrapper.backend.limit_val_batches == 0
        trainer_wrapper.backend.fit.assert_called_once_with(model=dummy_model, datamodule=datamodule)

    def test_predict_validate_test_raise(self, dummy_model, dummy_datamodule):
        """predict, validate, and test methods raise NotImplementedError."""
        trainer_wrapper = Trainer()
        for fn in (trainer_wrapper.predict, trainer_wrapper.validate, trainer_wrapper.test):
            with pytest.raises(NotImplementedError):
                fn()
