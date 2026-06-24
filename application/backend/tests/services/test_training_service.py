"""Unit tests for training/validation progress helpers and logging callbacks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from loguru import logger

from services.training_service import TrainingLogCallback, _extract_loss, _safe_progress


def _trainer(**kwargs) -> SimpleNamespace:
    defaults = {"global_step": 0, "max_steps": -1, "callback_metrics": {}}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class _Loss:
    """Minimal stand-in for a torch scalar tensor."""

    def __init__(self, value: float):
        self._value = value

    def detach(self) -> _Loss:
        return self

    def cpu(self) -> _Loss:
        return self

    def item(self) -> float:
        return self._value


@pytest.fixture
def loguru_messages():
    """Capture loguru log messages into a list for assertions."""
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


class TestSafeProgress:
    @pytest.mark.parametrize(
        ("global_step", "max_steps", "expected"),
        [
            (0, -1, 0),  # unset
            (5, 0, 0),  # zero divisor
            (0, 100, 0),
            (50, 100, 50),
            (100, 100, 100),
            (200, 100, 100),  # clamped high
        ],
    )
    def test_clamped(self, global_step, max_steps, expected):
        assert _safe_progress(global_step, max_steps) == expected


class TestExtractLoss:
    def test_from_mapping(self):
        assert _extract_loss({"loss": _Loss(0.5)}) == 0.5

    def test_from_bare_tensor(self):
        assert _extract_loss(_Loss(1.25)) == 1.25

    def test_none_and_unsupported(self):
        assert _extract_loss(None) is None
        assert _extract_loss({"no_loss": _Loss(1.0)}) is None
        assert _extract_loss(42) is None


class TestValidationLogging:
    def test_start_marker(self, loguru_messages):
        cb = TrainingLogCallback()
        cb.on_validation_start(_trainer(global_step=2400, max_steps=10000), None)
        assert any("Validation started at step=2400/10000" in m for m in loguru_messages)

    def test_first_batch_logs_with_loss(self, loguru_messages):
        cb = TrainingLogCallback()
        cb.every_n_steps = 10

        cb.on_validation_batch_end(_trainer(), None, _Loss(0.3), None, batch_idx=0)

        assert any("Validation progress: batch=1, val/loss_step=0.3" in m for m in loguru_messages)

    def test_respects_cadence(self, loguru_messages):
        cb = TrainingLogCallback()
        cb.every_n_steps = 10

        # batch_idx=4 -> current=5; not first and 5 % 10 != 0 -> skipped
        cb.on_validation_batch_end(_trainer(), None, _Loss(0.3), None, batch_idx=4)

        assert not loguru_messages

    def test_epoch_end_summary(self, loguru_messages):
        cb = TrainingLogCallback()
        cb.on_validation_start(_trainer(global_step=2400), None)
        trainer = _trainer(global_step=2400, callback_metrics={"val/loss": _Loss(0.123)})

        cb.on_validation_epoch_end(trainer, None)

        assert any("Validation finished at step=2400, val/loss=0.123" in m for m in loguru_messages)
