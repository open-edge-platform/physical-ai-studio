# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Progress, cancellation, and cleanup tests for native LeRobot training."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

import pytest

from physicalai.engine import Engine, TrainingCancelledError


class _Hooks:
    def __init__(self, stop_after: int | None = None) -> None:
        self.stop_after = stop_after
        self.steps: list[tuple[int, int | None, Mapping[str, float]]] = []
        self._stop = False

    def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None:
        self.steps.append((step, total, dict(metrics)))
        if self.stop_after == step:
            self._stop = True

    def should_stop(self) -> bool:
        return self._stop


def _config() -> Any:
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig

    return TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/pusht"), steps=2)


def _tracker(initial_step: int, loss: float) -> Any:
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    tracker = MetricsTracker(1, 4, 1, {"loss": AverageMeter("loss")}, initial_step=initial_step)
    tracker.loss = loss
    return tracker


def _patch_native_training(monkeypatch: pytest.MonkeyPatch, train_body: Any) -> tuple[Any, Any]:
    from lerobot.scripts import lerobot_train
    from lerobot.utils import import_utils

    native_update = lerobot_train.update_policy
    monkeypatch.setattr(import_utils, "register_third_party_plugins", lambda: None)
    monkeypatch.setattr(lerobot_train, "train", train_body)
    return lerobot_train, native_update


def test_engine_reports_native_steps_and_restores_globals_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.scripts import lerobot_train

    hooks = _Hooks()
    trackers = [_tracker(0, 2.5), _tracker(1, 1.75)]
    native_results: list[object] = []
    seen_argv: list[list[str]] = []

    def native_update(tracker: Any) -> tuple[Any, dict[str, float]]:
        return tracker, {"native": 1.0}

    def native_train(config: Any) -> None:
        assert config is not None
        seen_argv.append(sys.argv.copy())
        native_results.extend(lerobot_train.update_policy(tracker) for tracker in trackers)

    module, _ = _patch_native_training(monkeypatch, native_train)
    monkeypatch.setattr(module, "update_policy", native_update)
    original_update = module.update_policy
    original_argv = sys.argv
    scoped_argv = ["pytest", "unrelated", "--runner-flag"]
    monkeypatch.setattr(sys, "argv", scoped_argv)

    Engine().train(_config(), hooks=hooks)

    assert [(step, total) for step, total, _ in hooks.steps] == [(1, 2), (2, 2)]
    assert [metrics for _, _, metrics in hooks.steps] == [{"loss": 2.5}, {"loss": 1.75}]
    assert len(native_results) == 2
    assert all(result[1] == {"native": 1.0} for result in native_results)
    assert seen_argv == [["lerobot-train"]]
    assert module.update_policy is original_update
    assert sys.argv is scoped_argv
    assert sys.argv == ["pytest", "unrelated", "--runner-flag"]
    monkeypatch.setattr(sys, "argv", original_argv)


def test_stop_request_propagates_and_restores_globals_and_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.scripts import lerobot_train

    hooks = _Hooks(stop_after=1)
    calls = 0

    def native_update(tracker: Any) -> tuple[Any, dict[str, float]]:
        nonlocal calls
        calls += 1
        return tracker, {}

    def native_train(config: Any) -> None:
        del config
        lerobot_train.update_policy(_tracker(0, 1.0))
        lerobot_train.update_policy(_tracker(1, 0.5))

    module, _ = _patch_native_training(monkeypatch, native_train)
    monkeypatch.setattr(module, "update_policy", native_update)
    original_update = module.update_policy
    scoped_argv = ["pytest", "test-cancel"]
    monkeypatch.setattr(sys, "argv", scoped_argv)

    with pytest.raises(TrainingCancelledError) as exc_info:
        Engine().train(_config(), hooks=hooks)

    assert exc_info.value.step == 1
    assert calls == 1
    assert len(hooks.steps) == 1
    assert module.update_policy is original_update
    assert sys.argv is scoped_argv
    assert sys.argv == ["pytest", "test-cancel"]


def test_native_training_error_restores_patched_update_and_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.scripts import lerobot_train

    def native_update(tracker: Any) -> tuple[Any, dict[str, float]]:
        raise RuntimeError("native update failed")

    def native_train(config: Any) -> None:
        del config
        lerobot_train.update_policy(_tracker(0, 1.0))

    module, _ = _patch_native_training(monkeypatch, native_train)
    monkeypatch.setattr(module, "update_policy", native_update)
    original_update = module.update_policy
    scoped_argv = ["pytest", "test-error"]
    monkeypatch.setattr(sys, "argv", scoped_argv)

    with pytest.raises(RuntimeError, match="native update failed"):
        Engine().train(_config(), hooks=_Hooks())

    assert module.update_policy is original_update
    assert sys.argv is scoped_argv
    assert sys.argv == ["pytest", "test-error"]
