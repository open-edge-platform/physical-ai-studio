# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Callback base class, built-in callbacks, and InferenceModel wiring."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, override
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.inference.callbacks import Callback, LatencyMonitor, ThroughputMonitor
from physicalai.inference.model import InferenceModel


class _RecordingCallback(Callback):
    """Test callback that records all lifecycle events."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.predict_start_inputs: list[dict] = []
        self.predict_end_outputs: list[dict] = []

    @override
    def on_predict_start(self, inputs: dict[str, Any]) -> None:
        self.events.append("predict_start")
        self.predict_start_inputs.append(dict(inputs))

    @override
    def on_predict_end(self, outputs: dict[str, Any]) -> None:
        self.events.append("predict_end")
        self.predict_end_outputs.append(dict(outputs))

    @override
    def on_reset(self) -> None:
        self.events.append("reset")

    @override
    def on_load(self, model: InferenceModel) -> None:
        self.events.append("load")


class _InputModifyingCallback(Callback):
    """Test callback that modifies inputs by adding a key."""

    @override
    def on_predict_start(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {**inputs, "injected": np.array([42.0])}


class _OutputModifyingCallback(Callback):
    """Test callback that scales all output arrays."""

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    @override
    def on_predict_end(self, outputs: dict[str, Any]) -> dict[str, Any]:
        return {key: value * self.factor for key, value in outputs.items()}


@pytest.fixture
def mock_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.input_names = []
    adapter.output_names = ["actions"]
    adapter.predict.return_value = {"actions": np.array([[1.0, 2.0]])}
    adapter.default_device.return_value = "cpu"
    return adapter


@pytest.fixture
def mock_export_dir(tmp_path: Path) -> Path:
    import yaml

    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    metadata = {
        "policy_class": "physicalai.policies.act.ACT",
        "backend": "openvino",
        "use_action_queue": False,
        "chunk_size": 1,
    }
    with (export_dir / "metadata.yaml").open("w") as f:
        yaml.dump(metadata, f)

    (export_dir / "act.xml").touch()
    (export_dir / "act.bin").touch()

    return export_dir


def _make_model(
    export_dir: Path,
    mock_adapter: MagicMock,
    callbacks: list[Callback] | None = None,
) -> InferenceModel:
    with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
        return InferenceModel(export_dir, callbacks=callbacks)


class TestCallbackBase:
    def test_can_instantiate_base_callback(self) -> None:
        cb = Callback()
        assert isinstance(cb, Callback)

    def test_hooks_return_none_by_default(self) -> None:
        cb = Callback()
        assert cb.on_predict_start({"x": np.array([1.0])}) is None
        assert cb.on_predict_end({"action": np.array([1.0])}) is None
        assert cb.on_reset() is None
        model = MagicMock(spec=InferenceModel)
        assert cb.on_load(model) is None

    def test_repr(self) -> None:
        cb = Callback()
        assert repr(cb) == "Callback()"

    def test_subclass_repr_uses_class_name(self) -> None:
        cb = _RecordingCallback()
        assert repr(cb) == "_RecordingCallback()"


class TestLatencyMonitor:
    def test_initial_state(self) -> None:
        monitor = LatencyMonitor()
        assert monitor.latest_ms == 0.0
        assert monitor.total_calls == 0
        assert monitor.avg_ms == 0.0
        assert monitor.min_ms == 0.0
        assert monitor.max_ms == 0.0
        assert monitor.p95_ms == 0.0

    def test_records_duration(self) -> None:
        monitor = LatencyMonitor()
        monitor.on_predict_start({"x": np.array([1.0])})
        monitor.on_predict_end({"action": np.array([1.0])})
        assert monitor.latest_ms > 0.0
        assert monitor.total_calls == 1
        assert len(monitor._window) == 1
        assert monitor._window[0] == monitor.latest_ms

    def test_accumulates_in_window(self) -> None:
        monitor = LatencyMonitor()
        for _ in range(3):
            monitor.on_predict_start({"x": np.array([1.0])})
            monitor.on_predict_end({"action": np.array([1.0])})
        assert len(monitor._window) == 3
        assert monitor.total_calls == 3

    def test_window_evicts_oldest(self) -> None:
        monitor = LatencyMonitor(window_size=3)
        for _ in range(5):
            monitor.on_predict_start({"x": np.array([1.0])})
            monitor.on_predict_end({"action": np.array([1.0])})
        assert len(monitor._window) == 3
        assert monitor.total_calls == 5

    def test_statistics_with_known_values(self) -> None:
        monitor = LatencyMonitor(window_size=100)
        known_durations = [10.0, 20.0, 30.0, 40.0, 50.0]
        for duration in known_durations:
            monitor._window.append(duration)
            monitor.total_calls += 1
        assert monitor.avg_ms == pytest.approx(30.0)
        assert monitor.min_ms == pytest.approx(10.0)
        assert monitor.max_ms == pytest.approx(50.0)

    def test_p95_with_known_values(self) -> None:
        monitor = LatencyMonitor(window_size=200)
        for i in range(1, 101):
            monitor._window.append(float(i))
            monitor.total_calls += 1
        # 100 values 1..100, p95 nearest-rank = ceil(0.95*100)-1 = index 94 → value 95
        assert monitor.p95_ms == pytest.approx(95.0)

    def test_repr(self) -> None:
        monitor = LatencyMonitor()
        assert "LatencyMonitor" in repr(monitor)
        assert "latest=0.0ms" in repr(monitor)
        assert "calls=0" in repr(monitor)


class TestThroughputMonitor:
    def test_initial_state(self) -> None:
        monitor = ThroughputMonitor()
        assert monitor.throughput == 0.0
        assert monitor.total_calls == 0

    def test_records_predictions(self) -> None:
        monitor = ThroughputMonitor()
        for _ in range(5):
            monitor.on_predict_end({"action": np.array([1.0])})
        assert monitor.total_calls == 5
        assert monitor.throughput > 0.0

    def test_single_prediction_no_throughput(self) -> None:
        monitor = ThroughputMonitor()
        monitor.on_predict_end({"action": np.array([1.0])})
        assert monitor.total_calls == 1
        assert monitor.throughput == 0.0

    def test_default_window_seconds(self) -> None:
        monitor = ThroughputMonitor()
        assert monitor._window_seconds == 10.0

    def test_custom_window_seconds(self) -> None:
        monitor = ThroughputMonitor(window_seconds=5.0)
        assert monitor._window_seconds == 5.0

    def test_time_based_pruning(self) -> None:
        monitor = ThroughputMonitor(window_seconds=1.0)
        now = time.perf_counter()
        monitor._timestamps.append(now - 5.0)
        monitor._timestamps.append(now - 4.0)
        monitor._timestamps.append(now - 3.0)
        monitor.total_calls = 3

        monitor.on_predict_end({"action": np.array([1.0])})

        assert monitor.total_calls == 4
        assert len(monitor._timestamps) == 1

    def test_repr(self) -> None:
        monitor = ThroughputMonitor(window_seconds=5.0)
        assert "ThroughputMonitor" in repr(monitor)
        assert "throughput=0.0/s" in repr(monitor)
        assert "total=0" in repr(monitor)
        assert "window=5.0s" in repr(monitor)

    def test_throughput_updates_after_predictions(self) -> None:
        monitor = ThroughputMonitor(window_seconds=10.0)
        for _ in range(3):
            monitor.on_predict_end({"action": np.array([1.0])})
        assert monitor.throughput > 0.0
        assert monitor.total_calls == 3


class TestCallbackWiring:
    def test_on_load_fires_during_init(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        recorder = _RecordingCallback()
        model = _make_model(mock_export_dir, mock_adapter, callbacks=[recorder])
        assert "load" in recorder.events
        assert recorder.events == ["load"]
        assert model.callbacks == [recorder]

    def test_on_predict_start_and_end_fire(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        recorder = _RecordingCallback()
        model = _make_model(mock_export_dir, mock_adapter, callbacks=[recorder])
        recorder.events.clear()

        obs = {"state": np.array([1.0])}
        model(obs)

        assert recorder.events == ["predict_start", "predict_end"]

    def test_on_reset_fires(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        recorder = _RecordingCallback()
        model = _make_model(mock_export_dir, mock_adapter, callbacks=[recorder])
        recorder.events.clear()

        model.reset()

        assert recorder.events == ["reset"]

    def test_multiple_callbacks_fire_in_order(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        r1 = _RecordingCallback()
        r2 = _RecordingCallback()
        model = _make_model(mock_export_dir, mock_adapter, callbacks=[r1, r2])
        r1.events.clear()
        r2.events.clear()

        model({"state": np.array([1.0])})
        model.reset()

        assert r1.events == ["predict_start", "predict_end", "reset"]
        assert r2.events == ["predict_start", "predict_end", "reset"]

    def test_no_callbacks_is_default(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        model = _make_model(mock_export_dir, mock_adapter)
        assert model.callbacks == []

    def test_callback_can_modify_inputs(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        model = _make_model(
            mock_export_dir,
            mock_adapter,
            callbacks=[_InputModifyingCallback()],
        )

        obs = {"state": np.array([1.0])}
        model(obs)

        call_args = mock_adapter.predict.call_args[0][0]
        assert "injected" in call_args
        np.testing.assert_array_equal(call_args["injected"], np.array([42.0]))

    def test_callback_can_modify_outputs(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        mock_adapter.predict.return_value = {"actions": np.array([[5.0]])}
        model = _make_model(
            mock_export_dir,
            mock_adapter,
            callbacks=[_OutputModifyingCallback(factor=3.0)],
        )

        outputs = model({"state": np.array([1.0])})
        np.testing.assert_array_equal(outputs["actions"], np.array([[15.0]]))

    def test_chained_output_modifications(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        mock_adapter.predict.return_value = {"actions": np.array([[2.0]])}
        model = _make_model(
            mock_export_dir,
            mock_adapter,
            callbacks=[
                _OutputModifyingCallback(factor=3.0),
                _OutputModifyingCallback(factor=2.0),
            ],
        )

        outputs = model({"state": np.array([1.0])})
        # 2.0 * 3.0 = 6.0, then 6.0 * 2.0 = 12.0
        np.testing.assert_array_equal(outputs["actions"], np.array([[12.0]]))

    def test_latency_monitor_integration(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        monitor = LatencyMonitor()
        model = _make_model(mock_export_dir, mock_adapter, callbacks=[monitor])

        model({"state": np.array([1.0])})

        assert monitor.latest_ms > 0.0
        assert monitor.total_calls == 1
        assert len(monitor._window) == 1

    def test_throughput_monitor_integration(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        monitor = ThroughputMonitor()
        model = _make_model(mock_export_dir, mock_adapter, callbacks=[monitor])

        for _ in range(5):
            model({"state": np.array([1.0])})

        assert monitor.total_calls == 5
        assert monitor.throughput > 0.0


class TestContextManager:
    def test_enter_returns_self(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        model = _make_model(mock_export_dir, mock_adapter)
        with model as m:
            assert m is model

    def test_with_statement(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        with _make_model(mock_export_dir, mock_adapter) as model:
            outputs = model({"state": np.array([1.0])})
            assert isinstance(outputs, dict)

    def test_callbacks_work_inside_context(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        recorder = _RecordingCallback()
        with _make_model(mock_export_dir, mock_adapter, callbacks=[recorder]) as model:
            recorder.events.clear()
            model({"state": np.array([1.0])})
            model.reset()

        assert recorder.events == ["predict_start", "predict_end", "reset"]
