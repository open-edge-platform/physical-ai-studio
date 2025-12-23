# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Benchmark, LiberoBenchmark, and BenchmarkResults."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from getiaction.benchmark import Benchmark, BenchmarkResults, LiberoBenchmark, TaskResult
from getiaction.policies.base import PolicyLike


@pytest.fixture
def mock_gym():
    """Mock gym with required attributes."""
    gym = MagicMock()
    gym.task_suite_name, gym.task_id, gym.task_name = "test", 0, "Test"
    return gym


@pytest.fixture
def eval_result():
    """Mock evaluate_policy return value."""
    return {
        "aggregated": {"pc_success": 80.0, "avg_sum_reward": 0.8, "avg_episode_length": 50.0, "avg_fps": 30.0},
        "per_episode": [],
    }


@pytest.fixture
def sample_results():
    """BenchmarkResults with 2 tasks."""
    results = BenchmarkResults()
    results.task_results.append(TaskResult("task_0", "Task Zero", 10, 60.0, 0.5, 100.0, 25.0))
    results.task_results.append(TaskResult("task_1", "Task One", 10, 80.0, 0.7, 120.0, 30.0))
    return results


class TestTaskResult:
    def test_to_dict(self):
        result = TaskResult("t0", "Test", 10, 75.0, 0.5, 100.0)
        assert result.to_dict()["task_id"] == "t0"
        assert "per_episode_data" in result.to_dict(include_per_episode=True)


class TestBenchmarkResults:
    def test_empty(self):
        r = BenchmarkResults()
        assert r.n_tasks == 0 and r.aggregate_success_rate == 0.0 and "timestamp" in r.metadata

    def test_aggregates(self, sample_results):
        assert (sample_results.n_tasks, sample_results.n_episodes) == (2, 20)
        assert sample_results.aggregate_success_rate == 70.0
        assert sample_results.aggregate_reward == 0.6

    def test_summary(self, sample_results):
        assert "task_0" in sample_results.summary() and "70.0%" in sample_results.summary()

    def test_json_roundtrip(self, sample_results, tmp_path):
        path = tmp_path / "r.json"
        sample_results.to_json(path)
        assert BenchmarkResults.from_json(path).aggregate_success_rate == 70.0

    def test_csv_export(self, sample_results, tmp_path):
        path = tmp_path / "r.csv"
        sample_results.to_csv(path)
        assert "task_0" in path.read_text()


class TestBenchmark:
    def test_init(self):
        b = Benchmark(gyms=[MagicMock(), MagicMock()], num_episodes=10, max_steps=100)
        assert len(b.gyms) == 2 and b.num_episodes == 10

    def test_empty_gyms_raises(self):
        with pytest.raises(ValueError, match="At least one gym"):
            Benchmark(gyms=[], num_episodes=10)

    def test_repr(self):
        b = Benchmark(gyms=[MagicMock()], num_episodes=20, max_steps=300)
        assert "gyms=1" in repr(b) and "num_episodes=20" in repr(b)

    def test_evaluate(self, mock_gym, eval_result):
        benchmark = Benchmark(gyms=[mock_gym], num_episodes=5, max_steps=100)
        with patch("getiaction.benchmark.benchmark.evaluate_policy", return_value=eval_result):
            results = benchmark.evaluate(MagicMock())
        assert results.n_tasks == 1 and results.overall_success_rate == 80.0

    def test_evaluate_multiple_policies(self, mock_gym, eval_result):
        benchmark = Benchmark(gyms=[mock_gym], num_episodes=5, max_steps=100)
        with patch("getiaction.benchmark.benchmark.evaluate_policy", return_value=eval_result):
            results = benchmark.evaluate([MagicMock(), MagicMock()])
        assert isinstance(results, dict) and len(results) == 2


class TestLiberoBenchmark:
    def test_init(self):
        with patch("getiaction.gyms.create_libero_gyms", return_value=[MagicMock() for _ in range(10)]):
            b = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
        assert b.task_suite == "libero_10" and len(b.gyms) == 10 and b.max_steps == 300

    def test_task_ids_subset(self):
        with patch("getiaction.gyms.create_libero_gyms", return_value=[MagicMock() for _ in range(3)]):
            b = LiberoBenchmark(task_suite="libero_spatial", task_ids=[0, 1, 2])
        assert b.task_ids == [0, 1, 2] and len(b.gyms) == 3

    def test_repr(self):
        with patch("getiaction.gyms.create_libero_gyms", return_value=[MagicMock()]):
            assert "libero_10" in repr(LiberoBenchmark(task_suite="libero_10"))


class TestPolicyLikeProtocol:
    """Tests for PolicyLike protocol compatibility."""

    def test_policy_like_protocol_with_mock(self):
        """Test that mock objects satisfying PolicyLike work with benchmark."""

        class MockInferenceModel:
            """Mock class mimicking InferenceModel interface."""

            def __init__(self):
                self.policy_name = "mock_inference_model"

            def select_action(self, observation):
                """Return dummy action tensor."""
                return torch.zeros(1, 7)

            def reset(self):
                """Reset internal state."""
                pass

        model = MockInferenceModel()
        assert isinstance(model, PolicyLike), "Mock should satisfy PolicyLike protocol"

    def test_benchmark_accepts_policy_like(self, mock_gym, eval_result):
        """Test that Benchmark.evaluate accepts PolicyLike objects."""

        class MockInferenceModel:
            def __init__(self):
                self.policy_name = "test_model"

            def select_action(self, observation):
                return torch.zeros(1, 7)

            def reset(self):
                pass

        model = MockInferenceModel()
        benchmark = Benchmark(gyms=[mock_gym], num_episodes=5, max_steps=100)

        with patch("getiaction.benchmark.benchmark.evaluate_policy", return_value=eval_result):
            results = benchmark.evaluate(model)

        assert results.n_tasks == 1
        assert results.overall_success_rate == 80.0

    def test_policy_name_extraction_from_inference_model(self):
        """Test that _get_policy_name extracts policy_name from InferenceModel-like objects."""
        from getiaction.benchmark.benchmark import _get_policy_name

        class MockInferenceModel:
            def __init__(self):
                self.policy_name = "exported_act_policy"

            def select_action(self, observation):
                return torch.zeros(1, 7)

            def reset(self):
                pass

        model = MockInferenceModel()
        assert _get_policy_name(model, 0) == "exported_act_policy"
