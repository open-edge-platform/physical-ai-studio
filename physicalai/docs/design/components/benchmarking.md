# Benchmarking API Design

## Executive Summary

This document defines the benchmarking API for the `physicalai` ecosystem, split across two distributions:

- **`physicalai`** (runtime) — NumPy-only evaluation protocols, runner, results, and latency metrics
- **`physicalai-train`** (training) — Benchmark presets (Libero, PushT), torch↔numpy adapters, and the existing rollout runtime

**Key decisions:**

- **Protocols**: `EnvLike` and `PolicyLike` use `dict[str, np.ndarray]` observations and `np.ndarray` actions — no torch at import time
- **Environment semantics**: Gymnasium 5-tuple (`obs, reward, terminated, truncated, info`) for adapter compatibility, with a `step_done()` convenience helper
- **Runner**: `BenchmarkRunner` class for configuration and reuse, plus a thin `run_benchmark()` function for one-shot usage
- **Adapters**: Explicit `TorchPolicyAdapter` and `GymEnvAdapter` in `physicalai-train` — the runner never imports torch
- **Results**: `BenchmarkResults` and `TaskResult` move to runtime unchanged (already stdlib-only)
- **Latency**: `LatencyStats` with p50/p95/p99 percentiles — edge deployers need this
- **Video**: `VideoRecorder` stays in `physicalai-train` unless a runtime use case emerges

**Goal**: Any user with `pip install physicalai` can benchmark an exported model against a custom environment using only NumPy. Training users get the same workflow they have today via adapters.

---

## Background

### Current Architecture

The existing benchmark module (`getiaction.benchmark`) is tightly coupled to PyTorch:

| Component           | File                         | Torch Dependency                                                     |
| ------------------- | ---------------------------- | -------------------------------------------------------------------- |
| `Benchmark`         | `benchmark/benchmark.py`     | Calls `evaluate_policy()` which uses `torch.inference_mode()`        |
| `LiberoBenchmark`   | `benchmark/libero.py`        | Lazy-imports `create_libero_gyms`                                    |
| `BenchmarkResults`  | `benchmark/results.py`       | **None** — pure stdlib dataclasses                                   |
| `evaluate_policy()` | `eval/rollout/functional.py` | `torch.Tensor` throughout, `torch.stack()`, `torch.inference_mode()` |
| `Rollout` metric    | `eval/rollout/metric.py`     | `torchmetrics.Metric` subclass                                       |
| `VideoRecorder`     | `eval/video.py`              | **None** — numpy + imageio only                                      |

All existing interfaces use `torch.Tensor` and the `Observation` dataclass (torch-dependent):

```python
# Current PolicyLike — torch-dependent
class PolicyLike(Protocol):
    def select_action(self, observation: Observation) -> torch.Tensor: ...
    def reset(self) -> None: ...

# Current Gym ABC — torch-dependent
class Gym(ABC):
    def reset(self, *, seed=None) -> tuple[Observation, dict]: ...
    def step(self, action: torch.Tensor) -> tuple[Observation, float, bool, bool, dict]: ...
```

These interfaces cannot exist in the runtime distribution. The runtime needs entirely new protocols using standard types.

### Why Split Benchmarking?

The runtime distribution (`physicalai`) owns `InferenceModel` — the primary way users deploy exported policies. If the only way to benchmark a model requires `pip install physicalai-train` (and transitively PyTorch, CUDA, etc.), edge deployers face a bad choice: install 5GB of training dependencies just to run a benchmark, or write their own evaluation loop.

The split follows precedent:

- **HuggingFace** ships `evaluate` as a lightweight package separate from `transformers`
- **OpenVINO** ships `benchmark_app` in the runtime, not in training tools
- **MLflow** evaluation runs without training dependencies

---

## Design Principles

### NumPy-Only at the Boundary

All runtime protocols use `dict[str, np.ndarray]` for observations and `np.ndarray` for actions. No torch, no custom dataclasses with torch fields. This matches the Robot interface (`get_observation() → dict`) and Camera interface (`Frame.data → np.ndarray`).

### Gymnasium Semantics

The `EnvLike` protocol uses Gymnasium's 5-tuple step signature (`obs, reward, terminated, truncated, info`). This is the de facto standard. Fighting it means every adapter has to translate, and users can't wrap standard Gymnasium environments without glue code.

A `step_done()` helper flattens terminated/truncated into a single `done` boolean for the common case.

### Explicit Adapters Over Dual-Typed Interfaces

The runner accepts only NumPy protocols. Torch objects are wrapped by explicit adapters in `physicalai-train`. This keeps the runner's import graph clean and makes the dependency boundary testable with CI rules.

The alternative — making the runner accept both torch and numpy types — would require conditional imports, isinstance checks, and runtime torch detection. Rejected.

### Results Are Already Clean

`BenchmarkResults` and `TaskResult` use only stdlib types (json, csv, dataclasses, datetime). They move to the runtime distribution unchanged.

---

## Package Structure

### Runtime Distribution (`physicalai`)

```text
physicalai/
└── benchmark/
    ├── __init__.py          # Public API re-exports
    ├── _protocols.py        # EnvLike, PolicyLike protocols
    ├── _runner.py           # BenchmarkRunner, run_benchmark()
    ├── _results.py          # BenchmarkResults, TaskResult (migrated)
    └── _latency.py          # LatencyStats
```

### Training Distribution (`physicalai-train`)

```text
physicalai/
└── benchmark/
    ├── adapters.py          # TorchPolicyAdapter, GymEnvAdapter
    ├── libero.py            # LiberoBenchmark preset
    └── pusht.py             # PushTBenchmark preset (future)
```

Both distributions contribute to the `physicalai.benchmark` namespace via PEP 420 (implicit namespace packages — no `__init__.py` at the `physicalai/` root).

The runtime's `physicalai/benchmark/__init__.py` re-exports the public API:

```python
from physicalai.benchmark._protocols import EnvLike, PolicyLike
from physicalai.benchmark._runner import BenchmarkConfig, BenchmarkRunner, run_benchmark
from physicalai.benchmark._results import BenchmarkResults, TaskResult
from physicalai.benchmark._latency import LatencyStats

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResults",
    "BenchmarkRunner",
    "EnvLike",
    "LatencyStats",
    "PolicyLike",
    "TaskResult",
    "run_benchmark",
]
```

---

## Core Interface

### Observation and Action Types

```python
# physicalai/benchmark/_protocols.py
from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

Observation = Mapping[str, NDArray[np.generic]]
"""Robot observation as a mapping of named arrays.

Matches the Robot interface's get_observation() return type.
Typical keys: "images", "state", "language_instruction".
"""

Action = NDArray[np.generic]
"""Robot action as a numpy array (joint positions, velocities, etc.)."""
```

**Why `Mapping[str, NDArray]` instead of a dataclass?** The Robot interface returns `dict[str, Any]`. A protocol-level `Mapping` accepts any dict without conversion. A custom dataclass would force every environment and robot to construct a specific type.

### EnvLike Protocol

```python
class EnvLike(Protocol):
    """Environment protocol for benchmarking.

    Follows Gymnasium semantics: reset() returns (obs, info),
    step() returns the standard 5-tuple.
    """

    def reset(self, *, seed: int | None = None) -> tuple[Observation, Mapping[str, Any]]:
        """Reset environment and return initial observation.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (observation, info_dict).
        """
        ...

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, Mapping[str, Any]]:
        """Execute action and return result.

        Args:
            action: Action array to execute.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        ...

    def close(self) -> None:
        """Release environment resources."""
        ...
```

### StepOutcome Helper

```python
from typing import NamedTuple

class StepOutcome(NamedTuple):
    """Flattened step result for simple evaluation loops."""

    obs: Observation
    reward: float
    done: bool
    info: Mapping[str, Any]


def step_done(
    outcome: tuple[Observation, float, bool, bool, Mapping[str, Any]],
) -> StepOutcome:
    """Flatten terminated/truncated into a single done flag.

    Args:
        outcome: Raw 5-tuple from EnvLike.step().

    Returns:
        StepOutcome with done = terminated or truncated.
    """
    obs, reward, terminated, truncated, info = outcome
    return StepOutcome(obs, reward, terminated or truncated, info)
```

Most benchmark loops only care about "is the episode over?" — not the distinction between terminated and truncated. `step_done()` provides that without losing the raw 5-tuple for users who need it.

### PolicyLike Protocol

```python
class PolicyLike(Protocol):
    """Policy protocol for benchmarking.

    Uses NumPy types only. Training policies are wrapped via
    TorchPolicyAdapter in physicalai-train.
    """

    def select_action(self, observation: Observation) -> Action:
        """Select action given current observation.

        Args:
            observation: Current environment observation.

        Returns:
            Action array to execute.
        """
        ...

    def reset(self) -> None:
        """Reset internal state between episodes.

        Called before each episode begins. Stateless policies
        can implement this as a no-op.
        """
        ...
```

**Name alignment**: Both the current `PolicyLike` and `InferenceModel` use `select_action()` and `reset()`. The runtime protocol keeps the same method names — only the types change (torch → numpy). This means `InferenceModel` can satisfy the protocol directly if it returns numpy arrays.

---

## BenchmarkRunner

### Configuration

```python
# physicalai/benchmark/_runner.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for benchmark evaluation.

    Attributes:
        num_episodes: Number of episodes per environment.
        max_steps: Maximum steps per episode. None for no limit.
        seed: Random seed for reproducibility.
        measure_latency: Whether to record per-step policy latency.
    """

    num_episodes: int = 10
    max_steps: int | None = None
    seed: int | None = 42
    measure_latency: bool = True
```

### Runner Class

```python
import logging
import time
from typing import Iterable

from physicalai.benchmark._latency import LatencyStats
from physicalai.benchmark._protocols import EnvLike, PolicyLike, step_done
from physicalai.benchmark._results import BenchmarkResults, TaskResult

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Evaluates a policy across one or more environments.

    Orchestrates the episode loop, collects per-task results,
    measures policy inference latency, and aggregates into
    BenchmarkResults.

    Args:
        envs: Environments to evaluate on. Each is treated as a
            separate task.
        policy: Policy to evaluate. Must satisfy PolicyLike protocol.
        config: Evaluation configuration.

    Example:
        >>> from physicalai.benchmark import BenchmarkRunner, BenchmarkConfig
        >>> runner = BenchmarkRunner(envs=[env], policy=policy, config=BenchmarkConfig())
        >>> results = runner.evaluate()
        >>> print(results.summary())
    """

    def __init__(
        self,
        envs: Iterable[EnvLike],
        policy: PolicyLike,
        config: BenchmarkConfig | None = None,
    ) -> None:
        self._envs = list(envs)
        self._policy = policy
        self._config = config or BenchmarkConfig()

        if not self._envs:
            msg = "At least one environment is required."
            raise ValueError(msg)

    def evaluate(self) -> BenchmarkResults:
        """Run evaluation and return results.

        Iterates over environments, runs num_episodes per environment,
        collects success/reward/length metrics, and optionally measures
        policy inference latency.

        Returns:
            BenchmarkResults with per-task and aggregate metrics.
        """
        task_results: list[TaskResult] = []
        all_latencies: list[float] = []

        for env_idx, env in enumerate(self._envs):
            task_id = getattr(env, "task_id", f"env_{env_idx}")
            task_name = getattr(env, "task_name", task_id)
            logger.info("Evaluating task %s (%d/%d)", task_id, env_idx + 1, len(self._envs))

            episode_data: list[dict] = []

            for ep in range(self._config.num_episodes):
                seed = self._config.seed + ep if self._config.seed is not None else None
                obs, info = env.reset(seed=seed)
                self._policy.reset()

                total_reward = 0.0
                steps = 0
                done = False
                ep_latencies: list[float] = []

                while not done:
                    if self._config.max_steps is not None and steps >= self._config.max_steps:
                        break

                    t0 = time.perf_counter()
                    action = self._policy.select_action(obs)
                    if self._config.measure_latency:
                        ep_latencies.append(time.perf_counter() - t0)

                    result = env.step(action)
                    obs, reward, done, info = step_done(result)
                    total_reward += float(reward)
                    steps += 1

                success = bool(info.get("success", False))
                episode_data.append({
                    "episode": ep,
                    "reward": total_reward,
                    "steps": steps,
                    "success": success,
                })
                all_latencies.extend(ep_latencies)

            # Aggregate per-task
            n_eps = len(episode_data)
            successes = sum(1 for e in episode_data if e["success"])
            task_results.append(TaskResult(
                task_id=str(task_id),
                task_name=str(task_name),
                n_episodes=n_eps,
                success_rate=(successes / n_eps * 100) if n_eps > 0 else 0.0,
                avg_reward=sum(e["reward"] for e in episode_data) / max(n_eps, 1),
                avg_episode_length=sum(e["steps"] for e in episode_data) / max(n_eps, 1),
                per_episode_data=episode_data,
            ))

            env.close()

        latency_stats = LatencyStats.from_samples(all_latencies)

        return BenchmarkResults(
            task_results=task_results,
            metadata={
                "num_episodes_per_task": self._config.num_episodes,
                "max_steps": self._config.max_steps,
                "seed": self._config.seed,
                "latency": latency_stats.to_dict(),
            },
        )
```

### Convenience Function

```python
def run_benchmark(
    envs: Iterable[EnvLike],
    policy: PolicyLike,
    config: BenchmarkConfig | None = None,
) -> BenchmarkResults:
    """One-shot benchmark evaluation.

    Convenience wrapper around BenchmarkRunner for simple usage.

    Args:
        envs: Environments to evaluate on.
        policy: Policy to evaluate.
        config: Evaluation configuration.

    Returns:
        BenchmarkResults with per-task and aggregate metrics.

    Example:
        >>> from physicalai.benchmark import run_benchmark
        >>> results = run_benchmark(envs=[env], policy=policy)
    """
    return BenchmarkRunner(envs, policy, config).evaluate()
```

---

## Latency Metrics

Edge deployers care about inference latency — not just accuracy. The runner measures wall-clock time around `select_action()` and reports percentile statistics.

```python
# physicalai/benchmark/_latency.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    """Policy inference latency statistics.

    All values in milliseconds. Computed from per-step
    wall-clock measurements around policy.select_action().

    Attributes:
        count: Number of measurements.
        mean_ms: Mean latency.
        std_ms: Standard deviation.
        min_ms: Minimum latency.
        p50_ms: Median (50th percentile).
        p95_ms: 95th percentile.
        p99_ms: 99th percentile.
        max_ms: Maximum latency.
    """

    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @staticmethod
    def from_samples(samples: Sequence[float]) -> LatencyStats:
        """Compute statistics from raw timing samples.

        Args:
            samples: Per-step latencies in seconds (from time.perf_counter).

        Returns:
            LatencyStats with values converted to milliseconds.
        """
        if not samples:
            return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        arr = np.array(samples, dtype=np.float64) * 1000.0
        return LatencyStats(
            count=int(arr.size),
            mean_ms=float(arr.mean()),
            std_ms=float(arr.std()),
            min_ms=float(arr.min()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            max_ms=float(arr.max()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }

    def summary(self) -> str:
        """Human-readable latency summary."""
        if self.count == 0:
            return "No latency measurements."
        return (
            f"Latency ({self.count} samples): "
            f"mean={self.mean_ms:.1f}ms, "
            f"p50={self.p50_ms:.1f}ms, "
            f"p95={self.p95_ms:.1f}ms, "
            f"p99={self.p99_ms:.1f}ms"
        )
```

---

## Results (Migrated)

`BenchmarkResults` and `TaskResult` move from `getiaction.benchmark.results` to `physicalai.benchmark._results` **unchanged**. They use only stdlib types (json, csv, dataclasses, datetime) and have zero torch or numpy dependencies in their data fields.

The only addition: `BenchmarkResults.metadata` now carries a `"latency"` key with `LatencyStats.to_dict()` output when latency measurement is enabled.

### Current API (preserved)

```python
@dataclass
class TaskResult:
    task_id: str
    task_name: str
    n_episodes: int
    success_rate: float          # 0-100
    avg_reward: float
    avg_episode_length: float
    avg_fps: float = 0.0
    per_episode_data: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class BenchmarkResults:
    task_results: list[TaskResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Aggregate properties
    aggregate_success_rate: float   # Mean success rate across tasks
    aggregate_reward: float         # Mean reward across tasks
    aggregate_episode_length: float
    aggregate_fps: float

    # Export
    def summary(self) -> str: ...
    def to_json(self, path) -> Path: ...
    def to_csv(self, path) -> Path: ...
    def from_json(cls, path) -> BenchmarkResults: ...
```

No changes needed. The class is already distribution-ready.

---

## Adapters (`physicalai-train`)

Training users have torch-based policies and Gym environments. Explicit adapters bridge them to the runtime protocols.

### TorchPolicyAdapter

```python
# physicalai-train: physicalai/benchmark/adapters.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from physicalai.benchmark import Action, Observation

class TorchPolicyAdapter:
    """Wraps a torch-based PolicyLike to satisfy the NumPy PolicyLike protocol.

    Handles observation dict[str, np.ndarray] → dict[str, torch.Tensor]
    conversion before calling the torch policy, and converts the
    returned torch.Tensor action back to np.ndarray.

    Args:
        policy: Torch-based policy with select_action(Observation) → Tensor.
        device: Torch device for tensor conversion. Defaults to policy device.

    Example:
        >>> from physicalai.benchmark.adapters import TorchPolicyAdapter
        >>> adapter = TorchPolicyAdapter(torch_policy)
        >>> results = run_benchmark(envs=[env], policy=adapter)
    """

    def __init__(self, policy: Any, device: torch.device | str | None = None) -> None:
        self._policy = policy
        self._device = device or getattr(policy, "device", "cpu")

    def reset(self) -> None:
        """Delegate reset to wrapped policy."""
        self._policy.reset()

    def select_action(self, observation: Observation) -> Action:
        """Convert observation to torch, run policy, convert action to numpy.

        Args:
            observation: NumPy observation dict.

        Returns:
            Action as numpy array.
        """
        obs_tensors = {
            k: torch.from_numpy(np.asarray(v)).to(self._device)
            for k, v in observation.items()
        }
        with torch.inference_mode():
            action = self._policy.select_action(obs_tensors)
        return action.detach().cpu().numpy()
```

### GymEnvAdapter

```python
class GymEnvAdapter:
    """Wraps a getiaction Gym to satisfy the NumPy EnvLike protocol.

    Converts torch-based observations from the Gym ABC to
    dict[str, np.ndarray] format expected by the runtime runner.

    Args:
        gym: A getiaction Gym instance.

    Example:
        >>> from physicalai.benchmark.adapters import GymEnvAdapter
        >>> adapted_env = GymEnvAdapter(libero_gym)
        >>> results = run_benchmark(envs=[adapted_env], policy=policy)
    """

    def __init__(self, gym: Any) -> None:
        self._gym = gym

    @property
    def task_id(self) -> str:
        """Forward task_id from wrapped gym."""
        return getattr(self._gym, "task_id", "unknown")

    @property
    def task_name(self) -> str:
        """Forward task_name from wrapped gym."""
        return getattr(self._gym, "task_name", self.task_id)

    def reset(self, *, seed: int | None = None) -> tuple[Mapping[str, NDArray], Mapping[str, Any]]:
        """Reset wrapped gym and convert observation.

        Args:
            seed: Optional random seed.

        Returns:
            Tuple of (numpy observation dict, info dict).
        """
        obs, info = self._gym.reset(seed=seed)
        return self._to_numpy_obs(obs), info

    def step(
        self, action: NDArray,
    ) -> tuple[Mapping[str, NDArray], float, bool, bool, Mapping[str, Any]]:
        """Step wrapped gym with numpy action.

        Converts numpy action to torch, steps the gym, converts
        observation back to numpy.

        Args:
            action: Action as numpy array.

        Returns:
            Standard 5-tuple with numpy observations.
        """
        action_tensor = torch.from_numpy(np.asarray(action))
        obs, reward, terminated, truncated, info = self._gym.step(action_tensor)
        return self._to_numpy_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        """Close wrapped gym."""
        self._gym.close()

    @staticmethod
    def _to_numpy_obs(obs: Any) -> Mapping[str, NDArray]:
        """Convert gym observation to numpy dict.

        Handles both dict-like observations and the getiaction
        Observation dataclass (which has dict-like access).
        """
        if isinstance(obs, dict):
            return {k: np.asarray(v) for k, v in obs.items()}
        # Handle Observation dataclass — convert tensor fields
        result: dict[str, NDArray] = {}
        for key in vars(obs):
            val = getattr(obs, key)
            if hasattr(val, "numpy"):  # torch.Tensor
                result[key] = val.detach().cpu().numpy()
            elif isinstance(val, np.ndarray):
                result[key] = val
        return result
```

### Benchmark Presets

Benchmark presets like `LiberoBenchmark` stay in `physicalai-train` and compose the runtime's `BenchmarkRunner` with pre-configured environments:

```python
# physicalai-train: physicalai/benchmark/libero.py
from __future__ import annotations

from physicalai.benchmark import BenchmarkConfig, BenchmarkRunner, BenchmarkResults
from physicalai.benchmark.adapters import GymEnvAdapter, TorchPolicyAdapter


class LiberoBenchmark:
    """LIBERO benchmark preset.

    Creates Libero gym environments and wraps them with GymEnvAdapter
    for compatibility with the runtime BenchmarkRunner.

    Args:
        task_suite: LIBERO task suite name.
        num_episodes: Episodes per task.
        max_steps: Maximum steps per episode.
        seed: Random seed.

    Example:
        >>> benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
        >>> results = benchmark.evaluate(policy)
    """

    def __init__(
        self,
        task_suite: str = "libero_10",
        num_episodes: int = 20,
        max_steps: int = 300,
        seed: int = 42,
    ) -> None:
        self._task_suite = task_suite
        self._config = BenchmarkConfig(
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
        )

    def evaluate(self, policy: Any) -> BenchmarkResults:
        """Evaluate a policy on LIBERO tasks.

        Accepts both torch policies and numpy PolicyLike objects.
        Torch policies are automatically wrapped.

        Args:
            policy: Policy to evaluate.

        Returns:
            BenchmarkResults with per-task and aggregate metrics.
        """
        from getiaction.gyms.libero import create_libero_gyms

        gyms = create_libero_gyms(task_suite=self._task_suite)
        envs = [GymEnvAdapter(g) for g in gyms]

        # Wrap torch policy if needed
        adapted_policy = self._adapt_policy(policy)

        runner = BenchmarkRunner(envs=envs, policy=adapted_policy, config=self._config)
        return runner.evaluate()

    @staticmethod
    def _adapt_policy(policy: Any) -> Any:
        """Wrap policy if it's torch-based."""
        if hasattr(policy, "device"):  # Heuristic: torch policies have .device
            return TorchPolicyAdapter(policy)
        return policy
```

---

## InferenceModel Compatibility

`InferenceModel` currently uses `select_action(Observation) → torch.Tensor`. For the runtime protocol, it needs to accept `dict[str, np.ndarray]` and return `np.ndarray`.

### Approach: Add NumPy-Native `predict()`

Add a `predict()` method to the runtime `InferenceModel` that accepts and returns numpy types. The existing `select_action()` stays for backward compatibility but delegates to `predict()` internally.

```python
# physicalai/inference/model.py (runtime distribution)
class InferenceModel:
    """Exported model for inference.

    Satisfies the benchmark PolicyLike protocol directly —
    no adapter needed.
    """

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Run inference with numpy types.

        This is the primary inference method for the runtime
        distribution. Backend-specific (ONNX, OpenVINO, Torch)
        implementations handle format conversion internally.

        Args:
            observation: Observation as dict of numpy arrays.

        Returns:
            Action as numpy array.
        """
        ...

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Alias for predict(). Satisfies PolicyLike protocol."""
        return self.predict(observation)

    def reset(self) -> None:
        """Reset internal state. Satisfies PolicyLike protocol."""
        ...
```

This means `InferenceModel` satisfies the benchmark `PolicyLike` protocol directly — no adapter needed:

```python
from physicalai.benchmark import run_benchmark
from physicalai.inference import InferenceModel

model = InferenceModel.load("./exported_policy")
results = run_benchmark(envs=[my_env], policy=model)
```

---

## Usage Examples

### Runtime-Only: Edge Deployment Benchmark

```python
"""Benchmark an exported policy on custom hardware."""
from physicalai.benchmark import BenchmarkConfig, run_benchmark
from physicalai.inference import InferenceModel


# Custom environment wrapping real robot or simulation
class MyRobotEnv:
    def reset(self, *, seed=None):
        obs = {"state": np.zeros(7), "images": {"cam": np.zeros((480, 640, 3), dtype=np.uint8)}}
        return obs, {}

    def step(self, action):
        obs = {"state": np.zeros(7), "images": {"cam": np.zeros((480, 640, 3), dtype=np.uint8)}}
        reward = 0.0
        terminated = False
        truncated = False
        info = {"success": False}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


model = InferenceModel.load("./exported_policy")
config = BenchmarkConfig(num_episodes=50, max_steps=500, measure_latency=True)
results = run_benchmark(envs=[MyRobotEnv()], policy=model, config=config)

print(results.summary())
print(results.metadata["latency"])
# {'count': 25000, 'mean_ms': 12.3, 'p50_ms': 11.8, 'p95_ms': 15.2, 'p99_ms': 18.7, ...}
```

### Training: LIBERO Benchmark (Current Workflow)

```python
"""Standard training benchmark — same ergonomics as today."""
from physicalai.benchmark.libero import LiberoBenchmark
from physicalai.policies import ACT

policy = ACT.load_from_checkpoint("checkpoints/last.ckpt")
benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
results = benchmark.evaluate(policy)
print(f"Success rate: {results.aggregate_success_rate:.1f}%")
results.to_json("results.json")
```

### Training: Manual Adapter Usage

```python
"""Explicit adapter usage for custom gym environments."""
from physicalai.benchmark import BenchmarkConfig, BenchmarkRunner
from physicalai.benchmark.adapters import GymEnvAdapter, TorchPolicyAdapter

# Wrap torch objects
adapted_envs = [GymEnvAdapter(gym) for gym in my_torch_gyms]
adapted_policy = TorchPolicyAdapter(my_torch_policy)

runner = BenchmarkRunner(
    envs=adapted_envs,
    policy=adapted_policy,
    config=BenchmarkConfig(num_episodes=10),
)
results = runner.evaluate()
```

---

## Video Recording

`VideoRecorder` (numpy + imageio, no torch) stays in `physicalai-train` for now. Reasons:

1. **imageio is not a runtime dependency** — adding it to the lightweight distribution increases install size for a feature most edge deployers don't need
2. **No clear runtime use case** — edge benchmarks care about latency numbers, not video files
3. **Easy to move later** — if a runtime use case emerges, `VideoRecorder` has zero torch deps and can be extracted with a lazy imageio import

If `VideoRecorder` moves to runtime in the future, it would be gated behind an optional dependency:

```python
# Hypothetical future: physicalai/benchmark/_video.py
def _get_imageio():
    try:
        import imageio
        return imageio
    except ImportError:
        msg = "Video recording requires imageio. Install with: pip install physicalai[video]"
        raise ImportError(msg)
```

---

## Migration Plan

### Phase 1: Create Runtime Benchmark Module

1. Create `physicalai/benchmark/` with protocols, runner, results, latency
2. Copy `BenchmarkResults` and `TaskResult` unchanged from `getiaction.benchmark.results`
3. Implement `BenchmarkRunner` with the evaluation loop (new code, not a copy of the torch-dependent `evaluate_policy`)
4. Add `LatencyStats`
5. **Validation**: Import `physicalai.benchmark` with only numpy installed — no torch import errors

### Phase 2: Create Training Adapters

1. Create `TorchPolicyAdapter` and `GymEnvAdapter` in `physicalai-train`
2. Create `LiberoBenchmark` preset that composes `BenchmarkRunner` with adapters
3. **Validation**: Existing training benchmark workflow produces identical results

### Phase 3: Deprecate Old API

1. Add deprecation warnings to `getiaction.benchmark.Benchmark.evaluate()`
2. Point users to `physicalai.benchmark.BenchmarkRunner` or `physicalai.benchmark.libero.LiberoBenchmark`
3. Remove after one release cycle

### Import Boundary CI Rule

```yaml
# CI check: physicalai (runtime) must never import torch
- name: Verify runtime import boundary
  run: |
    python -c "
    import sys
    import physicalai.benchmark
    assert 'torch' not in sys.modules, 'Runtime benchmark imported torch!'
    "
```

---

## Open Design Decisions

| Decision                          | Current Stance                        | Revisit When                                        |
| --------------------------------- | ------------------------------------- | --------------------------------------------------- |
| VideoRecorder placement           | `physicalai-train`                    | Runtime use case emerges                            |
| Batched/vectorized envs           | Not supported                         | Performance bottleneck from sequential env stepping |
| Observation schema validation     | Not included                          | Users report schema mismatch debugging pain         |
| `InferenceModel.predict()` naming | `predict()` + `select_action()` alias | Broader API review of InferenceModel                |

---

## References

- [Packaging Strategy](../packaging/physical-ai-two-repo-options.md) — Two-distribution PEP 420 namespace split
- [Robot Interface](./robot-interface.md) — `get_observation() → dict[str, Any]` pattern
- [Camera Interface](./camera-interface.md) — `Frame.data → np.ndarray` pattern
- [Gymnasium API](https://gymnasium.farama.org/api/env/) — Standard environment interface
- [HuggingFace evaluate](https://github.com/huggingface/evaluate) — Lightweight evaluation package precedent

---

_Last Updated: 2026-02-20_
