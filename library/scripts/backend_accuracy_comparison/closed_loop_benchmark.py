#!/usr/bin/env python3
"""Closed-Loop Benchmark: OpenVINO vs PyTorch Accuracy Comparison.

Runs a FULL closed-loop simulation in LiberoGym + MuJoCo:
- Each action affects the next observation.
- Policy errors accumulate over the whole episode (up to 520 steps).
- Success rate measures real production accuracy.

This is the CORRECT way to compare accuracy because it captures:
  ✓ Error accumulation over time.
  ✓ Interaction with the physics simulator.
  ✓ Final task completion (did the robot achieve the goal?).

Usage:
    python closed_loop_benchmark.py \\
        --checkpoint checkpoints/pi05_libero.ckpt \\
        --policy-class physicalai.policies.pi05.Pi05 \\
        --task-suite libero_10 \\
        --task-ids 0 1 2 \\
        --num-episodes 20 \\
        --seed 42

Example output:

    PyTorch  Success Rate: 75.0% (45/60 episodes)
    OpenVINO Success Rate: 74.5% (44.7/60 episodes)
    Delta: 0.5% (within measurement noise)
    Conclusion: Backends are equivalent
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Silence verbose third-party loggers (ONNX optimizer prints thousands of
# "Replaced initializer 'val_X' with existing initializer 'val_Y'" lines
# during torch → ONNX → OpenVINO export).
for _noisy in (
    "onnxscript",
    "onnxscript.optimizer",
    "onnx",
    "onnxoptimizer",
    "openvino",
    "nncf",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _require_hf_token() -> None:
    """Fail fast if no HuggingFace token is available.

    Checks (canonical HF resolution order):
      1. HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variables
      2. ~/.cache/huggingface/token (set by `huggingface-cli login`)

    Exits with code 2 and an actionable message if neither is present.
    On success, propagates the token into HF_TOKEN so downstream libraries see it.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
        except Exception:
            token = None
    if not token:
        sys.stderr.write(
            "ERROR: No HuggingFace token found.\n"
            "  Gated models required by this script:\n"
            "    - lerobot/pi05_libero_finetuned_v044\n"
            "    - google/paligemma-3b-pt-224\n"
            "  Fix (one-time):\n"
            "    huggingface-cli login\n"
            "  Or export HF_TOKEN=<your_token> before running.\n"
        )
        sys.exit(2)
    os.environ["HF_TOKEN"] = token


@dataclass
class TaskResult:
    """Aggregated results for a single LIBERO task."""

    task_id: str
    num_episodes: int
    success_rate: float
    avg_reward: float
    avg_episode_length: float
    avg_fps: float
    episodes: list[dict[str, Any]]


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for a single backend."""

    backend: str
    overall_success_rate: float
    total_episodes: int
    successful_episodes: int
    task_results: list[TaskResult]
    export_time: float
    total_time: float


def load_policy(checkpoint_path: str, policy_class_path: str):
    """Load a policy from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file.
        policy_class_path: Fully-qualified policy class path,
            e.g. "physicalai.policies.pi05.Pi05".

    Returns:
        Policy instance in eval mode.
    """
    # Dynamic import of the policy class.
    module_path, class_name = policy_class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    policy_cls = getattr(module, class_name)

    logger.info(f"Loading policy from {checkpoint_path}")
    policy = policy_cls.load_from_checkpoint(checkpoint_path)
    policy.eval()

    logger.info(f"Policy loaded: {type(policy).__name__}")
    logger.info(f"Supported export backends: {policy.get_supported_export_backends()}")

    return policy


def export_policy(policy, backend: str, output_dir: Path, force: bool = False):
    """Export a policy to the given backend, caching by manifest.json presence.

    Args:
        policy: Policy instance to export.
        backend: "torch" or "openvino".
        output_dir: Output directory (a `<backend>/` subdir is created).
        force: Re-export even if cached artifacts exist.

    Returns:
        Tuple of (export_path, elapsed_seconds). `elapsed_seconds` is 0.0 when cache is reused.
    """
    export_path = output_dir / backend
    export_path.mkdir(parents=True, exist_ok=True)

    # Skip if manifest.json (written at the end of a successful export) is present.
    manifest = export_path / "manifest.json"
    if manifest.exists() and not force:
        logger.info(f"✓ Reusing cached {backend} export at {export_path} (use --force-export to re-export)")
        return export_path, 0.0

    start = time.perf_counter()

    if backend == "torch":
        policy.export(export_path, backend="torch")
    elif backend == "openvino":
        policy.export(export_path, backend="openvino")
    else:
        msg = f"Unknown backend: {backend}"
        raise ValueError(msg)

    elapsed = time.perf_counter() - start
    logger.info(f"✓ Exported {backend} in {elapsed:.2f}s to {export_path}")

    return export_path, elapsed


def run_single_episode(gym, policy, max_steps: int, seed: int) -> dict[str, Any]:
    """Run a single closed-loop episode in the simulator.

    Args:
        gym: LiberoGym instance.
        policy: Policy (PyTorch) or InferenceModel (any backend).
        max_steps: Maximum number of steps.
        seed: Random seed. Applied to torch/numpy RNG as well as the
            simulator so the same (seed, episode) pair is reproducible
            across backends — important for stochastic policies (Pi0.5
            flow matching) so PyTorch and OpenVINO episodes see the
            same noise prior.

    Returns:
        Dict with episode metrics.
    """
    # Seed RNGs before each episode so backends see identical noise.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Reset environment and policy.
    obs, info = gym.reset(seed=seed)
    policy.reset()

    # Metrics
    total_reward = 0.0
    success = False
    step = 0
    start_time = time.perf_counter()

    for step in range(max_steps):
        # InferenceModel preprocessors (e.g. StatsNormalizer for the OpenVINO
        # export) call ``dict(inputs)``, which triggers Python's mapping
        # protocol on ``Observation`` — ``obs.keys()`` returns string field
        # names, but ``Observation.__getitem__`` only supports int/slice batch
        # indexing and crashes on string keys. Convert to a plain dict (and
        # drop ``None`` fields the policy doesn't use) before invoking the
        # policy so both backends see the same input shape.

        obs_dict = {k: v for k, v in obs.items() if v is not None}
        # InferenceModel preprocessors (e.g. Pi05Preprocessor) expect numpy
        # arrays — LiberoGym returns torch.Tensors (including nested dicts
        # like ``images``: {camera: Tensor}), so convert recursively here
        # so both backends see the same numpy inputs.
        def _to_numpy(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy()
            if isinstance(v, dict):
                return {k: _to_numpy(x) for k, x in v.items()}
            return v

        obs_dict = {k: _to_numpy(v) for k, v in obs_dict.items()}

        # select_action() uses the internal ActionCursor to dispense one
        # action per call, re-predicting at chunk boundaries automatically.
        with torch.no_grad():
            action = policy.select_action(obs_dict)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        # Drop leading batch dim if present.
        if action.ndim >= 2 and action.shape[0] == 1:
            action = action[0]

        # Step the simulator — the key part:
        # MuJoCo applies the action, updates the world state, renders new
        # camera images, checks collisions, etc.
        obs, reward, terminated, truncated, info = gym.step(action)

        # Accumulate metrics
        total_reward += float(reward)

        # Check success
        if info.get("is_success", False):
            success = True

        # Check termination
        if terminated or truncated:
            break

    elapsed = time.perf_counter() - start_time
    episode_length = step + 1
    fps = episode_length / elapsed if elapsed > 0 else 0

    return {
        "episode_length": episode_length,
        "total_reward": total_reward,
        "success": success,
        "fps": fps,
        "elapsed_time": elapsed,
    }


def evaluate_backend(
    policy,
    task_suite: str,
    task_ids: list[int],
    num_episodes: int,
    max_steps: int,
    seed: int,
    backend_name: str,
) -> BenchmarkResult:
    """Evaluate a policy on the given LIBERO tasks.

    Args:
        policy: Policy or InferenceModel to evaluate.
        task_suite: LIBERO task suite name (e.g. "libero_10").
        task_ids: Task IDs to evaluate.
        num_episodes: Number of episodes per task.
        max_steps: Maximum steps per episode.
        seed: Random seed (each episode gets a unique derived seed).
        backend_name: Backend name used in log messages.

    Returns:
        BenchmarkResult with aggregated metrics.
    """
    from physicalai.gyms import LiberoGym

    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating {backend_name} Backend")
    logger.info(f"{'='*70}")

    task_results = []
    total_success = 0
    total_episodes_count = 0
    benchmark_start = time.perf_counter()

    for task_id in task_ids:
        logger.info(f"\n--- Task {task_suite}_{task_id} ---")

        # Build a gym for this task.
        gym = LiberoGym(
            task_suite=task_suite,
            task_id=task_id,
            observation_height=256,
            observation_width=256,
        )

        logger.info(f"Task name: {gym.task_name}")

        # Run all episodes for this task.
        episodes = []
        for ep_idx in range(num_episodes):
            episode_seed = seed + task_id * 1000 + ep_idx
            result = run_single_episode(gym, policy, max_steps, episode_seed)
            episodes.append(result)

            status = "✓" if result["success"] else "✗"
            logger.info(
                f"  Episode {ep_idx + 1}/{num_episodes}: {status} "
                f"reward={result['total_reward']:.3f}, "
                f"steps={result['episode_length']}, "
                f"fps={result['fps']:.1f}"
            )

        # Aggregate per-task metrics.
        success_count = sum(ep["success"] for ep in episodes)
        success_rate = (success_count / num_episodes) * 100
        avg_reward = np.mean([ep["total_reward"] for ep in episodes])
        avg_length = np.mean([ep["episode_length"] for ep in episodes])
        avg_fps = np.mean([ep["fps"] for ep in episodes])

        task_result = TaskResult(
            task_id=f"{task_suite}_{task_id}",
            num_episodes=num_episodes,
            success_rate=success_rate,
            avg_reward=avg_reward,
            avg_episode_length=avg_length,
            avg_fps=avg_fps,
            episodes=episodes,
        )
        task_results.append(task_result)

        total_success += success_count
        total_episodes_count += num_episodes

        logger.info(
            f"\nTask Summary: success_rate={success_rate:.1f}%, "
            f"avg_reward={avg_reward:.3f}, avg_fps={avg_fps:.1f}"
        )

        gym.close()

    # Aggregate across all tasks.
    overall_success_rate = (total_success / total_episodes_count) * 100
    total_time = time.perf_counter() - benchmark_start

    logger.info(f"\n{'='*70}")
    logger.info(
        f"{backend_name} Overall: {overall_success_rate:.1f}% "
        f"({total_success}/{total_episodes_count} episodes)"
    )
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"{'='*70}\n")

    return BenchmarkResult(
        backend=backend_name,
        overall_success_rate=overall_success_rate,
        total_episodes=total_episodes_count,
        successful_episodes=total_success,
        task_results=task_results,
        export_time=0.0,  # Filled in by the caller.
        total_time=total_time,
    )


def compare_results(pytorch_result: BenchmarkResult, openvino_result: BenchmarkResult) -> dict:
    """Compare results of both backends.

    Args:
        pytorch_result: PyTorch benchmark result.
        openvino_result: OpenVINO benchmark result.

    Returns:
        Dict with deltas and an equivalence verdict.
    """
    success_rate_delta = abs(
        pytorch_result.overall_success_rate - openvino_result.overall_success_rate
    )

    # Per-task comparison
    per_task_deltas = []
    for pt_task, ov_task in zip(pytorch_result.task_results, openvino_result.task_results):
        delta = abs(pt_task.success_rate - ov_task.success_rate)
        per_task_deltas.append(
            {
                "task_id": pt_task.task_id,
                "pytorch_success": pt_task.success_rate,
                "openvino_success": ov_task.success_rate,
                "delta": delta,
            }
        )

    # Determine equivalence.
    # Threshold: a 5% absolute delta is within measurement noise for small
    # numbers of episodes; tighten this if running many episodes.
    threshold = 5.0
    is_equivalent = success_rate_delta < threshold

    if is_equivalent:
        conclusion = (
            f"✅ Backends are EQUIVALENT (delta={success_rate_delta:.1f}% < {threshold}%)"
        )
    else:
        conclusion = (
            f"⚠️  Backends show DIFFERENCE (delta={success_rate_delta:.1f}% >= {threshold}%)"
        )

    return {
        "overall_success_rate_delta": success_rate_delta,
        "per_task_deltas": per_task_deltas,
        "is_equivalent": is_equivalent,
        "conclusion": conclusion,
        "threshold": threshold,
    }


def _eval_cache_key(backend: str, args: argparse.Namespace) -> str:
    """Build a stable cache key from evaluation-determining args.

    Includes only parameters that change rollout results: backend, checkpoint,
    policy class, task suite/ids, episode count, max steps, and seed. For the
    OpenVINO backend the device string is included too, since it affects the
    timing metrics stored in the cached result.
    """
    payload = {
        "backend": backend,
        "checkpoint": str(args.checkpoint),
        "policy_class": args.policy_class,
        "task_suite": args.task_suite,
        "task_ids": list(args.task_ids),
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }
    if backend.lower() == "openvino":
        payload["ov_device"] = args.ov_device
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob, usedforsecurity=False).hexdigest()[:16]


def _cache_path(cache_dir: Path, backend: str, args: argparse.Namespace) -> Path:
    return cache_dir / f"{backend.lower()}_{_eval_cache_key(backend, args)}.json"


def _load_cached_result(path: Path) -> BenchmarkResult | None:
    """Load a cached BenchmarkResult from JSON, or return None if missing."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        task_results = [TaskResult(**tr) for tr in data["task_results"]]
        data["task_results"] = task_results
        return BenchmarkResult(**data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to load eval cache {path}: {e}")
        return None


def _save_cached_result(result: BenchmarkResult, path: Path) -> None:
    """Persist a BenchmarkResult to JSON for later reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info(f"✓ Cached {result.backend} eval results to {path}")


def save_results(
    pytorch_result: BenchmarkResult,
    openvino_result: BenchmarkResult,
    comparison: dict,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    """Save benchmark results to JSON.

    Args:
        pytorch_result: PyTorch benchmark result.
        openvino_result: OpenVINO benchmark result.
        comparison: Output of :func:`compare_results`.
        output_path: JSON output file path.
        args: Parsed CLI arguments.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "checkpoint": str(args.checkpoint),
            "policy_class": args.policy_class,
            "task_suite": args.task_suite,
            "task_ids": args.task_ids,
            "num_episodes_per_task": args.num_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "pytorch": {
            "overall_success_rate": pytorch_result.overall_success_rate,
            "total_episodes": pytorch_result.total_episodes,
            "successful_episodes": pytorch_result.successful_episodes,
            "export_time": pytorch_result.export_time,
            "total_time": pytorch_result.total_time,
            "task_results": [
                {
                    "task_id": tr.task_id,
                    "success_rate": tr.success_rate,
                    "avg_reward": tr.avg_reward,
                    "avg_episode_length": tr.avg_episode_length,
                    "avg_fps": tr.avg_fps,
                }
                for tr in pytorch_result.task_results
            ],
        },
        "openvino": {
            "overall_success_rate": openvino_result.overall_success_rate,
            "total_episodes": openvino_result.total_episodes,
            "successful_episodes": openvino_result.successful_episodes,
            "export_time": openvino_result.export_time,
            "total_time": openvino_result.total_time,
            "task_results": [
                {
                    "task_id": tr.task_id,
                    "success_rate": tr.success_rate,
                    "avg_reward": tr.avg_reward,
                    "avg_episode_length": tr.avg_episode_length,
                    "avg_fps": tr.avg_fps,
                }
                for tr in openvino_result.task_results
            ],
        },
        "comparison": comparison,
    }

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_path}")


def print_summary(
    pytorch_result: BenchmarkResult,
    openvino_result: BenchmarkResult,
    comparison: dict,
) -> None:
    """Print a human-readable summary to stdout.

    Args:
        pytorch_result: PyTorch benchmark result.
        openvino_result: OpenVINO benchmark result.
        comparison: Output of :func:`compare_results`.
    """
    print("\n" + "=" * 80)
    print("CLOSED-LOOP BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Backend':<15} {'Success Rate':>15} {'Episodes':>12} {'Export (s)':>12} {'Total (s)':>12}")
    print("-" * 80)

    print(
        f"{'PyTorch':<15} {pytorch_result.overall_success_rate:>14.1f}% "
        f"{pytorch_result.successful_episodes:>5}/{pytorch_result.total_episodes:<5} "
        f"{pytorch_result.export_time:>12.2f} {pytorch_result.total_time:>12.1f}"
    )

    print(
        f"{'OpenVINO':<15} {openvino_result.overall_success_rate:>14.1f}% "
        f"{openvino_result.successful_episodes:>5}/{openvino_result.total_episodes:<5} "
        f"{openvino_result.export_time:>12.2f} {openvino_result.total_time:>12.1f}"
    )

    print("-" * 80)
    print(f"{'Delta':<15} {comparison['overall_success_rate_delta']:>14.1f}%")
    print("=" * 80)

    print(f"\n{comparison['conclusion']}")

    print("\nPer-Task Breakdown:")
    print(f"{'Task ID':<20} {'PyTorch':>12} {'OpenVINO':>12} {'Delta':>10}")
    print("-" * 60)
    for task_delta in comparison["per_task_deltas"]:
        print(
            f"{task_delta['task_id']:<20} "
            f"{task_delta['pytorch_success']:>11.1f}% "
            f"{task_delta['openvino_success']:>11.1f}% "
            f"{task_delta['delta']:>9.1f}%"
        )

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Closed-Loop Benchmark: OpenVINO vs PyTorch Accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (.ckpt)",
    )

    parser.add_argument(
        "--policy-class",
        type=str,
        required=True,
        help="Fully qualified policy class path (e.g., 'physicalai.policies.ACT')",
    )

    parser.add_argument(
        "--task-suite",
        type=str,
        default="libero_10",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite",
    )

    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific task IDs to evaluate (default: all tasks in suite)",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Number of episodes per task (more episodes = more reliable)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per episode (default: suite-specific, e.g., 520 for libero_10)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/closed_loop_results.json"),
        help="Output JSON file for results",
    )

    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("exports"),
        help="Directory for exported models",
    )

    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export models even if cached artifacts exist in --export-dir",
    )

    parser.add_argument(
        "--eval-cache-dir",
        type=Path,
        default=Path("results/eval_cache"),
        help="Directory for cached per-backend evaluation results",
    )

    parser.add_argument(
        "--no-eval-cache",
        action="store_true",
        help="Disable evaluation result caching (always re-run rollouts)",
    )

    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch rollouts and require a cached result (useful when iterating on OpenVINO)",
    )

    parser.add_argument(
        "--ov-device",
        type=str,
        default="CPU",
        help=(
            "OpenVINO device string passed to ov.Core().compile_model(). "
            "Examples: 'CPU', 'GPU', 'NPU', 'AUTO', 'AUTO:GPU,CPU', 'HETERO:GPU,CPU'. "
            "See https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html"
        ),
    )

    args = parser.parse_args()

    # Fail fast before any heavy work (model download / export / simulator init).
    _require_hf_token()

    # Set deterministic seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Starting Closed-Loop Backend Comparison")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Policy: {args.policy_class}")
    logger.info(f"Task suite: {args.task_suite}")
    logger.info(f"Task IDs: {args.task_ids or 'all'}")
    logger.info(f"Episodes per task: {args.num_episodes}")
    logger.info(f"Seed: {args.seed}")

    # 1. Load policy
    policy = load_policy(args.checkpoint, args.policy_class)

    # Determine max_steps if not provided
    if args.max_steps is None:
        from physicalai.benchmark.libero.libero import LiberoMaxSteps

        args.max_steps = getattr(
            LiberoMaxSteps, args.task_suite, LiberoMaxSteps.DEFAULT
        ).value
        logger.info(f"Using default max_steps for {args.task_suite}: {args.max_steps}")

    # Determine task_ids if not provided
    if args.task_ids is None:
        # Get all tasks in suite
        from physicalai.gyms.libero import LiberoGym

        test_gym = LiberoGym(task_suite=args.task_suite, task_id=0)
        # Count available tasks
        suite_info = {
            "libero_spatial": 10,
            "libero_object": 10,
            "libero_goal": 10,
            "libero_10": 10,
            "libero_90": 90,
        }
        num_tasks = suite_info.get(args.task_suite, 10)
        args.task_ids = list(range(num_tasks))
        test_gym.close()
        logger.info(f"Evaluating all {num_tasks} tasks in {args.task_suite}")

    # 2. Export to both backends
    logger.info("\nExporting models...")
    pytorch_path, pytorch_export_time = export_policy(policy, "torch", args.export_dir, force=args.force_export)
    openvino_path, openvino_export_time = export_policy(policy, "openvino", args.export_dir, force=args.force_export)

    # 3. Load exported models for inference
    from physicalai.inference import InferenceModel
    from physicalai.inference.runners import ActionChunking, SinglePass

    # The default manifest exports declare a ``SinglePass`` runner, which
    # makes ``InferenceModel.select_action()`` raise (it requires action
    # chunking). Wrap the runner explicitly so both backends queue
    # ``chunk_size`` predicted actions and dispense one per call — this
    # also matches what ``Policy.select_action()`` does in eager PyTorch.
    chunk_size = int(getattr(policy.config, "chunk_size", 1))

    def _make_runner() -> ActionChunking:
        return ActionChunking(SinglePass(), chunk_size=chunk_size)

    use_cache = not args.no_eval_cache
    pytorch_cache = _cache_path(args.eval_cache_dir, "PyTorch", args)
    openvino_cache = _cache_path(args.eval_cache_dir, "OpenVINO", args)

    # 4. Evaluate PyTorch backend (use cache when available)
    pytorch_result = _load_cached_result(pytorch_cache) if use_cache else None
    if pytorch_result is not None:
        logger.info(f"\n✓ Reusing cached PyTorch results from {pytorch_cache}")
    elif args.skip_pytorch:
        msg = (
            f"--skip-pytorch was set but no cached result was found at {pytorch_cache}. "
            "Run once without --skip-pytorch to populate the cache."
        )
        raise SystemExit(msg)
    else:
        logger.info("\nLoading PyTorch exported model...")
        pytorch_model = InferenceModel.load(pytorch_path, runner=_make_runner())
        pytorch_result = evaluate_backend(
            pytorch_model,
            args.task_suite,
            args.task_ids,
            args.num_episodes,
            args.max_steps,
            args.seed,
            "PyTorch",
        )
        pytorch_result.export_time = pytorch_export_time
        if use_cache:
            _save_cached_result(pytorch_result, pytorch_cache)

    # 5. Evaluate OpenVINO backend (use cache when available)
    openvino_result = _load_cached_result(openvino_cache) if use_cache else None
    if openvino_result is not None:
        logger.info(f"\n✓ Reusing cached OpenVINO results from {openvino_cache}")
    else:
        logger.info(f"\nLoading OpenVINO exported model on device={args.ov_device}...")
        openvino_model = InferenceModel.load(openvino_path, device=args.ov_device, runner=_make_runner())
        openvino_result = evaluate_backend(
            openvino_model,
            args.task_suite,
            args.task_ids,
            args.num_episodes,
            args.max_steps,
            args.seed,
            "OpenVINO",
        )
        openvino_result.export_time = openvino_export_time
        if use_cache:
            _save_cached_result(openvino_result, openvino_cache)

    # 6. Compare results
    comparison = compare_results(pytorch_result, openvino_result)

    # 7. Save and print results
    save_results(pytorch_result, openvino_result, comparison, args.output, args)
    print_summary(pytorch_result, openvino_result, comparison)


if __name__ == "__main__":
    main()
