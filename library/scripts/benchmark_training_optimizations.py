# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001
"""Benchmark training optimizations across all policies: precision and torch.compile.

Runs 4 configurations per policy on a local dataset and reports time-per-step
(excluding warmup), peak GPU memory, and total wall-clock time for each.

Policies tested:
  - ACT:     Pure float32, no pretrained VLM — full benefit from bf16-mixed.
  - SmolVLA: VLM backbone loaded in bfloat16 by default; expert head is float32.
  - Pi0:     Entire model defaults to bfloat16 (config dtype="bfloat16").
  - Pi0.5:   Defaults to float32 (config dtype="float32"), opt-in bfloat16.
  - Groot:   Uses torch.autocast with use_bf16=True by default.

Configurations per policy:
  1. precision=32,        compile=False  (baseline)
  2. precision=bf16-mixed, compile=False
  3. precision=32,        compile=True
  4. precision=bf16-mixed, compile=True

Usage:
    python scripts/benchmark_training_optimizations.py
        [--policies POLICY ...] [--max-steps N] [--batch-size B] [--warmup-steps W]

Examples:
    # All policies
    python scripts/benchmark_training_optimizations.py

    # Specific policies only
    python scripts/benchmark_training_optimizations.py --policies act smolvla

    # Custom steps and batch size
    python scripts/benchmark_training_optimizations.py --policies act --max-steps 200 --batch-size 8
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch.callbacks import Callback

from physicalai.data.lerobot import LeRobotDataModule
from physicalai.train import Trainer

if TYPE_CHECKING:
    from physicalai.policies.base import Policy

logger = logging.getLogger(__name__)

DATASET_PATH = Path.home() / ".cache" / "physicalai" / "datasets" / "pick_and_place"
REPO_ID = "local"  # arbitrary — not used for local loading

AVAILABLE_POLICIES = ("act", "smolvla", "pi0", "pi05", "groot")


def create_policy(name: str, *, compile_model: bool) -> Policy:
    """Create a policy instance by name with compile settings.

    Args:
        name: Policy name (act, smolvla, pi0, pi05, groot).
        compile_model: Whether to enable torch.compile.

    Returns:
        Instantiated policy ready for training.

    Raises:
        ValueError: If the policy name is not recognized.
    """
    if name == "act":
        from physicalai.policies import ACT  # noqa: PLC0415

        return ACT(compile_model=compile_model)

    if name == "smolvla":
        from physicalai.policies import SmolVLA  # noqa: PLC0415

        return SmolVLA(compile_model=compile_model)

    if name == "pi0":
        from physicalai.policies import Pi0  # noqa: PLC0415

        return Pi0(compile_model=compile_model)

    if name == "pi05":
        from physicalai.policies import Pi05  # noqa: PLC0415

        return Pi05(compile_model=compile_model)

    if name == "groot":
        from physicalai.policies import Groot  # noqa: PLC0415

        return Groot(compile_model=compile_model)

    msg = f"Unknown policy: {name}. Available: {AVAILABLE_POLICIES}"
    raise ValueError(msg)


@dataclass
class BenchmarkResult:
    """Stores the timing and memory results for a single benchmark configuration."""

    policy_name: str
    label: str
    precision: str
    compile_model: bool
    total_time_s: float
    warmup_steps: int
    warmup_time_s: float
    steady_steps: int
    steady_step_times: list[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    error: str | None = None

    @property
    def steady_time_per_step_ms(self) -> float:
        """Return the mean steady-state step time in milliseconds."""
        if not self.steady_step_times:
            return 0.0
        return statistics.mean(self.steady_step_times) * 1000

    @property
    def steady_median_ms(self) -> float:
        """Return the median steady-state step time in milliseconds."""
        if not self.steady_step_times:
            return 0.0
        return statistics.median(self.steady_step_times) * 1000

    @property
    def total_steps(self) -> int:
        """Return the total number of steps (warmup + steady)."""
        return self.warmup_steps + self.steady_steps

    def to_dict(self) -> dict[str, Any]:
        """Serialize benchmark result to a dictionary for JSON export.

        Returns:
            Dictionary containing all benchmark metrics suitable for JSON serialization.
        """
        return {
            "policy_name": self.policy_name,
            "label": self.label,
            "precision": self.precision,
            "compile_model": self.compile_model,
            "total_time_s": self.total_time_s,
            "warmup_steps": self.warmup_steps,
            "warmup_time_s": self.warmup_time_s,
            "steady_steps": self.steady_steps,
            "steady_step_times": self.steady_step_times,
            "steady_time_per_step_ms": self.steady_time_per_step_ms,
            "steady_median_ms": self.steady_median_ms,
            "total_steps": self.total_steps,
            "peak_memory_mb": self.peak_memory_mb,
            "error": self.error,
        }


class TimingCallback(Callback):
    """Records per-step wall-clock times, separating warmup from steady-state."""

    def __init__(self, warmup_steps: int = 5) -> None:
        """Initialize the timing callback.

        Args:
            warmup_steps: Number of initial steps to treat as warmup (excluded from steady-state stats).
        """
        self.warmup_steps = warmup_steps
        self.step_start: float = 0.0
        self.step_times: list[float] = []
        self._step_count: int = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # noqa: ANN001, ARG002
        """Record the start time of each training batch."""
        self.step_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001, ARG002
        """Record the elapsed time for the completed training batch."""
        elapsed = time.perf_counter() - self.step_start
        self.step_times.append(elapsed)
        self._step_count += 1

    @property
    def warmup_times(self) -> list[float]:
        """Return step times recorded during the warmup phase."""
        return self.step_times[: self.warmup_steps]

    @property
    def steady_times(self) -> list[float]:
        """Return step times recorded after the warmup phase."""
        return self.step_times[self.warmup_steps :]


def run_benchmark(
    policy_name: str,
    precision: str,
    *,
    compile_model: bool,
    max_steps: int,
    batch_size: int,
    dataset_path: Path,
    warmup_steps: int,
) -> BenchmarkResult:
    """Run a single benchmark configuration for the given policy and settings.

    Args:
        policy_name: Name of the policy to benchmark.
        precision: PyTorch Lightning precision string (e.g. "32" or "bf16-mixed").
        compile_model: Whether to enable torch.compile for the policy.
        max_steps: Total number of training steps to run.
        batch_size: Training batch size.
        dataset_path: Path to the local dataset directory.
        warmup_steps: Number of initial steps to exclude from steady-state timing.

    Returns:
        BenchmarkResult containing timing, memory, and error information.
    """
    label = f"precision={precision}, compile={compile_model}"
    logger.info("\n%s", "=" * 70)
    logger.info("  [%s] %s", policy_name.upper(), label)
    logger.info("  (%d warmup + %d measured steps)", warmup_steps, max_steps - warmup_steps)
    logger.info("%s", "=" * 70)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        datamodule = LeRobotDataModule(
            repo_id=REPO_ID,
            root=str(dataset_path),
            train_batch_size=batch_size,
            data_format="physicalai",
        )

        policy = create_policy(policy_name, compile_model=compile_model)

        timer = TimingCallback(warmup_steps=warmup_steps)

        trainer = Trainer(
            max_steps=max_steps,
            precision=precision,
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            callbacks=[timer],
        )

        start = time.perf_counter()
        trainer.fit(policy, datamodule)
        total_time = time.perf_counter() - start

        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        warmup_time = sum(timer.warmup_times)

        return BenchmarkResult(
            policy_name=policy_name,
            label=label,
            precision=precision,
            compile_model=compile_model,
            total_time_s=total_time,
            warmup_steps=len(timer.warmup_times),
            warmup_time_s=warmup_time,
            steady_steps=len(timer.steady_times),
            steady_step_times=timer.steady_times,
            peak_memory_mb=peak_mem,
        )
    except Exception as e:
        logger.exception("  ERROR")
        return BenchmarkResult(
            policy_name=policy_name,
            label=label,
            precision=precision,
            compile_model=compile_model,
            total_time_s=0.0,
            warmup_steps=0,
            warmup_time_s=0.0,
            steady_steps=0,
            peak_memory_mb=0.0,
            error=str(e),
        )
    finally:
        # Clean up GPU memory between runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_policy_results(policy_name: str, results: list[BenchmarkResult]) -> None:
    """Print benchmark results for a single policy."""
    logger.info("\n%s", "=" * 100)
    logger.info("  %s — BENCHMARK RESULTS (steady-state, excluding warmup)", policy_name.upper())
    logger.info("%s", "=" * 100)

    header = (
        f"{'Configuration':<40} {'Steps':>6} {'Mean':>10} {'Median':>10} {'Warmup':>10} {'Total':>10} {'Peak Mem':>10}"
    )
    logger.info(header)
    logger.info("%s", "-" * 100)

    baseline = results[0] if results else None

    for r in results:
        if r.error:
            logger.info("%s %s   %s", f"{r.label:<40}", f"{'FAILED':>6}", r.error)
            continue

        speedup = ""
        if baseline and r is not baseline and not baseline.error and baseline.steady_time_per_step_ms > 0:
            ratio = baseline.steady_time_per_step_ms / r.steady_time_per_step_ms
            speedup = f" ({ratio:.2f}x)"

        mem_str = f"{r.peak_memory_mb:.0f} MB" if r.peak_memory_mb > 0 else "N/A"
        logger.info(
            "%s %s %s %s %s %s %s%s",
            f"{r.label:<40}",
            f"{r.steady_steps:>6}",
            f"{r.steady_time_per_step_ms:>7.1f} ms",
            f"{r.steady_median_ms:>7.1f} ms",
            f"{r.warmup_time_s:>8.1f} s",
            f"{r.total_time_s:>8.1f} s",
            f"{mem_str:>10}",
            speedup,
        )

    logger.info("%s", "=" * 100)

    # Print warmup details for compile runs
    compile_results = [r for r in results if r.compile_model and not r.error]
    if compile_results:
        logger.info("\n  WARMUP DETAILS (torch.compile):")
        for r in compile_results:
            if r.warmup_steps > 0:
                warmup_avg = (r.warmup_time_s / r.warmup_steps) * 1000
                logger.info("    %s", r.label)
                logger.info(
                    "      Warmup: %d steps, %.1fs total, %.0f ms/step avg",
                    r.warmup_steps,
                    r.warmup_time_s,
                    warmup_avg,
                )
                if r.steady_steps > 0:
                    logger.info(
                        "      Steady: %d steps, %.1f ms/step avg",
                        r.steady_steps,
                        r.steady_time_per_step_ms,
                    )


def print_summary(all_results: dict[str, list[BenchmarkResult]]) -> None:
    """Print cross-policy comparison summary."""
    logger.info("\n%s", "#" * 100)
    logger.info("  CROSS-POLICY SUMMARY (best config per policy)")
    logger.info("%s", "#" * 100)

    header = f"{'Policy':<12} {'Best Config':<40} {'Mean ms/step':>12} {'Peak Mem':>10} {'vs Baseline':>12}"
    logger.info(header)
    logger.info("%s", "-" * 100)

    for policy_name, results in all_results.items():
        valid = [r for r in results if not r.error and r.steady_steps > 0]
        if not valid:
            logger.info("%s %s", f"{policy_name:<12}", f"{'ALL FAILED':<40}")
            continue

        baseline = valid[0]
        best = min(valid, key=lambda r: r.steady_time_per_step_ms)
        mem_str = f"{best.peak_memory_mb:.0f} MB" if best.peak_memory_mb > 0 else "N/A"

        speedup = ""
        if baseline.steady_time_per_step_ms > 0:
            ratio = baseline.steady_time_per_step_ms / best.steady_time_per_step_ms
            speedup = f"{ratio:.2f}x"

        logger.info(
            "%s %s %s %s %s",
            f"{policy_name:<12}",
            f"{best.label:<40}",
            f"{best.steady_time_per_step_ms:>9.1f} ms",
            f"{mem_str:>10}",
            f"{speedup:>12}",
        )

    logger.info("%s", "#" * 100)


def save_results(all_results: dict[str, list[BenchmarkResult]], output_path: Path, args: argparse.Namespace) -> None:
    """Save all benchmark metrics to a JSON file."""
    data = {
        "config": {
            "policies": args.policies,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "dataset_path": args.dataset_path,
        },
        "results": {policy_name: [r.to_dict() for r in results] for policy_name, results in all_results.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("\nMetrics saved to %s", output_path)


def main() -> None:
    """Run the benchmark CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Benchmark training optimizations across all policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(AVAILABLE_POLICIES),
        choices=AVAILABLE_POLICIES,
        help=f"Policies to benchmark (default: all). Choices: {', '.join(AVAILABLE_POLICIES)}",
    )
    parser.add_argument("--max-steps", type=int, default=400, help="Total training steps per config (default: 400)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Warmup steps to exclude from timing (default: 20)",
    )
    parser.add_argument("--dataset-path", type=str, default=str(DATASET_PATH), help="Path to local dataset")
    parser.add_argument(
        "--output-path",
        type=str,
        default="benchmark_results.json",
        help="Path to save metrics JSON (default: benchmark_results.json)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error("Error: Dataset not found at %s", dataset_path)
        logger.error("Use --dataset-path to specify the local dataset directory.")
        return

    if args.warmup_steps >= args.max_steps:
        logger.error(
            "Error: warmup-steps (%d) must be less than max-steps (%d)",
            args.warmup_steps,
            args.max_steps,
        )
        return

    configs = [
        ("32", False),
        ("32", True),
        ("bf16-mixed", False),
        ("bf16-mixed", True),
    ]

    all_results: dict[str, list[BenchmarkResult]] = {}

    for policy_name in args.policies:
        logger.info("\n%s", "#" * 70)
        logger.info("  BENCHMARKING: %s", policy_name.upper())
        logger.info("%s", "#" * 70)

        policy_results: list[BenchmarkResult] = []
        for precision, compile_model in configs:
            result = run_benchmark(
                policy_name=policy_name,
                precision=precision,
                compile_model=compile_model,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                dataset_path=dataset_path,
                warmup_steps=args.warmup_steps,
            )
            policy_results.append(result)

        all_results[policy_name] = policy_results
        print_policy_results(policy_name, policy_results)

    # Print cross-policy summary if multiple policies were tested
    if len(all_results) > 1:
        print_summary(all_results)

    save_results(all_results, Path(args.output_path), args)


if __name__ == "__main__":
    main()
