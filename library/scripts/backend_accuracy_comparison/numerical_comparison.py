#!/usr/bin/env python3
"""Numerical Comparison: OpenVINO vs PyTorch Single-Step Accuracy.

Compares per-sample predictions on a dataset:
- Each dataset observation is processed independently.
- No simulator involved — only a single forward pass through the policy.
- No error accumulation.
- Measures numerical differences between backends.

This is an EXPORT SANITY CHECK:
  ✓ Verifies the export is numerically lossless.
  ✓ Fast (minutes, not hours).
  ✗ Does NOT measure real-world accuracy (no closed-loop).
  ✗ Does NOT account for error accumulation.

Usage:
    python numerical_comparison.py \\
        --checkpoint checkpoints/pi05.ckpt \\
        --policy-class physicalai.policies.pi05.Pi05 \\
        --dataset lerobot/libero_10_image \\
        --num-samples 100

Example output:
    Max absolute difference: 0.000342
    Mean absolute difference: 0.000019
    P99 difference: 0.000123
    Conclusion: Numerically equivalent (within bfloat16 precision)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

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


def load_policy(checkpoint_path: str, policy_class_path: str):
    """Load a policy from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file.
        policy_class_path: Fully-qualified policy class path.

    Returns:
        Policy instance in eval mode.
    """
    module_path, class_name = policy_class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    policy_cls = getattr(module, class_name)

    logger.info(f"Loading policy from {checkpoint_path}")
    policy = policy_cls.load_from_checkpoint(checkpoint_path)
    policy.eval()

    logger.info(f"Policy loaded: {type(policy).__name__}")
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
    logger.info(f"✓ Exported {backend} in {elapsed:.2f}s")

    return export_path, elapsed


def load_dataset_samples(dataset_name: str, num_samples: int) -> list[dict]:
    """Load samples from a LeRobot dataset.

    Args:
        dataset_name: Dataset repo id (e.g. "lerobot/libero_10_image").
        num_samples: Number of samples to load.

    Returns:
        List of observation dicts ready for inference.
    """
    from physicalai.data import LeRobotDataModule

    logger.info(f"Loading {num_samples} samples from {dataset_name}")

    # Build datamodule. We need ~num_samples observations; the easiest way is
    # to read from the train dataloader (val split defaults to 0.0, which
    # produces an empty val_dataloader).
    datamodule = LeRobotDataModule(
        repo_id=dataset_name,
        train_batch_size=1,
        val_batch_size=1,
        episodes=None,  # All episodes
    )

    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()

    # Collect samples
    samples = []
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        # Convert batch to inference observation format. Keep the batch dim
        # (B=1) because downstream policy preprocessors (e.g. Pi0.5) require
        # 2D state arrays `(B, D)`.
        from physicalai.data.lerobot import FormatConverter

        obs = FormatConverter.to_observation(batch)
        sample_dict = obs.to_numpy().to_dict(flatten=False)

        samples.append(sample_dict)

        if (i + 1) % 10 == 0:
            logger.info(f"  Loaded {i + 1}/{num_samples} samples")

    logger.info(f"✓ Loaded {len(samples)} samples")
    if not samples:
        msg = (
            f"No samples were loaded from '{dataset_name}'. "
            "Check that the dataset has episodes available."
        )
        raise RuntimeError(msg)
    return samples


def compare_single_sample(
    pytorch_model,
    openvino_model,
    sample: dict,
) -> dict:
    """Compare predictions of both backends on a single sample.

    Args:
        pytorch_model: InferenceModel (PyTorch backend).
        openvino_model: InferenceModel (OpenVINO backend).
        sample: Observation dict.

    Returns:
        Dict with per-sample diffs and both raw actions.
    """
    # Reset both models (important for stateful policies).
    pytorch_model.reset()
    openvino_model.reset()

    # Compare full action chunks. Works for all policies after the
    # ActionCursor refactor (predict_action_chunk is universal).
    pytorch_action = pytorch_model.predict_action_chunk(sample)
    openvino_action = openvino_model.predict_action_chunk(sample)

    # Convert to numpy and flatten.
    if isinstance(pytorch_action, torch.Tensor):
        pytorch_action = pytorch_action.cpu().numpy()
    if isinstance(openvino_action, torch.Tensor):
        openvino_action = openvino_action.cpu().numpy()

    # Flatten if action is a chunk (first step is the comparable one).
    if pytorch_action.ndim > 1:
        pytorch_action = pytorch_action.reshape(-1)
    if openvino_action.ndim > 1:
        openvino_action = openvino_action.reshape(-1)

    # Differences
    abs_diff = np.abs(pytorch_action - openvino_action)

    return {
        "max_diff": float(np.max(abs_diff)),
        "mean_diff": float(np.mean(abs_diff)),
        "pytorch_action": pytorch_action.tolist(),
        "openvino_action": openvino_action.tolist(),
    }


def run_comparison(
    pytorch_model,
    openvino_model,
    samples: list[dict],
) -> dict:
    """Run the comparison across all samples.

    Args:
        pytorch_model: PyTorch InferenceModel.
        openvino_model: OpenVINO InferenceModel.
        samples: Dataset samples.

    Returns:
        Dict with aggregated metrics and per-sample diffs.
    """
    logger.info(f"\nComparing {len(samples)} samples...")

    max_diffs = []
    mean_diffs = []
    sample_results = []

    start_time = time.perf_counter()

    for i, sample in enumerate(samples):
        sample_start = time.perf_counter()
        result = compare_single_sample(pytorch_model, openvino_model, sample)
        sample_elapsed = time.perf_counter() - sample_start

        max_diffs.append(result["max_diff"])
        mean_diffs.append(result["mean_diff"])

        # Keep the first 10 samples with full action vectors (for debugging).
        if i < 10:
            sample_results.append(
                {
                    "sample_idx": i,
                    "max_diff": result["max_diff"],
                    "mean_diff": result["mean_diff"],
                    "pytorch_action": result["pytorch_action"],
                    "openvino_action": result["openvino_action"],
                }
            )

        # CPU runs are slow (seconds per sample for Pi0.5); log every sample.
        logger.info(
            f"  Sample {i + 1}/{len(samples)} "
            f"max_diff={result['max_diff']:.6f} "
            f"mean_diff={result['mean_diff']:.6f} "
            f"({sample_elapsed:.2f}s)"
        )

    elapsed = time.perf_counter() - start_time

    # Aggregate statistics
    max_diffs_arr = np.array(max_diffs)
    mean_diffs_arr = np.array(mean_diffs)

    results = {
        "num_samples": len(samples),
        "overall_max_diff": float(np.max(max_diffs_arr)),
        "overall_mean_diff": float(np.mean(mean_diffs_arr)),
        "overall_std_diff": float(np.std(mean_diffs_arr)),
        "p50_max_diff": float(np.percentile(max_diffs_arr, 50)),
        "p95_max_diff": float(np.percentile(max_diffs_arr, 95)),
        "p99_max_diff": float(np.percentile(max_diffs_arr, 99)),
        "per_sample_max_diffs": max_diffs,
        "per_sample_mean_diffs": mean_diffs,
        "sample_results": sample_results,  # First 10 with full actions
        "elapsed_time": elapsed,
    }

    logger.info(f"✓ Comparison complete in {elapsed:.2f}s")

    return results


def analyze_results(results: dict) -> dict:
    """Analyze aggregated diffs and decide whether the export is lossless.

    Args:
        results: Output of :func:`run_comparison`.

    Returns:
        Dict with the human-readable conclusion and thresholds.
    """
    max_diff = results["overall_max_diff"]
    mean_diff = results["overall_mean_diff"]
    p99_diff = results["p99_max_diff"]

    # Thresholds by numeric precision:
    # - float32: ~1e-7 (machine epsilon)
    # - bfloat16: ~1e-3 (typical for VLA / LLM-based policies)
    # - Practical threshold: 1e-3

    threshold_strict = 1e-5  # float32 precision
    threshold_practical = 1e-3  # bfloat16 precision

    if max_diff < threshold_strict:
        level = "exact"
        conclusion = "✅ Numerically IDENTICAL (within float32 precision)"
    elif max_diff < threshold_practical:
        level = "equivalent"
        conclusion = "✅ Numerically EQUIVALENT (within bfloat16 precision)"
    elif max_diff < 0.01:
        level = "acceptable"
        conclusion = "⚠️  Small numerical differences (may be acceptable)"
    else:
        level = "significant"
        conclusion = "❌ SIGNIFICANT numerical differences - investigate export!"

    return {
        "level": level,
        "conclusion": conclusion,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "p99_diff": p99_diff,
        "threshold_strict": threshold_strict,
        "threshold_practical": threshold_practical,
        "is_equivalent": max_diff < threshold_practical,
    }


def save_results(
    comparison_results: dict,
    analysis: dict,
    output_path: Path,
    args: argparse.Namespace,
    export_times: dict,
) -> None:
    """Save results to JSON.

    Args:
        comparison_results: Output of :func:`run_comparison`.
        analysis: Output of :func:`analyze_results`.
        output_path: JSON output file path.
        args: Parsed CLI arguments.
        export_times: {"pytorch": seconds, "openvino": seconds}.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "checkpoint": str(args.checkpoint),
            "policy_class": args.policy_class,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "export_times": export_times,
        "comparison": {
            "num_samples": comparison_results["num_samples"],
            "overall_max_diff": comparison_results["overall_max_diff"],
            "overall_mean_diff": comparison_results["overall_mean_diff"],
            "overall_std_diff": comparison_results["overall_std_diff"],
            "p50_max_diff": comparison_results["p50_max_diff"],
            "p95_max_diff": comparison_results["p95_max_diff"],
            "p99_max_diff": comparison_results["p99_max_diff"],
            "elapsed_time": comparison_results["elapsed_time"],
            "sample_results": comparison_results["sample_results"],
            # Per-sample diffs are needed by the demo notebook for histograms.
            "per_sample_max_diffs": comparison_results["per_sample_max_diffs"],
            "per_sample_mean_diffs": comparison_results["per_sample_mean_diffs"],
        },
        "analysis": analysis,
    }

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_path}")


def print_summary(comparison_results: dict, analysis: dict, export_times: dict) -> None:
    """Print a human-readable summary to stdout.

    Args:
        comparison_results: Output of :func:`run_comparison`.
        analysis: Output of :func:`analyze_results`.
        export_times: {"pytorch": seconds, "openvino": seconds}.
    """
    print("\n" + "=" * 80)
    print("NUMERICAL COMPARISON RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nSamples analyzed: {comparison_results['num_samples']}")
    print(f"Comparison time: {comparison_results['elapsed_time']:.2f}s")

    print(f"\n{'Backend':<15} {'Export Time (s)':>20}")
    print("-" * 40)
    print(f"{'PyTorch':<15} {export_times['pytorch']:>20.2f}")
    print(f"{'OpenVINO':<15} {export_times['openvino']:>20.2f}")

    print(f"\n{'Metric':<25} {'Value':>20}")
    print("-" * 50)
    print(f"{'Max Absolute Diff':<25} {analysis['max_diff']:>20.8f}")
    print(f"{'Mean Absolute Diff':<25} {analysis['mean_diff']:>20.8f}")
    print(f"{'P50 Max Diff':<25} {comparison_results['p50_max_diff']:>20.8f}")
    print(f"{'P95 Max Diff':<25} {comparison_results['p95_max_diff']:>20.8f}")
    print(f"{'P99 Max Diff':<25} {comparison_results['p99_max_diff']:>20.8f}")

    print("\n" + "-" * 80)
    print(f"\n{analysis['conclusion']}")
    print("\n" + "=" * 80)

    print("\nInterpretation:")
    print(f"  • float32 threshold: {analysis['threshold_strict']:.2e}")
    print(f"  • bfloat16 threshold: {analysis['threshold_practical']:.2e}")
    print(f"  • Your max diff: {analysis['max_diff']:.8f}")

    if analysis["is_equivalent"]:
        print("\n✓ Export is LOSSLESS for numerical accuracy")
        print("  → Single-step predictions are equivalent")
        print("  → Run closed_loop_benchmark.py to test with error accumulation")
    else:
        print("\n⚠ Export may have issues:")
        print("  1. Check if your model uses bfloat16 (threshold: 1e-3)")
        print("  2. Inspect sample_results in JSON for patterns")
        print("  3. Run closed_loop_benchmark.py to see if it affects success rate")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Numerical Comparison: OpenVINO vs PyTorch",
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
        help="Fully qualified policy class path",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'lerobot/pusht', 'lerobot/libero_10_image')",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to compare (more = more reliable)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/numerical_results.json"),
        help="Output JSON file",
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

    args = parser.parse_args()

    # Fail fast before any heavy work (model download / export / dataset load).
    _require_hf_token()

    logger.info("Starting Numerical Backend Comparison")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Policy: {args.policy_class}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Num samples: {args.num_samples}")

    # 1. Load policy
    policy = load_policy(args.checkpoint, args.policy_class)

    # 2. Export to both backends
    logger.info("\nExporting models...")
    pytorch_path, pytorch_time = export_policy(policy, "torch", args.export_dir, force=args.force_export)
    openvino_path, openvino_time = export_policy(policy, "openvino", args.export_dir, force=args.force_export)

    export_times = {"pytorch": pytorch_time, "openvino": openvino_time}

    # 3. Load exported models
    from physicalai.inference import InferenceModel

    logger.info("\nLoading exported models...")
    pytorch_model = InferenceModel.load(pytorch_path)
    openvino_model = InferenceModel.load(openvino_path)
    logger.info("✓ Models loaded")

    # 4. Load dataset samples
    samples = load_dataset_samples(args.dataset, args.num_samples)

    # 5. Run comparison
    comparison_results = run_comparison(pytorch_model, openvino_model, samples)

    # 6. Analyze results
    analysis = analyze_results(comparison_results)

    # 7. Save and print
    save_results(comparison_results, analysis, args.output, args, export_times)
    print_summary(comparison_results, analysis, export_times)


if __name__ == "__main__":
    main()
