#!/usr/bin/env python3
"""Batch Comparison Script - Compare Multiple Models.

Runs both benchmarks for several models and generates a combined summary table.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_comparison(
    checkpoint: str,
    policy_class: str,
    model_name: str,
    task_suite: str,
    task_ids: list[int],
    num_episodes: int,
    dataset: str,
    num_samples: int,
    output_dir: Path,
) -> dict:
    """Run both benchmarks for a single model.

    Returns:
        Dict with results from both benchmarks.
    """
    results = {"model_name": model_name}

    print(f"\n{'=' * 70}")
    print(f"Processing: {model_name}")
    print(f"{'=' * 70}")

    # 1. Numerical comparison
    numerical_output = output_dir / f"{model_name}_numerical.json"
    print(f"\n1. Running numerical comparison...")

    cmd = [
        "python",
        "numerical_comparison.py",
        "--checkpoint",
        checkpoint,
        "--policy-class",
        policy_class,
        "--dataset",
        dataset,
        "--num-samples",
        str(num_samples),
        "--output",
        str(numerical_output),
    ]

    try:
        subprocess.run(cmd, check=True)
        with numerical_output.open() as f:
            numerical_data = json.load(f)

        results["numerical"] = {
            "max_diff": numerical_data["comparison"]["overall_max_diff"],
            "mean_diff": numerical_data["comparison"]["overall_mean_diff"],
            "is_equivalent": numerical_data["analysis"]["is_equivalent"],
        }
        print(f"   ✓ Numerical: max_diff={results['numerical']['max_diff']:.8f}")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Numerical failed: {e}")
        results["numerical"] = {"error": str(e)}

    # 2. Closed-loop benchmark
    closed_loop_output = output_dir / f"{model_name}_closed_loop.json"
    print(f"\n2. Running closed-loop benchmark...")

    cmd = [
        "python",
        "closed_loop_benchmark.py",
        "--checkpoint",
        checkpoint,
        "--policy-class",
        policy_class,
        "--task-suite",
        task_suite,
        "--task-ids",
        *[str(i) for i in task_ids],
        "--num-episodes",
        str(num_episodes),
        "--output",
        str(closed_loop_output),
    ]

    try:
        subprocess.run(cmd, check=True)
        with closed_loop_output.open() as f:
            closed_loop_data = json.load(f)

        results["closed_loop"] = {
            "pytorch_success": closed_loop_data["pytorch"]["overall_success_rate"],
            "openvino_success": closed_loop_data["openvino"]["overall_success_rate"],
            "delta": closed_loop_data["comparison"]["overall_success_rate_delta"],
            "is_equivalent": closed_loop_data["comparison"]["is_equivalent"],
        }
        print(
            f"   ✓ Closed-loop: PyTorch={results['closed_loop']['pytorch_success']:.1f}%, "
            f"OpenVINO={results['closed_loop']['openvino_success']:.1f}%, "
            f"delta={results['closed_loop']['delta']:.1f}%"
        )
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Closed-loop failed: {e}")
        results["closed_loop"] = {"error": str(e)}

    return results


def print_summary(all_results: list[dict]) -> None:
    """Print summary table."""
    print("\n" + "=" * 100)
    print("BATCH COMPARISON SUMMARY")
    print("=" * 100)

    # Numerical table
    print("\nNumerical Comparison:")
    print(f"{'Model':<20} {'Max Diff':>15} {'Mean Diff':>15} {'Status':>15}")
    print("-" * 70)

    for result in all_results:
        model = result["model_name"]
        numerical = result.get("numerical", {})

        if "error" in numerical:
            print(f"{model:<20} {'ERROR':>15} {'ERROR':>15} {'✗ FAILED':>15}")
        else:
            max_diff = numerical["max_diff"]
            mean_diff = numerical["mean_diff"]
            status = "✓ OK" if numerical["is_equivalent"] else "⚠ CHECK"
            print(f"{model:<20} {max_diff:>15.8f} {mean_diff:>15.8f} {status:>15}")

    # Closed-loop table
    print("\nClosed-Loop Benchmark:")
    print(
        f"{'Model':<20} {'PyTorch SR':>12} {'OpenVINO SR':>12} {'Delta':>10} {'Status':>15}"
    )
    print("-" * 75)

    for result in all_results:
        model = result["model_name"]
        closed_loop = result.get("closed_loop", {})

        if "error" in closed_loop:
            print(f"{model:<20} {'ERROR':>12} {'ERROR':>12} {'ERROR':>10} {'✗ FAILED':>15}")
        else:
            pt_sr = closed_loop["pytorch_success"]
            ov_sr = closed_loop["openvino_success"]
            delta = closed_loop["delta"]
            status = "✓ OK" if closed_loop["is_equivalent"] else "⚠ CHECK"
            print(
                f"{model:<20} {pt_sr:>11.1f}% {ov_sr:>11.1f}% {delta:>9.1f}% {status:>15}"
            )

    print("=" * 100)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch Comparison: Run benchmarks for multiple models"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Models as 'name:checkpoint:policy_class' (e.g., 'act:act.ckpt:physicalai.policies.ACT')",
    )

    parser.add_argument(
        "--task-suite",
        type=str,
        default="libero_10",
        help="LIBERO task suite for closed-loop",
    )

    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Task IDs for closed-loop",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Episodes per task for closed-loop",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="lerobot/libero_10_image",
        help="Dataset for numerical comparison",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Samples for numerical comparison",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse models
    models = []
    for model_spec in args.models:
        parts = model_spec.split(":")
        if len(parts) != 3:
            print(f"Invalid model spec: {model_spec}")
            print("Expected format: name:checkpoint:policy_class")
            sys.exit(1)

        name, checkpoint, policy_class = parts
        models.append(
            {
                "name": name,
                "checkpoint": checkpoint,
                "policy_class": policy_class,
            }
        )

    print(f"Running batch comparison for {len(models)} models:")
    for model in models:
        print(f"  - {model['name']} ({model['policy_class']})")

    # Run comparisons
    all_results = []
    for model in models:
        result = run_comparison(
            checkpoint=model["checkpoint"],
            policy_class=model["policy_class"],
            model_name=model["name"],
            task_suite=args.task_suite,
            task_ids=args.task_ids,
            num_episodes=args.num_episodes,
            dataset=args.dataset,
            num_samples=args.num_samples,
            output_dir=output_dir,
        )
        all_results.append(result)

    # Save aggregate results
    aggregate_output = output_dir / "batch_comparison_summary.json"
    with aggregate_output.open("w") as f:
        json.dump(
            {
                "models": all_results,
                "config": {
                    "task_suite": args.task_suite,
                    "task_ids": args.task_ids,
                    "num_episodes": args.num_episodes,
                    "dataset": args.dataset,
                    "num_samples": args.num_samples,
                },
            },
            f,
            indent=2,
        )

    # Print summary
    print_summary(all_results)

    print(f"\n✓ Batch comparison complete")
    print(f"  Individual results: {output_dir}/")
    print(f"  Summary: {aggregate_output}")


if __name__ == "__main__":
    main()
