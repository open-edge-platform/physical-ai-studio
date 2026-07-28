# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""Compare native PyTorch ACT vs its OpenVINO export - numerical and closed-loop parity.

Loads a real pretrained LeRobot ACT checkpoint (HF Hub repo id or local dir) via
``physicalai.policies.act.ACT(pretrained_name_or_path=...)``, exports it to OpenVINO,
and demonstrates that the export reproduces the native model's behavior:

  1. Numerical equivalence: ``predict_action_chunk`` outputs on real observations
     sampled from the gym-aloha env (max abs diff / cosine similarity).
  2. Closed-loop equivalence: per-episode success/fail outcomes and success rate on
     gym-aloha's ``AlohaTransferCube-v0`` task, run with the same seeds on both backends.

This is intended as reference evidence for the PR that adds ``pretrained_name_or_path``
checkpoint loading to native ``ACT`` (which already supports OpenVINO/ONNX/ExecuTorch
export via ``ExportablePolicyMixin``), closing the gap where a real ACT checkpoint could
not be both loaded with real weights and exported to OpenVINO in the same code path.

Usage:
    # Export + compare against the official lerobot/act_aloha_sim_transfer_cube_human checkpoint
    uv run python scripts/compare_act_backends.py \\
        --checkpoint lerobot/act_aloha_sim_transfer_cube_human \\
        --export-dir /tmp/act_export_openvino \\
        --n-episodes 10

    # Reuse an existing export, skip re-exporting
    uv run python scripts/compare_act_backends.py \\
        --checkpoint lerobot/act_aloha_sim_transfer_cube_human \\
        --export-dir /tmp/act_export_openvino \\
        --n-episodes 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import gym_aloha  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from physicalai.data.observation import Observation
from physicalai.inference import InferenceModel
from physicalai.policies.act.policy import ACT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ENV_ID = "gym_aloha/AlohaTransferCube-v0"


def set_seed(seed: int) -> None:
    """Seed all RNGs used by the comparison for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def obs_to_observation(obs: dict) -> Observation:
    """Convert a gym-aloha observation into physicalai's native ``Observation``.

    Args:
        obs: Raw observation dict returned by the gym-aloha environment.

    Returns:
        An ``Observation`` with batched state/images tensors, ready for the native policy.
    """
    state = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0)
    images = torch.from_numpy(obs["pixels"]["top"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return Observation(state=state, images=images)


def obs_to_input(obs: dict) -> dict[str, np.ndarray]:
    """Convert a gym-aloha observation into the exported model's expected input dict.

    Args:
        obs: Raw observation dict returned by the gym-aloha environment.

    Returns:
        A dict of batched numpy arrays matching the OpenVINO export's input signature.
    """
    image = obs["pixels"]["top"].astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[None, ...]
    state = obs["agent_pos"].astype(np.float32)[None, ...]
    return {"state": state, "images": image}


def load_native_policy(checkpoint: str) -> ACT:
    """Load the native physicalai ACT policy from a real pretrained checkpoint.

    Args:
        checkpoint: HuggingFace Hub repo id or local directory containing
            ``config.json`` and ``model.safetensors``.

    Returns:
        The loaded ``ACT`` policy in eval mode.
    """
    logger.info("Loading native ACT policy from checkpoint: %s", checkpoint)
    policy = ACT(pretrained_name_or_path=checkpoint)
    policy.eval()
    return policy


def export_to_openvino(policy: ACT, export_dir: Path, *, force: bool, compress_to_fp16: bool) -> Path:
    """Export the native policy to OpenVINO, unless a compatible export already exists.

    Args:
        policy: Loaded native ACT policy.
        export_dir: Destination directory for the exported OpenVINO IR + manifest.
        force: Re-export even if ``export_dir`` already contains a manifest.
        compress_to_fp16: Whether to compress exported weights to FP16. Note: this has
            previously been shown to cause a measurable accuracy regression for policies
            like ACT; default to False for parity validation unless explicitly testing that.

    Returns:
        The export directory.
    """
    manifest = export_dir / "manifest.json"
    if manifest.exists() and not force:
        logger.info("Reusing existing export at %s (pass --force-export to regenerate)", export_dir)
        return export_dir

    logger.info("Exporting to OpenVINO at %s (compress_to_fp16=%s)", export_dir, compress_to_fp16)
    export_dir.mkdir(parents=True, exist_ok=True)
    policy.export(export_dir, backend="openvino", compress_to_fp16=compress_to_fp16)
    return export_dir


def numeric_comparison(
    native_policy: ACT,
    exported_model: InferenceModel,
    env: gym.Env,
    n_samples: int,
    seed_offset: int,
) -> dict[str, Any]:
    """Compare native vs exported ``predict_action_chunk`` outputs on real observations.

    Args:
        native_policy: Loaded native ACT policy (PyTorch).
        exported_model: Loaded OpenVINO ``InferenceModel``.
        env: gym-aloha environment used to sample realistic observations.
        n_samples: Number of independent observations to compare.
        seed_offset: Seed offset for sampling observations (env reset seeds).

    Returns:
        A dict with per-sample and aggregate max-abs-diff / cosine-similarity metrics.
    """
    per_sample = []
    for i in range(n_samples):
        seed = seed_offset + i
        obs, _info = env.reset(seed=seed)

        native_policy.reset()
        with torch.inference_mode():
            native_chunk = native_policy.predict_action_chunk(obs_to_observation(obs))
        native_arr = native_chunk[0].cpu().numpy() if native_chunk.ndim == 3 else native_chunk.cpu().numpy()  # noqa: PLR2004

        exported_chunk = np.asarray(exported_model.predict_action_chunk(obs_to_input(obs)))
        exported_arr = exported_chunk[0] if exported_chunk.ndim == 3 else exported_chunk  # noqa: PLR2004

        diff = np.abs(native_arr - exported_arr)
        native_flat = native_arr.flatten()
        exported_flat = exported_arr.flatten()
        cosine_similarity = float(
            np.dot(native_flat, exported_flat) / (np.linalg.norm(native_flat) * np.linalg.norm(exported_flat) + 1e-12),
        )
        per_sample.append({
            "seed": seed,
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "cosine_similarity": cosine_similarity,
        })
        logger.info(
            f"[numeric {i + 1}/{n_samples}] seed={seed} max_abs_diff={diff.max():.6f} "
            f"cosine_similarity={cosine_similarity:.6f}",
        )

    return {
        "per_sample": per_sample,
        "max_abs_diff": max(s["max_abs_diff"] for s in per_sample),
        "mean_abs_diff": float(np.mean([s["mean_abs_diff"] for s in per_sample])),
        "min_cosine_similarity": min(s["cosine_similarity"] for s in per_sample),
    }


def run_native_episode(env: gym.Env, policy: ACT, seed: int) -> dict[str, Any]:
    """Run one closed-loop episode with the native PyTorch policy.

    Args:
        env: gym-aloha environment.
        policy: Loaded native ACT policy.
        seed: Episode seed.

    Returns:
        A dict with ``success`` and ``steps`` for the episode.
    """
    policy.reset()
    obs, _info = env.reset(seed=seed)
    done = False
    success = False
    steps = 0
    while not done:
        with torch.inference_mode():
            action = policy.select_action(obs_to_observation(obs))
        action_np = action.squeeze(0).cpu().numpy()
        obs, _reward, terminated, truncated, info = env.step(action_np)
        success = success or bool(info.get("is_success", False))
        done = terminated or truncated
        steps += 1
    return {"success": success, "steps": steps}


def run_exported_episode(env: gym.Env, model: InferenceModel, seed: int, n_action_steps: int) -> dict[str, Any]:
    """Run one closed-loop episode with the OpenVINO-exported model.

    Args:
        env: gym-aloha environment.
        model: Loaded OpenVINO ``InferenceModel``.
        seed: Episode seed.
        n_action_steps: Number of actions to consume per predicted chunk before re-planning.

    Returns:
        A dict with ``success`` and ``steps`` for the episode.
    """
    obs, _info = env.reset(seed=seed)
    success = False
    steps = 0
    max_steps = getattr(env.spec, "max_episode_steps", None) or 10_000
    action_queue: list[np.ndarray] = []

    while steps < max_steps:
        if not action_queue:
            chunk = np.asarray(model.predict_action_chunk(obs_to_input(obs)))
            if chunk.ndim == 3:  # noqa: PLR2004
                chunk = chunk[0]
            action_queue = list(chunk[:n_action_steps])

        action = action_queue.pop(0)
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break

    return {"success": success, "steps": steps}


def closed_loop_comparison(
    native_policy: ACT,
    exported_model: InferenceModel,
    env: gym.Env,
    n_episodes: int,
    seed_offset: int,
    n_action_steps: int,
) -> dict[str, Any]:
    """Run matching closed-loop rollouts for both backends and compare success rates.

    Args:
        native_policy: Loaded native ACT policy.
        exported_model: Loaded OpenVINO ``InferenceModel``.
        env: gym-aloha environment.
        n_episodes: Number of episodes to run per backend.
        seed_offset: Seed offset for episodes (same seeds used for both backends).
        n_action_steps: Action-chunk consumption length for the exported backend.

    Returns:
        A dict with per-episode results for both backends plus aggregate success rates.
    """
    episodes = []
    native_success = 0
    exported_success = 0
    matches = 0
    for ep in range(n_episodes):
        seed = seed_offset + ep
        native_result = run_native_episode(env, native_policy, seed)
        exported_result = run_exported_episode(env, exported_model, seed, n_action_steps)
        native_success += int(native_result["success"])
        exported_success += int(exported_result["success"])
        outcome_matches = native_result["success"] == exported_result["success"]
        matches += int(outcome_matches)
        episodes.append({
            "episode": ep,
            "seed": seed,
            "native": native_result,
            "exported": exported_result,
            "outcome_matches": outcome_matches,
        })
        logger.info(
            f"[episode {ep + 1}/{n_episodes}] seed={seed} "
            f"native_success={native_result['success']} exported_success={exported_result['success']} "
            f"match={outcome_matches}",
        )

    return {
        "episodes": episodes,
        "native_success_rate_pct": native_success / n_episodes * 100,
        "exported_success_rate_pct": exported_success / n_episodes * 100,
        "per_episode_outcome_match_pct": matches / n_episodes * 100,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="lerobot/act_aloha_sim_transfer_cube_human",
                        help="HF Hub repo id or local dir with config.json + model.safetensors")
    parser.add_argument("--export-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "act_export_openvino",
                        help="Directory for the OpenVINO export (reused if it already exists)")
    parser.add_argument("--force-export", action="store_true", help="Re-export even if --export-dir already exists")
    parser.add_argument("--compress-to-fp16", action="store_true",
                        help="Export with FP16 weight compression (known to regress ACT accuracy; off by default)")
    parser.add_argument("--device", default="CPU", help="OpenVINO device for the exported model")
    parser.add_argument("--n-numeric-samples", type=int, default=20, help="Number of single-step samples to compare")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of closed-loop episodes to compare")
    parser.add_argument("--n-action-steps", type=int, default=100, help="Action-chunk consumption length")
    parser.add_argument("--seed-offset", type=int, default=0, help="Seed offset shared by both comparisons")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed for reproducibility")
    parser.add_argument("--max-abs-diff-tolerance", type=float, default=0.05,
                        help="Numeric comparison PASS threshold for max abs diff")
    parser.add_argument("--success-rate-diff-tolerance", type=float, default=15.0,
                        help="PASS threshold (percentage points) for |native_rate - exported_rate|")
    parser.add_argument("--output", type=Path,
                        default=Path(tempfile.gettempdir()) / "compare_act_backends_report.json",
                        help="Destination JSON report path")
    return parser.parse_args()


def main() -> int:
    """Run the full native-vs-OpenVINO ACT comparison and print a PASS/FAIL summary."""
    args = parse_args()
    set_seed(args.seed)

    native_policy = load_native_policy(args.checkpoint)
    export_to_openvino(native_policy, args.export_dir, force=args.force_export, compress_to_fp16=args.compress_to_fp16)
    exported_model = InferenceModel(str(args.export_dir), device=args.device)

    env = gym.make(ENV_ID, obs_type="pixels_agent_pos")

    start = time.time()
    numeric_results = numeric_comparison(native_policy, exported_model, env, args.n_numeric_samples, args.seed_offset)
    closed_loop_results = closed_loop_comparison(
        native_policy, exported_model, env, args.n_episodes, args.seed_offset, args.n_action_steps,
    )
    elapsed = time.time() - start

    success_rate_diff = abs(
        closed_loop_results["native_success_rate_pct"] - closed_loop_results["exported_success_rate_pct"],
    )
    numeric_pass = numeric_results["max_abs_diff"] <= args.max_abs_diff_tolerance
    closed_loop_pass = success_rate_diff <= args.success_rate_diff_tolerance
    overall_pass = numeric_pass and closed_loop_pass

    report = {
        "checkpoint": args.checkpoint,
        "export_dir": str(args.export_dir),
        "compress_to_fp16": args.compress_to_fp16,
        "elapsed_sec": elapsed,
        "numeric_comparison": numeric_results,
        "closed_loop_comparison": closed_loop_results,
        "verdict": {
            "numeric_pass": numeric_pass,
            "closed_loop_pass": closed_loop_pass,
            "overall_pass": overall_pass,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 72)
    logger.info(f"Numeric:     max_abs_diff={numeric_results['max_abs_diff']:.6f} "
                f"(tolerance {args.max_abs_diff_tolerance}) -> {'PASS' if numeric_pass else 'FAIL'}")
    logger.info(f"Numeric:     min_cosine_similarity={numeric_results['min_cosine_similarity']:.6f}")
    logger.info(f"Closed-loop: native={closed_loop_results['native_success_rate_pct']:.1f}%  "
                f"exported={closed_loop_results['exported_success_rate_pct']:.1f}%  "
                f"per_episode_match={closed_loop_results['per_episode_outcome_match_pct']:.1f}% "
                f"-> {'PASS' if closed_loop_pass else 'FAIL'} (tolerance {args.success_rate_diff_tolerance}pp)")
    logger.info(f"Overall: {'PASS' if overall_pass else 'FAIL'}  (report: {args.output})")
    logger.info("=" * 72)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
