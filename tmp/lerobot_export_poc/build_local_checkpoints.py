# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build self-contained local LeRobot 0.7.1 checkpoints for export POC validation.

Executed inside the isolated ``export-vla`` environment by ``validate_local.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _min_max(size: int) -> dict[str, torch.Tensor]:
    return {"min": torch.full((size,), -2.0), "max": torch.full((size,), 3.0)}


def _save(config: object, stats: object, path: Path) -> None:
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import get_policy_class

    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    policy = get_policy_class(config.type)(config).eval()
    if config.type == "tdmpc":
        for head in [policy.model._reward, *policy.model._Qs]:
            torch.nn.init.normal_(head[-1].weight)
    policy.save_pretrained(path)
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)


def build_checkpoints(root: Path, policies: set[str], *, lerobot_src: Path | None, use_group_norm: bool) -> None:
    """Materialize requested local checkpoints under *root*."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

    if lerobot_src is not None and str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    root.mkdir(parents=True, exist_ok=True)

    if "act" in policies:
        from lerobot.policies.act.configuration_act import ACTConfig

        _save(
            ACTConfig(
                input_features={
                    "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
                    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
                },
                output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
                device="cpu",
                chunk_size=4,
                n_action_steps=4,
                n_encoder_layers=1,
                n_decoder_layers=1,
                n_heads=2,
                dim_model=32,
                dim_feedforward=64,
                use_vae=False,
                pretrained_backbone_weights=None,
            ),
            {
                "observation.images.top": {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
                OBS_STATE: {"mean": torch.zeros(4), "std": torch.ones(4)},
                ACTION: {"mean": torch.zeros(4), "std": torch.ones(4)},
            },
            root / "act",
        )
        print("Built: act", flush=True)

    if "diffusion" in policies:
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

        _save(
            DiffusionConfig(
                input_features={
                    "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
                    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                },
                output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
                n_obs_steps=3,
                horizon=8,
                n_action_steps=4,
                drop_n_last_frames=2,
                crop_shape=(84, 84),
                pretrained_backbone_weights=None,
                use_group_norm=use_group_norm,
                spatial_softmax_num_keypoints=8,
                down_dims=(16, 32),
                kernel_size=3,
                n_groups=4,
                diffusion_step_embed_dim=16,
                num_train_timesteps=10,
                num_inference_steps=3,
                device="cpu",
            ),
            {
                "observation.images.front": {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
                OBS_STATE: _min_max(2),
                ACTION: _min_max(2),
            },
            root / "diffusion",
        )
        print(f"Built: diffusion (use_group_norm={use_group_norm})", flush=True)

    if "vqbet" in policies:
        from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig

        _save(
            VQBeTConfig(
                input_features={
                    "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
                    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                },
                output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
                n_obs_steps=3,
                n_action_pred_token=2,
                action_chunk_size=4,
                crop_shape=(84, 84),
                pretrained_backbone_weights=None,
                use_group_norm=use_group_norm,
                spatial_softmax_num_keypoints=8,
                vqvae_n_embed=8,
                vqvae_embedding_dim=16,
                vqvae_enc_hidden_dim=16,
                gpt_input_dim=32,
                gpt_output_dim=32,
                gpt_hidden_dim=32,
                gpt_n_layer=1,
                gpt_n_head=2,
                bet_softmax_temperature=10.0,
                device="cpu",
            ),
            {OBS_STATE: _min_max(2), ACTION: _min_max(2)},
            root / "vqbet",
        )
        print(f"Built: vqbet (use_group_norm={use_group_norm})", flush=True)

    if "tdmpc" in policies:
        from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig

        _save(
            TDMPCConfig(
                input_features={
                    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                    OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
                },
                output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
                image_encoder_hidden_dim=8,
                state_encoder_hidden_dim=16,
                latent_dim=8,
                mlp_dim=16,
                horizon=3,
                cem_iterations=2,
                n_gaussian_samples=16,
                n_pi_samples=4,
                n_elites=4,
                q_ensemble_size=3,
                device="cpu",
            ),
            {ACTION: _min_max(2)},
            root / "tdmpc",
        )
        print("Built: tdmpc", flush=True)

    if "gaussian_actor" in policies:
        from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig

        _save(
            GaussianActorConfig(
                input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,))},
                output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
                normalization_mapping={"STATE": NormalizationMode.IDENTITY, "ACTION": NormalizationMode.IDENTITY},
                device="cpu",
            ),
            None,
            root / "gaussian_actor",
        )
        print("Built: gaussian_actor", flush=True)

    if "groot" in policies:
        from tests.export import tiny_groot

        groot_ckpt = tiny_groot.save_checkpoint(root / "groot_work", "albumentations", use_bf16=False)
        (root / "groot.path").write_text(str(groot_ckpt), encoding="utf-8")
        print(f"Built: groot -> {groot_ckpt}", flush=True)

    if "g05" in policies:
        from tests.export import tiny_g05

        tiny_g05.save_checkpoint(root / "g05")
        print("Built: g05", flush=True)

    if "cosmos3_edge" in policies:
        from tests.export import tiny_cosmos3

        tiny_cosmos3.save_checkpoint(root / "cosmos3_edge")
        print("Built: cosmos3_edge", flush=True)

    if "lingbot_va" in policies:
        from tests.export import tiny_lingbot_va

        tiny_lingbot_va.save_checkpoint(root / "lingbot_va")
        print("Built: lingbot_va", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local LeRobot 0.7.1 checkpoints for POC validation.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--lerobot-src", type=Path, default=None)
    parser.add_argument("--group-norm", action="store_true")
    args = parser.parse_args()
    build_checkpoints(
        args.output_dir,
        set(args.policies),
        lerobot_src=args.lerobot_src,
        use_group_norm=args.group_norm,
    )


if __name__ == "__main__":
    main()
