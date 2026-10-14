# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Seeded CPU action parity for a native LeRobot ACT artifact and Runtime loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.integration
def test_runtime_actions_match_native_lerobot_across_reset(tmp_path: Path) -> None:
    """Compare a tiny local ACT model without external data or GPU dependencies."""
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from physicalai.engine import Engine

    torch.manual_seed(42)
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_heads=2,
        dim_model=16,
        dim_feedforward=32,
        use_vae=False,
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(config)
    dataset_stats = {
        "observation.state": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "observation.environment_state": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "action": {"mean": torch.zeros(1), "std": torch.ones(1)},
    }
    native_dir = tmp_path / "native" / "pretrained_model"
    native_dir.mkdir(parents=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=dataset_stats,
    )
    policy.save_pretrained(native_dir)
    preprocessor.save_pretrained(native_dir)
    postprocessor.save_pretrained(native_dir)

    package = Engine().export(native_dir, output_dir=tmp_path / "runtime-package", framework="lerobot")
    runtime_model = Engine().load(package, device="cpu")
    native_observation = {
        "observation.state": torch.tensor([0.2, -0.1]),
        "observation.environment_state": torch.tensor([0.3, 0.4]),
    }
    runtime_observation = {key: value.numpy() for key, value in native_observation.items()}

    def assert_next_action_matches() -> None:
        with torch.inference_mode():
            native_action = postprocessor(policy.select_action(preprocessor(native_observation)))
        runtime_action = runtime_model.select_action(runtime_observation)
        np.testing.assert_allclose(
            native_action.detach().cpu().numpy().reshape(-1),
            runtime_action,
            rtol=1e-6,
            atol=1e-7,
        )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    runtime_model.reset()
    for _ in range(3):
        assert_next_action_matches()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    runtime_model.reset()
    for _ in range(2):
        assert_next_action_matches()
