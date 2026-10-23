# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the modular Patch Policy pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import lightning as L  # noqa: N812
import pytest
import torch

from physicalai.data import Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies.patch_policy import PatchPolicy
from physicalai.policies.patch_policy.model import PatchPolicyModel
from physicalai.policies.patch_policy.processors import PatchPolicyPostprocessor, PatchPolicyPreprocessor


def features() -> tuple[list[Feature], list[Feature]]:
    """Return a minimal visual/action contract."""
    return (
        [Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 32, 32))],
        [Feature(name="action", ftype=FeatureType.ACTION, shape=(2,))],
    )


def make_policy(action_head: str, *, use_random_input_noise: bool = True) -> PatchPolicy:
    """Construct a small policy without downloading pretrained weights."""
    inputs, outputs = features()
    return PatchPolicy(
        input_features=inputs,
        output_features=outputs,
        encoder_name="resnet18",
        encoder_output="average",
        encoder_pretrained=False,
        image_size=(32, 32),
        action_head=action_head,  # type: ignore[arg-type]
        n_obs_steps=2,
        chunk_size=4,
        n_action_steps=2,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
        num_inference_steps=2,
        use_random_input_noise=use_random_input_noise,
    )


def batch() -> Observation:
    """Return a synthetic two-step visual batch."""
    return Observation(
        images={"front": torch.rand(2, 2, 3, 32, 32)},
        action=torch.rand(2, 4, 2) * 2 - 1,
    )


@torch.no_grad()
def test_processors_normalize_images_and_concatenated_actions() -> None:
    """Shared feature transforms preserve camera and flattened-action ordering."""
    inputs = [
        Feature(
            name="front",
            ftype=FeatureType.VISUAL,
            shape=(3, 4, 4),
            normalization_data=NormalizationParameters(mean=[0.5, 1.0, 1.5], std=[0.5, 1.0, 0.5]),
        ),
        Feature(
            name="wrist",
            ftype=FeatureType.VISUAL,
            shape=(3, 4, 4),
            normalization_data=NormalizationParameters(mean=[1.0, 1.0, 1.0], std=[2.0, 2.0, 2.0]),
        ),
    ]
    outputs = [
        Feature(
            name="action.arm",
            ftype=FeatureType.ACTION,
            shape=(2,),
            normalization_data=NormalizationParameters(mean=[1.0, 2.0], std=[2.0, 4.0]),
        ),
        Feature(
            name="action.gripper",
            ftype=FeatureType.ACTION,
            shape=(1,),
            normalization_data=NormalizationParameters(mean=[3.0], std=[0.5]),
        ),
    ]
    preprocessor = PatchPolicyPreprocessor(inputs, outputs, (4, 4))
    postprocessor = PatchPolicyPostprocessor(outputs, n_action_steps=1)
    raw_actions = torch.tensor([[[3.0, 6.0, 3.5], [1.0, 2.0, 3.0]]])

    processed = preprocessor(
        {
            "images": {
                "front": torch.cat(
                    [torch.full((1, 1, 1, 4, 4), value) for value in (0.5, 1.0, 2.0)],
                    dim=2,
                ),
                "wrist": torch.full((1, 1, 3, 4, 4), 3.0),
            },
            "action": raw_actions,
        },
        require_actions=True,
    )

    torch.testing.assert_close(processed["images"][0, 0, 0, :, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(processed["images"][0, 0, 1, :, 0, 0], torch.ones(3))
    torch.testing.assert_close(processed["action"], torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]))
    torch.testing.assert_close(postprocessor(processed["action"]), raw_actions[:, :1])


@torch.no_grad()
def test_diffusion_shapes() -> None:
    """Diffusion computes a scalar loss and an execution chunk."""
    policy = make_policy("diffusion")
    policy.train()
    loss, _ = policy(batch())
    assert loss.shape == ()
    policy.eval()
    assert policy.predict_action_chunk(batch()).shape == (2, 2, 2)


@torch.no_grad()
def test_behavior_transformer_shapes() -> None:
    """Behavior Transformer computes a scalar loss and an execution chunk."""
    policy = make_policy("behavior_transformer")
    policy.train()
    loss, _ = policy(batch())
    assert loss.shape == ()
    policy.eval()
    assert policy.predict_action_chunk(batch()).shape == (2, 2, 2)


def test_temporal_indices() -> None:
    """The model requests exactly its configured observation and action windows."""
    model = make_policy("diffusion").model
    assert model is not None
    assert model.observation_delta_indices == [-1, 0]
    assert model.action_delta_indices == [0, 1, 2, 3]


def test_from_config_reconstructs_explicit_values() -> None:
    """Construction from config preserves every model setting without an argument dictionary."""
    original = make_policy("diffusion", use_random_input_noise=False)
    config = replace(original.config, eta=0.0, clip_sample_range=0.5, use_random_input_noise=False)
    restored = PatchPolicy.from_config(config)

    assert restored.config == config
    assert not hasattr(restored, "_config_values")
    assert isinstance(restored.model, PatchPolicyModel)
    assert getattr(restored.model.action_head, "eta") == 0.0
    assert getattr(restored.model.action_head, "clip_sample_range") == 0.5
    assert getattr(restored.model.action_head, "use_random_input_noise") is False


def test_random_input_noise_propagates_from_policy() -> None:
    """The public policy option reaches the resolved config and diffusion head."""
    policy = make_policy("diffusion", use_random_input_noise=False)

    assert policy.config.use_random_input_noise is False
    assert isinstance(policy.model, PatchPolicyModel)
    assert getattr(policy.model.action_head, "use_random_input_noise") is False


@pytest.mark.parametrize("image_size", [(224, 224), (112, 224)])
@torch.no_grad()
def test_dinov2_uses_configured_image_size(image_size: tuple[int, int]) -> None:
    """DINOv2 accepts the configured resolution and initializes the matching patch count."""
    config = replace(
        make_policy("diffusion").config,
        encoder_name="dinov2",
        encoder_output="patches",
        image_size=image_size,
    )
    model = PatchPolicyModel.from_config(config)
    expected_tokens = (image_size[0] // 14) * (image_size[1] // 14)

    assert model.encoder.num_tokens == expected_tokens
    tokens = model.encoder(torch.rand(1, 2, 1, 3, *image_size))
    assert tokens.shape == (1, 2, 1, expected_tokens, model.encoder.output_dim)


def test_checkpoint_hooks_restore_complete_config() -> None:
    """Checkpoint hooks rebuild a lazy policy from its complete serialized config."""
    original = make_policy("diffusion", use_random_input_noise=False)
    checkpoint: dict[str, object] = {}
    original.on_save_checkpoint(checkpoint)

    restored = PatchPolicy(n_action_steps=original.config.n_action_steps)
    restored.on_load_checkpoint(checkpoint)

    assert restored.config == original.config
    assert isinstance(restored.model, PatchPolicyModel)


def test_load_from_checkpoint_round_trip(tmp_path: Path) -> None:
    """Lightning checkpoint loading reconstructs modules before restoring their tensors."""
    original = make_policy("diffusion", use_random_input_noise=False)
    checkpoint = {
        "state_dict": original.state_dict(),
        "hyper_parameters": dict(original.hparams),
        "pytorch-lightning_version": L.__version__,
    }
    original.on_save_checkpoint(checkpoint)
    path = tmp_path / "patch_policy.ckpt"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch  # Test-only checkpoint is generated from trusted local state.
    torch.save(checkpoint, path)

    restored = PatchPolicy.load_from_checkpoint(path)

    assert restored.config == original.config
    assert set(restored.state_dict()) == set(original.state_dict())
    assert restored.config.use_random_input_noise is False
