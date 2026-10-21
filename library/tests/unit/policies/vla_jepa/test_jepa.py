# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, SLF001, PLR2004

"""Joint JEPA/action training, temporal safety, and checkpoint mode contracts."""

from pathlib import Path
from unittest.mock import patch

import lightning
import pytest
import torch
from torch.utils.data import DataLoader

from physicalai.data import Observation
from physicalai.policies.vla_jepa import VLAJEPA, VLAJEPAConfig, VLAJEPAModel
from physicalai.policies.vla_jepa.processors import VLAJEPAPreprocessor


def test_full_training_is_default_and_action_only_is_explicit() -> None:
    """Fresh training defaults to JEPA, independently of eager inference loading."""
    assert VLAJEPA().enable_world_model is True
    assert VLAJEPA(enable_world_model=False).enable_world_model is False
    assert VLAJEPA(train_action_head_only=True).enable_world_model is False


def test_joint_loss_trains_qwen_head_and_predictor_not_teacher(
    jepa_config: VLAJEPAConfig,
    video_observation: Observation,
) -> None:
    """Both objectives backpropagate while the target encoder remains frozen and in eval mode."""
    policy = VLAJEPA.from_config(jepa_config)
    policy.train()
    loss, metrics = policy(video_observation)
    torch.testing.assert_close(
        loss.detach(),
        metrics["action_flow_loss"] + jepa_config.world_model_loss_weight * metrics["world_model_loss"],
    )
    assert metrics["world_model_loss"] > 0
    loss.backward()
    assert policy.model.video_predictor.predictor_embed.weight.grad is not None
    assert policy.model.qwen.language_model.embed_tokens.weight.grad is not None
    assert policy.model.action_model.action_encoder.layer1.weight.grad is not None
    assert not policy.model.video_encoder.training
    assert all(
        not parameter.requires_grad and parameter.grad is None for parameter in policy.model.video_encoder.parameters()
    )
    assert policy.model.observation_delta_indices == list(range(8))


def test_future_frames_and_states_never_condition_policy(
    jepa_config: VLAJEPAConfig,
    video_observation: Observation,
) -> None:
    """Only frame/state zero reaches Qwen and the action head; future frames supervise JEPA."""
    processor = VLAJEPAPreprocessor(jepa_config)
    before = processor(video_observation.to_dict(), include_video=True)
    video_observation.images["front"][:, 1:] = 1 - video_observation.images["front"][:, 1:]
    video_observation.state[:, 1:] += 100
    after = processor(video_observation.to_dict(), include_video=True)
    for name in ("input_ids", "pixel_values", "image_grid_thw", "state", "latent_positions"):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    assert not torch.equal(before["videos"], after["videos"])


@pytest.mark.parametrize("mask_location", ["extra", "images"])
def test_video_padding_excludes_invalid_future_tubelets(
    jepa_config: VLAJEPAConfig,
    video_observation: Observation,
    mask_location: str,
) -> None:
    """Episode-end clamping does not create JEPA supervision on duplicated frames."""
    key = "observation.images.front_is_pad" if mask_location == "extra" else "front_is_pad"
    getattr(video_observation, mask_location)[key] = torch.tensor([
        [False] * 8,
        [False] * 5 + [True] * 3,
    ])
    processed = VLAJEPAPreprocessor(jepa_config)(video_observation.to_dict(), include_video=True)
    torch.testing.assert_close(processed["video_valid"], torch.tensor([[True, True, True], [True, False, False]]))
    model = VLAJEPAModel.from_config(jepa_config)
    processed["video_valid"].zero_()
    latent_tokens = model._context(processed)["latent_tokens"]
    loss = model._world_model_loss(processed, latent_tokens)
    assert loss.item() == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_view_concatenation_keeps_batches_separate(jepa_config: VLAJEPAConfig, video_observation: Observation) -> None:
    """Two views from one sample must never be merged with a different sample's views."""
    model = VLAJEPAModel.from_config(jepa_config)
    processed = VLAJEPAPreprocessor(jepa_config)(video_observation.to_dict(), include_video=True)
    encoded = torch.tensor([0.0, 1.0, 10.0, 11.0])[:, None, None].expand(4, 16, 32)
    with (
        patch.object(model.video_encoder, "get_vision_features", return_value=encoded),
        patch.object(
            model.video_predictor,
            "forward",
            side_effect=lambda states, _actions: torch.zeros_like(states),
        ) as predict,
    ):
        model._world_model_loss(processed, torch.randn(2, 24, 32))
    states = predict.call_args.args[0]
    torch.testing.assert_close(states[0, :, :32], torch.zeros(12, 32))
    torch.testing.assert_close(states[0, :, 32:], torch.ones(12, 32))
    torch.testing.assert_close(states[1, :, :32], torch.full((12, 32), 10.0))
    torch.testing.assert_close(states[1, :, 32:], torch.full((12, 32), 11.0))


def test_single_view_is_padded_without_adding_policy_cameras(
    jepa_config: VLAJEPAConfig,
    video_observation: Observation,
) -> None:
    """One camera produces two teacher views while Qwen still receives one camera."""
    processed = VLAJEPAPreprocessor(jepa_config)(video_observation.to_dict(), include_video=True)
    assert processed["videos"].shape == (2, 2, 8, 3, 32, 32)
    torch.testing.assert_close(processed["videos"][:, 0], processed["videos"][:, 1])
    assert processed["image_grid_thw"].shape[0] == 2


def test_predictor_attention_is_frame_causal(jepa_config: VLAJEPAConfig) -> None:
    """Later context frames cannot change an earlier frame's prediction."""
    predictor = VLAJEPAModel.from_config(jepa_config).video_predictor.eval()
    states = torch.randn(2, 12, 64)
    actions = torch.randn(2, 24, 32)
    before = predictor(states, actions)
    states[:, 4:] += 10
    actions[:, 8:] -= 10
    after = predictor(states, actions)
    torch.testing.assert_close(before[:, :4], after[:, :4])


def test_joint_inference_does_not_execute_world_model(jepa_config: VLAJEPAConfig, observation: Observation) -> None:
    """Inference accepts a single current frame and does not run teacher or predictor."""
    policy = VLAJEPA.from_config(jepa_config).eval()
    with (
        patch.object(
            policy.model.video_encoder,
            "get_vision_features",
            side_effect=AssertionError("teacher inference"),
        ),
        patch.object(policy.model.video_predictor, "forward", side_effect=AssertionError("predictor inference")),
    ):
        assert policy.predict_action_chunk(observation).shape == (2, 3, 3)


def test_full_checkpoint_restores_offline_and_can_strip_world_model(
    jepa_config: VLAJEPAConfig,
    video_observation: Observation,
    tmp_path: Path,
) -> None:
    """Resume retains JEPA; explicit action-only loading omits all training-only tensors."""
    policy = VLAJEPA.from_config(jepa_config)
    trainer = lightning.Trainer(
        accelerator="cpu",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    with patch.object(policy, "setup"):
        trainer.fit(policy, train_dataloaders=DataLoader([video_observation], batch_size=None))
    path = tmp_path / "joint.ckpt"
    trainer.save_checkpoint(path, weights_only=True)
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained", side_effect=AssertionError("download")):
        restored = VLAJEPA.load_from_checkpoint(path, weights_only=True)
        action_only = VLAJEPA.load_from_checkpoint(path, weights_only=True, enable_world_model=False)
    assert restored.enable_world_model
    assert restored.model.video_encoder is not None
    assert not action_only.enable_world_model
    assert action_only.model.video_encoder is None
    assert not any(name.startswith("video_") for name in action_only.model.state_dict())
    assert action_only.model.observation_delta_indices is None
    torch.testing.assert_close(
        restored.model.qwen.language_model.embed_tokens.weight,
        action_only.model.qwen.language_model.embed_tokens.weight,
    )


def test_old_checkpoints_remain_action_only(tiny_config: VLAJEPAConfig) -> None:
    """Version-one checkpoints do not acquire random JEPA weights on restore."""
    data = tiny_config.to_dict()
    data["format_version"] = 1
    data.pop("enable_world_model")
    restored = VLAJEPAConfig.from_dict(data)
    assert restored.format_version == 2
    assert not restored.enable_world_model
    policy = VLAJEPA.from_config(restored)
    assert policy.model.video_encoder is None
    with pytest.raises(ValueError, match="action-only"):
        VLAJEPA(enable_world_model=True).on_load_checkpoint({"policy_config": restored.to_dict()})
