# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Patch Policy image and action processing."""

import pytest
import torch
from torchvision.transforms.v2 import functional as tvf

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.policies.patch_policy.config import PatchPolicyConfig
from physicalai.policies.patch_policy.processors import (
    PatchPolicyPostprocessor,
    PatchPolicyPreprocessor,
    make_policy_processors,
    resolve_camera_name,
)
from physicalai.policies.utils.normalization import NormalizationType


def _camera(name: str = "front") -> Feature:
    return Feature(name=name, ftype=FeatureType.VISUAL, shape=(3, 8, 8))


def _actions() -> list[Feature]:
    return [
        Feature(
            name="arm",
            ftype=FeatureType.ACTION,
            shape=(2,),
            normalization_data=NormalizationParameters(
                mean=[1.0, 2.0], std=[2.0, 4.0], min=[-1.0, -2.0], max=[3.0, 6.0]
            ),
        ),
        Feature(
            name="gripper",
            ftype=FeatureType.ACTION,
            shape=(1,),
            normalization_data=NormalizationParameters(mean=[3.0], std=[5.0], min=[-2.0], max=[8.0]),
        ),
    ]


def test_uint8_images_are_scaled_resized_and_given_a_time_axis() -> None:
    processor = PatchPolicyPreprocessor([_camera()], [], (4, 6))
    image = torch.arange(192, dtype=torch.uint8).reshape(1, 3, 8, 8)
    original = image.clone()

    result = processor({"images": {"front": image}})["images.front"]

    expected = tvf.resize(image.float()[:, None] / 255.0, [4, 6], antialias=False)
    assert result.shape == (1, 1, 3, 4, 6)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(image, original)


@pytest.mark.parametrize("flatten", [False, True])
def test_temporal_images_follow_configured_camera_order_not_mapping_order(flatten: bool) -> None:
    processor = PatchPolicyPreprocessor([_camera("observation.images.right"), _camera("images.left")], [], (4, 4), 2)
    left = torch.stack([torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8)], dim=1)
    right = left + 2.0
    batch = {"images.left": left, "images.right": right} if flatten else {"images": {"left": left, "right": right}}

    result = processor(batch)

    assert list(result) == ["images.right", "images.left"]
    assert result["images.right"].shape == (1, 2, 3, 4, 4)
    torch.testing.assert_close(result["images.right"][:, :, 0, 0, 0], torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(result["images.left"][:, :, 0, 0, 0], torch.tensor([[0.0, 1.0]]))


def test_images_use_channelwise_mean_std_normalization() -> None:
    camera = Feature(
        name="front",
        ftype=FeatureType.VISUAL,
        shape=(3, 8, 8),
        normalization_data=NormalizationParameters(mean=[0.0, 0.5, 1.0], std=[1.0, 0.5, 2.0]),
    )
    processor = PatchPolicyPreprocessor([camera], [], (4, 4))

    result = processor({"images": torch.full((1, 3, 8, 8), 255, dtype=torch.uint8)})["images.front"]

    torch.testing.assert_close(result[0, 0, :, 0, 0], torch.tensor([1.0, 1.0, 0.0]))


@pytest.mark.parametrize("flatten", [False, True])
def test_actions_normalize_in_feature_order_and_preserve_padding_without_mutation(flatten: bool) -> None:
    processor = PatchPolicyPreprocessor([_camera()], _actions(), (4, 4))
    actions = torch.tensor([[[3.0, 6.0, 8.0], [1.0, 2.0, 3.0]]])
    original = actions.clone()
    padding = torch.tensor([[False, True]])
    batch = {"images": {"front": torch.zeros(1, 3, 8, 8)}, "action": actions}
    batch.update({"extra.action_is_pad": padding} if flatten else {"extra": {"action_is_pad": padding}})

    result = processor(batch)

    torch.testing.assert_close(result["action"], torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]))
    assert result["extra.action_is_pad"] is padding
    torch.testing.assert_close(actions, original)


def test_postprocessor_denormalizes_each_feature_and_trims_execution_horizon() -> None:
    processor = PatchPolicyPostprocessor(_actions(), n_action_steps=2)
    actions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]])
    original = actions.clone()

    result = processor(actions)

    torch.testing.assert_close(result, torch.tensor([[[1.0, 2.0, 3.0], [3.0, 6.0, 8.0]]]))
    torch.testing.assert_close(actions, original)


@pytest.mark.parametrize(
    "head, mode, expected",
    [
        ("diffusion", NormalizationType.MIN_MAX, [[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]),
        ("behavior_transformer", NormalizationType.IDENTITY, [[[3.0, 6.0, 8.0], [1.0, 2.0, 3.0]]]),
    ],
)
def test_processor_factory_selects_action_head_normalization_and_encoder_image_size(
    head: str,
    mode: NormalizationType,
    expected: list,
) -> None:
    config = PatchPolicyConfig(
        input_features=[_camera()],
        output_features=_actions(),
        action_dim=3,
        action_head=head,
        n_obs_steps=1,
        chunk_size=2,
        n_action_steps=1,
    )
    preprocessor, postprocessor = make_policy_processors(config, (4, 6))
    actions = torch.tensor([[[3.0, 6.0, 8.0], [1.0, 2.0, 3.0]]])

    result = preprocessor({"images": {"front": torch.zeros(1, 3, 8, 8)}, "action": actions})

    assert preprocessor.action_normalization is mode
    assert result["images.front"].shape == (1, 1, 3, 4, 6)
    torch.testing.assert_close(result["action"], torch.tensor(expected))
    torch.testing.assert_close(postprocessor(result["action"]), actions[:, :1])


def test_missing_action_statistics_warn_and_fall_back_to_identity() -> None:
    features = [
        Feature(
            name="action", ftype=FeatureType.ACTION, shape=(1,), normalization_data=NormalizationParameters(mean=[1.0])
        )
    ]
    with pytest.warns(UserWarning, match="Missing action mean/std.*normalization.*action"):
        preprocessor = PatchPolicyPreprocessor([_camera()], features, (4, 4))
    with pytest.warns(UserWarning, match="Missing action mean/std.*de-normalization.*action"):
        postprocessor = PatchPolicyPostprocessor(features, 1)
    actions = torch.tensor([[[4.0], [5.0]]])

    torch.testing.assert_close(preprocessor.normalize_actions(actions), actions)
    torch.testing.assert_close(postprocessor(actions), actions[:, :1])


def test_camera_names_and_temporal_image_layouts_are_validated() -> None:
    assert resolve_camera_name(_camera("observation.images.front")) == "front"
    with pytest.raises(ValueError, match="non-empty camera names"):
        resolve_camera_name(_camera("images."))
    with pytest.raises(ValueError, match="must be unique"):
        PatchPolicyPreprocessor([_camera("front"), _camera("images.front")], [], (4, 4))
    with pytest.raises(ValueError, match="n_obs_steps must be positive"):
        PatchPolicyPreprocessor([_camera()], [], (4, 4), 0)
    processor = PatchPolicyPreprocessor([_camera()], [], (4, 4), 2)
    with pytest.raises(KeyError, match="Missing visual feature"):
        processor({"images": {"other": torch.zeros(1, 2, 3, 8, 8)}})
    with pytest.raises(ValueError, match="Expected image shape"):
        processor({"images": {"front": torch.zeros(1, 3, 8, 8)}})
    with pytest.raises(ValueError, match="Expected 2 observation timestep"):
        processor({"images": {"front": torch.zeros(1, 3, 3, 8, 8)}})


@pytest.mark.parametrize(
    "action, error, message",
    [
        ([1.0], TypeError, "Expected actions to be a tensor"),
        (torch.zeros(1, 3), ValueError, "Expected action shape"),
        (torch.zeros(1, 2, 4), ValueError, "does not match configured width"),
    ],
)
def test_action_tensor_rank_and_width_are_validated(action: object, error: type[Exception], message: str) -> None:
    processor = PatchPolicyPreprocessor([_camera()], _actions(), (4, 4))
    with pytest.raises(error, match=message):
        processor({"images": {"front": torch.zeros(1, 3, 8, 8)}, "action": action})
    with pytest.raises(TypeError, match="not flattened action fields"):
        processor({"images": {"front": torch.zeros(1, 3, 8, 8)}, "action.arm": torch.zeros(1, 2, 2)})
    with pytest.raises(ValueError, match="does not match configured width"):
        PatchPolicyPostprocessor(_actions(), 1)(torch.zeros(1, 2, 4))
