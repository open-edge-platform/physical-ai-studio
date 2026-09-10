# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MolmoAct2 preprocessing and postprocessing."""

from dataclasses import replace

import pytest
import torch

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Feature, FeatureType, NormalizationParameters
from physicalai.policies.molmoact2 import MolmoAct2Config
from physicalai.policies.molmoact2.constants import (
    SO101_DEGREES_PER_NORMALIZED_UNIT,
    SO101_JOINT_OFFSETS,
    SO101_JOINT_SIGNS,
)
from physicalai.policies.molmoact2.processors import (
    MolmoAct2Postprocessor,
    MolmoAct2Preprocessor,
    make_molmoact2_preprocessors,
)
from physicalai.policies.molmoact2.processors.image import MolmoAct2ImageProcessor
from physicalai.policies.molmoact2.processors.inputs import (
    MolmoAct2InputLayout,
    _build_batched_images,
    _default_action_dim_is_pad,
    _expand_image_placeholders,
)
from physicalai.policies.molmoact2.processors.normalization import MolmoAct2NormalizeTransform
from physicalai.policies.molmoact2.processors.preprocess_steps import (
    ActionPadder,
    ImagePacker,
    PreprocessBatchBundle,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)
from physicalai.policies.utils import JointFrameTransform


def _so101_joint_transform() -> JointFrameTransform:
    return JointFrameTransform(signs=SO101_JOINT_SIGNS, offsets=SO101_JOINT_OFFSETS)


def test_factory_builds_matched_processors(tiny_molmoact2_config: MolmoAct2Config) -> None:
    preprocessor, postprocessor = make_molmoact2_preprocessors(tiny_molmoact2_config)

    assert isinstance(preprocessor, MolmoAct2Preprocessor)
    assert isinstance(postprocessor, MolmoAct2Postprocessor)


def test_factory_requires_resolved_features(tiny_molmoact2_config: MolmoAct2Config) -> None:
    tiny_molmoact2_config.input_features = None

    with pytest.raises(ValueError, match="features must be set"):
        make_molmoact2_preprocessors(tiny_molmoact2_config)


def test_factory_uses_action_feature_dimension_for_default_mask(tiny_molmoact2_config: MolmoAct2Config) -> None:
    tiny_molmoact2_config.output_features = [Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(2,))]

    preprocessor, _ = make_molmoact2_preprocessors(tiny_molmoact2_config)
    mask = _default_action_dim_is_pad(preprocessor._input_layout, batch_size=1, device=torch.device("cpu"))

    assert mask.tolist() == [[False, False, True, True]]


def test_factory_requires_resolved_action_feature(tiny_molmoact2_config: MolmoAct2Config) -> None:
    tiny_molmoact2_config.output_features = []

    with pytest.raises(ValueError, match="action output feature"):
        make_molmoact2_preprocessors(tiny_molmoact2_config)


def test_normalization_round_trip() -> None:
    feature = Feature(
        name=ACTION,
        ftype=FeatureType.ACTION,
        shape=(2,),
        normalization_data=NormalizationParameters(q01=[0.0, -2.0], q99=[2.0, 2.0]),
    )
    normalizer = MolmoAct2NormalizeTransform(input_features=[], output_features=[feature])
    denormalizer = MolmoAct2NormalizeTransform(input_features=[], output_features=[feature], inverse=True)
    action = torch.tensor([[[0.5, 1.0]]])

    normalized = normalizer({ACTION: action})[ACTION]
    restored = denormalizer({ACTION: normalized})[ACTION]

    torch.testing.assert_close(restored, action)


def test_joint_transform_maps_normalization_to_checkpoint_frame() -> None:
    normalization = NormalizationParameters(
        mean=[1.0, -40.0, 50.0, 4.0, 5.0, 6.0, 7.0],
        std=[2.0] * 7,
        min=[-10.0, -100.0, 10.0, -4.0, -5.0, 0.0, -7.0],
        max=[10.0, 20.0, 90.0, 4.0, 5.0, 30.0, 7.0],
        q01=[-8.0, -90.0, 20.0, -3.0, -4.0, 1.0, -6.0],
        q99=[8.0, 10.0, 80.0, 3.0, 4.0, 20.0, 6.0],
        mask=[True, True, True, True, True, True, False],
    )

    transformed = _so101_joint_transform().forward_normalization(normalization, dimension=7)

    assert transformed.mean == [1.0, 130.0, 140.0, 4.0, 5.0, 6.0, 7.0]
    assert transformed.std == [2.0] * 7
    assert transformed.min == [-10.0, 70.0, 100.0, -4.0, -5.0, 0.0, -7.0]
    assert transformed.max == [10.0, 190.0, 180.0, 4.0, 5.0, 30.0, 7.0]
    assert transformed.q01 == [-8.0, 80.0, 110.0, -3.0, -4.0, 1.0, -6.0]
    assert transformed.q99 == [8.0, 180.0, 170.0, 3.0, 4.0, 20.0, 6.0]
    assert transformed.mask == normalization.mask
    assert transformed is not normalization


def test_joint_transform_aligns_pretrained_stats_with_normalized_so101_runtime() -> None:
    degree_scales = [*SO101_DEGREES_PER_NORMALIZED_UNIT, 1.0]
    offsets = [0.0, 90.0, 90.0, 0.0, 0.0, 0.0]
    runtime_q01 = [-80.0, 110.0, -50.0, -25.0, -10.0, 2.0]
    runtime_q99 = [70.0, -60.0, 60.0, 35.0, 20.0, 95.0]
    checkpoint_q01 = [
        offsets[index] + degree_scales[index] * (value - offsets[index])
        for index, value in enumerate(runtime_q01)
    ]
    checkpoint_q99 = [
        offsets[index] + degree_scales[index] * (value - offsets[index])
        for index, value in enumerate(runtime_q99)
    ]
    normalization = NormalizationParameters(
        mean=checkpoint_q01,
        std=[2.0 * scale for scale in degree_scales],
        min=checkpoint_q01,
        max=checkpoint_q99,
        q01=checkpoint_q01,
        q99=checkpoint_q99,
        mask=[True, True, True, True, True, True],
    )

    transformed = _so101_joint_transform().forward_normalization_from_scaled_input(
        normalization,
        dimension=6,
        scales=SO101_DEGREES_PER_NORMALIZED_UNIT,
    )

    assert transformed.mean == pytest.approx(runtime_q01)
    assert transformed.std == pytest.approx([2.0] * 6)
    assert transformed.min == pytest.approx([min(a, b) for a, b in zip(runtime_q01, runtime_q99, strict=True)])
    assert transformed.max == pytest.approx([max(a, b) for a, b in zip(runtime_q01, runtime_q99, strict=True)])
    assert transformed.q01 == pytest.approx([min(a, b) for a, b in zip(runtime_q01, runtime_q99, strict=True)])
    assert transformed.q99 == pytest.approx([max(a, b) for a, b in zip(runtime_q01, runtime_q99, strict=True)])
    assert transformed.mask == normalization.mask


def test_corrected_pretrained_stats_match_explicit_degree_conversion() -> None:
    degree_scales = torch.tensor([*SO101_DEGREES_PER_NORMALIZED_UNIT, 1.0])
    signs = torch.tensor(SO101_JOINT_SIGNS)
    offsets = torch.tensor(SO101_JOINT_OFFSETS)
    checkpoint_stats = NormalizationParameters(
        q01=[-42.0, 44.0, 38.0, 6.0, -63.0, 1.0],
        q99=[48.0, 185.0, 173.0, 92.0, 43.0, 44.0],
    )
    corrected_stats = _so101_joint_transform().forward_normalization_from_scaled_input(
        checkpoint_stats,
        dimension=6,
        scales=SO101_DEGREES_PER_NORMALIZED_UNIT,
    )
    checkpoint_feature = Feature(
        name=STATE,
        ftype=FeatureType.STATE,
        shape=(6,),
        normalization_data=checkpoint_stats,
    )
    corrected_feature = replace(checkpoint_feature, normalization_data=corrected_stats)
    robot_state = torch.tensor([[-50.0, 25.0, -30.0, 10.0, 15.0, 60.0]])
    checkpoint_state = signs * degree_scales * robot_state + offsets
    adapted_state = _so101_joint_transform().forward(robot_state)
    reference_normalizer = MolmoAct2NormalizeTransform(input_features=[checkpoint_feature], output_features=[])
    corrected_normalizer = MolmoAct2NormalizeTransform(input_features=[corrected_feature], output_features=[])

    reference_state = reference_normalizer({STATE: checkpoint_state})[STATE]
    corrected_state = corrected_normalizer({STATE: adapted_state})[STATE]

    torch.testing.assert_close(corrected_state, reference_state)

    normalized_action = torch.tensor([[[-0.5, 0.25, 0.75, -0.25, 0.0, 0.5]]])
    checkpoint_action_feature = replace(checkpoint_feature, name=ACTION, ftype=FeatureType.ACTION)
    corrected_action_feature = replace(checkpoint_action_feature, normalization_data=corrected_stats)
    reference_denormalizer = MolmoAct2NormalizeTransform(
        input_features=[],
        output_features=[checkpoint_action_feature],
        inverse=True,
    )
    corrected_denormalizer = MolmoAct2NormalizeTransform(
        input_features=[],
        output_features=[corrected_action_feature],
        inverse=True,
    )
    checkpoint_action = reference_denormalizer({ACTION: normalized_action})[ACTION]
    expected_robot_action = _so101_joint_transform().inverse(checkpoint_action) / degree_scales
    corrected_action = corrected_denormalizer({ACTION: normalized_action})[ACTION]
    actual_robot_action = _so101_joint_transform().inverse(corrected_action)

    torch.testing.assert_close(actual_robot_action, expected_robot_action)


def test_joint_transform_rejects_mismatched_statistic_length() -> None:
    normalization = NormalizationParameters(q01=[-1.0], q99=[1.0])

    with pytest.raises(ValueError, match="does not match feature dimension"):
        _so101_joint_transform().forward_normalization(normalization, dimension=6)


def test_joint_transform_rejects_nested_statistics() -> None:
    normalization = NormalizationParameters(q01=[[[-1.0]]], q99=[[[1.0]]])

    with pytest.raises(ValueError, match="scalar or one-dimensional"):
        _so101_joint_transform().forward_normalization(normalization, dimension=1)


def test_joint_transform_rejects_mismatched_mask_length() -> None:
    normalization = NormalizationParameters(q01=-1.0, q99=1.0, mask=[True])

    with pytest.raises(ValueError, match="mask length"):
        _so101_joint_transform().forward_normalization(normalization, dimension=6)


def test_extractor_accepts_flattened_observations() -> None:
    extractor = StateTaskImageExtractor(image_keys=["front"])
    image = torch.zeros(2, 3, 8, 8)

    bundle = extractor.extract({STATE: torch.zeros(2, 4), TASK: "Pick block.", f"{IMAGES}.front": image})

    assert bundle.tasks == ["pick block", "pick block"]
    assert len(bundle.images_by_example) == 2
    assert bundle.images_by_example[0][0].shape == (3, 8, 8)


def test_extractor_sorts_nested_fallback_but_preserves_explicit_order() -> None:
    images = {
        "top": torch.zeros(1, 3, 8, 8),
        "wrist": torch.ones(1, 3, 8, 8),
    }
    batch = {STATE: torch.zeros(1, 4), TASK: "Pick block.", IMAGES: images}

    fallback = StateTaskImageExtractor(image_keys=[]).extract(batch).images_by_example[0]
    explicit = StateTaskImageExtractor(image_keys=["wrist", "top"]).extract(batch).images_by_example[0]

    assert torch.equal(fallback[0], images["top"][0])
    assert torch.equal(fallback[1], images["wrist"][0])
    assert torch.equal(explicit[0], images["wrist"][0])
    assert torch.equal(explicit[1], images["top"][0])


def test_prompt_encoder_includes_state_and_image_tokens() -> None:
    encoder = RobotPromptEncoder(
        num_state_tokens=16,
        setup_type="tabletop",
        control_mode="joint",
        add_setup_tokens=True,
        add_control_tokens=True,
    )
    bundle = PreprocessBatchBundle(
        state=torch.zeros(1, 2),
        tasks=["pick block"],
        images_by_example=[[torch.zeros(3, 8, 8)]],
    )

    prompt = encoder.encode(bundle).prompt_texts[0]

    assert "pick block" in prompt
    assert "<|image|>" in prompt
    assert "<state_start>" in prompt


def test_image_processing_and_packing_shapes() -> None:
    packer = ImagePacker(image_size=(28, 28))
    packed, mask = packer([[torch.zeros(3, 28, 28)], [torch.ones(3, 28, 28)]])
    processor = MolmoAct2ImageProcessor(
        crop_mode="resize",
        size={"height": 28, "width": 28},
        patch_size=14,
        pooling_size=[2, 2],
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    processed = processor(packed[0])

    assert packed.shape == (1, 2, 3, 28, 28)
    assert mask.tolist() == [[True, True]]
    assert processed["pixel_values"].shape == (2, 4, 14 * 14 * 3)


def test_placeholder_expansion_uses_configured_padding() -> None:
    layout = MolmoAct2InputLayout(
        env_action_dim=2,
        max_action_dim=4,
        image_placeholder_token_id=99,
        image_patch_id=11,
        image_start_token_id=10,
        image_end_token_id=12,
    )

    input_ids, attention_mask, _ = _expand_image_placeholders(
        layout=layout,
        pad_token_id=7,
        input_ids=torch.tensor([[99, 5], [99, 99]]),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        image_grids=torch.tensor([[1, 1, 0, 0]] * 3),
    )

    assert input_ids[0].tolist() == [10, 11, 12, 5, 7, 7]
    assert attention_mask[0].tolist() == [1, 1, 1, 1, 0, 0]


def test_build_batched_images_supports_multi_crop_grids() -> None:
    layout = MolmoAct2InputLayout(
        env_action_dim=2,
        max_action_dim=4,
        image_placeholder_token_id=99,
        image_patch_id=11,
        image_start_token_id=10,
        image_end_token_id=12,
    )

    images, pooling = _build_batched_images(
        layout,
        input_ids=torch.tensor([[10, 11, 12, 10, 11, 12], [10, 11, 12, 10, 11, 12]]),
        pixel_values=torch.arange(8, dtype=torch.float32).reshape(2, 4, 1),
        image_token_pooling=torch.arange(10, dtype=torch.long).reshape(10, 1) % 4,
        image_grids=torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]]),
        image_num_crops=torch.ones(2, dtype=torch.long),
    )

    assert images.shape == (2, 1, 4, 1)
    assert pooling.shape == (2, 5, 1)


def test_action_padder_returns_values_and_masks() -> None:
    padded, horizon_mask, dim_mask = ActionPadder(max_action_dim=4)(
        torch.tensor([[[2.0, -2.0]]]),
    )

    torch.testing.assert_close(padded, torch.tensor([[[1.0, -1.0, 0.0, 0.0]]]))
    assert horizon_mask.tolist() == [[False]]
    assert dim_mask.tolist() == [[False, False, True, True]]


def test_postprocessor_clamps_and_denormalizes() -> None:
    feature = Feature(
        name=ACTION,
        ftype=FeatureType.ACTION,
        shape=(1,),
        normalization_data=NormalizationParameters(q01=[0.0], q99=[2.0]),
    )
    postprocessor = MolmoAct2Postprocessor(output_features=[feature])

    result = postprocessor({ACTION: torch.tensor([[[-2.0], [2.0]]])})[ACTION]

    torch.testing.assert_close(result, torch.tensor([[[0.0], [2.0]]]))


def test_invalid_processor_inputs_raise() -> None:
    with pytest.raises(ValueError, match="state tensor"):
        StateTaskImageExtractor(image_keys=[]).extract({TASK: "task"})
    with pytest.raises(ValueError, match="action tensor"):
        MolmoAct2Postprocessor(output_features=[])({})
