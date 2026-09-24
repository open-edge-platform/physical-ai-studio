# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MolmoAct2 preprocessing and postprocessing."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Feature, FeatureType, NormalizationParameters
from physicalai.policies.molmoact2 import MolmoAct2Config
from physicalai.policies.molmoact2.constants import SO101_JOINT_OFFSETS, SO101_JOINT_SIGNS
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
from physicalai.policies.molmoact2.so101 import (
    load_so101_calibration,
    make_so101_joint_transform,
    so101_degrees_per_runtime_unit,
    so101_joint_scales,
)
from physicalai.policies.utils import JointFrameTransform


def _so101_joint_transform() -> JointFrameTransform:
    return JointFrameTransform(signs=SO101_JOINT_SIGNS, offsets=SO101_JOINT_OFFSETS)


def test_factory_builds_matched_processors(tiny_molmoact2_config: MolmoAct2Config) -> None:
    preprocessor, postprocessor = make_molmoact2_preprocessors(tiny_molmoact2_config)

    assert isinstance(preprocessor, MolmoAct2Preprocessor)
    assert isinstance(postprocessor, MolmoAct2Postprocessor)


def test_factory_requires_resolved_features(tiny_molmoact2_config: MolmoAct2Config) -> None:
    config = replace(tiny_molmoact2_config, input_features=None)

    with pytest.raises(ValueError, match="features must be set"):
        make_molmoact2_preprocessors(config)


def test_factory_uses_action_feature_dimension_for_default_mask(tiny_molmoact2_config: MolmoAct2Config) -> None:
    config = replace(
        tiny_molmoact2_config,
        output_features=[Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(2,))],
    )

    preprocessor, _ = make_molmoact2_preprocessors(config)
    mask = _default_action_dim_is_pad(preprocessor._input_layout, batch_size=1, device=torch.device("cpu"))

    assert mask.tolist() == [[False, False, True, True]]


def test_factory_requires_resolved_action_feature(tiny_molmoact2_config: MolmoAct2Config) -> None:
    config = replace(tiny_molmoact2_config, output_features=[])

    with pytest.raises(ValueError, match="action output feature"):
        make_molmoact2_preprocessors(config)


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


def test_so101_joint_transform_maps_runtime_units_to_checkpoint_degrees(mock_so101_calibration) -> None:
    transform = make_so101_joint_transform(mock_so101_calibration)
    scales = torch.tensor(so101_degrees_per_runtime_unit(mock_so101_calibration))
    signs = torch.tensor(SO101_JOINT_SIGNS)
    offsets = torch.tensor(SO101_JOINT_OFFSETS)
    runtime_state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 50.0], [-100.0, 100.0, 50.0, -20.0, 10.0, 0.0]])

    checkpoint_state = transform.forward(runtime_state)

    torch.testing.assert_close(checkpoint_state, signs * scales * runtime_state + offsets)
    torch.testing.assert_close(checkpoint_state[0], torch.tensor([0.0, 90.0, 90.0, 0.0, 0.0, 50.0]))
    torch.testing.assert_close(transform.inverse(checkpoint_state), runtime_state)


def test_so101_processors_match_lerobot_degrees_pipeline(mock_so101_calibration) -> None:
    """Runtime ticks through PhysicalAI units must match LeRobot degrees through the same checkpoint stats."""
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    range_min = torch.tensor([float(mock_so101_calibration[joint]["range_min"]) for joint in joints])
    range_max = torch.tensor([float(mock_so101_calibration[joint]["range_max"]) for joint in joints])
    range_mid = (range_min + range_max) / 2
    range_width = range_max - range_min
    is_body = torch.tensor([True, True, True, True, True, False])

    def runtime_from_ticks(ticks: torch.Tensor) -> torch.Tensor:
        body = (ticks - range_mid) * 200.0 / range_width
        gripper = (ticks - range_min) * 100.0 / range_width
        return torch.where(is_body, body, gripper)

    def lerobot_from_ticks(ticks: torch.Tensor) -> torch.Tensor:
        body = (ticks - range_mid) * 360.0 / 4095.0
        gripper = (ticks - range_min) * 100.0 / range_width
        return torch.where(is_body, body, gripper)

    def ticks_from_runtime(values: torch.Tensor) -> torch.Tensor:
        body = values * range_width / 200.0 + range_mid
        gripper = values * range_width / 100.0 + range_min
        return torch.where(is_body, body, gripper)

    def ticks_from_lerobot(values: torch.Tensor) -> torch.Tensor:
        body = values * 4095.0 / 360.0 + range_mid
        gripper = values * range_width / 100.0 + range_min
        return torch.where(is_body, body, gripper)

    checkpoint_stats = NormalizationParameters(
        q01=[-42.0, 44.0, 38.0, 6.0, -63.0, 1.0],
        q99=[48.0, 185.0, 173.0, 92.0, 43.0, 44.0],
    )
    state_feature = Feature(name=STATE, ftype=FeatureType.STATE, shape=(6,), normalization_data=checkpoint_stats)
    action_feature = Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(6,), normalization_data=checkpoint_stats)
    normalizer = MolmoAct2NormalizeTransform(input_features=[state_feature], output_features=[])
    denormalizer = MolmoAct2NormalizeTransform(input_features=[], output_features=[action_feature], inverse=True)
    physicalai_transform = make_so101_joint_transform(mock_so101_calibration)
    lerobot_transform = JointFrameTransform(signs=SO101_JOINT_SIGNS, offsets=SO101_JOINT_OFFSETS)

    ticks = range_min + torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9, 0.2]]) * range_width
    physicalai_state = normalizer({STATE: physicalai_transform.forward(runtime_from_ticks(ticks))})[STATE]
    lerobot_state = normalizer({STATE: lerobot_transform.forward(lerobot_from_ticks(ticks))})[STATE]

    torch.testing.assert_close(physicalai_state, lerobot_state)

    checkpoint_action = denormalizer({ACTION: torch.tensor([[-0.5, 0.25, 0.75, -0.25, 0.0, 0.5]])})[ACTION]
    physicalai_ticks = ticks_from_runtime(physicalai_transform.inverse(checkpoint_action))
    lerobot_ticks = ticks_from_lerobot(lerobot_transform.inverse(checkpoint_action))

    torch.testing.assert_close(physicalai_ticks, lerobot_ticks)


def test_factory_uses_calibrated_so101_joint_transform(
    tiny_molmoact2_config: MolmoAct2Config,
    mock_so101_calibration,
) -> None:
    config = replace(tiny_molmoact2_config, adapt_to_so101=True, calibration=mock_so101_calibration)
    preprocessor, postprocessor = make_molmoact2_preprocessors(config)
    expected = make_so101_joint_transform(mock_so101_calibration)
    runtime_values = torch.tensor([[10.0, -20.0, 30.0, -40.0]])

    torch.testing.assert_close(
        preprocessor._joint_transform.forward(runtime_values),
        expected.forward(runtime_values),
    )
    torch.testing.assert_close(
        postprocessor._joint_transform.inverse(runtime_values),
        expected.inverse(runtime_values),
    )


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


def test_so101_degrees_per_runtime_unit():
    calibration = {
        "shoulder_pan": {"range_min": 701, "range_max": 3381},
        "shoulder_lift": {"range_min": 882, "range_max": 3244},
        "elbow_flex": {"range_min": 888, "range_max": 3088},
        "wrist_flex": {"range_min": 778, "range_max": 3092},
        "wrist_roll": {"range_min": 28, "range_max": 4065},
        # The gripper keeps [0, 100] in both conventions, so its range is ignored.
        "gripper": {"range_min": 2031, "range_max": 3538},
    }

    result = so101_degrees_per_runtime_unit(calibration)

    expected = (
        *(width * 360.0 / (200.0 * 4095.0) for width in (2680, 2362, 2200, 2314, 4037)),
        1.0,
    )

    assert result == pytest.approx(expected)


def test_so101_degrees_per_runtime_unit_missing_joint():
    calibration = {
        "shoulder_pan": {"range_min": 701, "range_max": 3381},
        "shoulder_lift": {"range_min": 882, "range_max": 3244},
        "elbow_flex": {"range_min": 888, "range_max": 3088},
        "wrist_flex": {"range_min": 778, "range_max": 3092},
        # wrist_roll missing
    }

    with pytest.raises(ValueError, match="missing required joint 'wrist_roll'"):
        so101_degrees_per_runtime_unit(calibration)


def test_so101_degrees_per_runtime_unit_missing_range():
    calibration = {
        "shoulder_pan": {"range_min": 701},
        "shoulder_lift": {"range_min": 882, "range_max": 3244},
        "elbow_flex": {"range_min": 888, "range_max": 3088},
        "wrist_flex": {"range_min": 778, "range_max": 3092},
        "wrist_roll": {"range_min": 28, "range_max": 4065},
    }

    with pytest.raises(ValueError, match="missing 'range_max'"):
        so101_degrees_per_runtime_unit(calibration)


def test_so101_degrees_per_runtime_unit_invalid_range():
    calibration = {
        "shoulder_pan": {"range_min": 3381, "range_max": 701},
        "shoulder_lift": {"range_min": 882, "range_max": 3244},
        "elbow_flex": {"range_min": 888, "range_max": 3088},
        "wrist_flex": {"range_min": 778, "range_max": 3092},
        "wrist_roll": {"range_min": 28, "range_max": 4065},
    }

    with pytest.raises(ValueError, match="Invalid SO-101 calibration range"):
        so101_degrees_per_runtime_unit(calibration)


def test_so101_joint_scales_default_to_one_without_calibration():
    assert so101_joint_scales(None) == (1.0,) * 6


@pytest.mark.parametrize("as_type", [str, Path])
def test_load_so101_calibration_reads_json_path(tmp_path: Path, mock_so101_calibration, as_type) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(mock_so101_calibration), encoding="utf-8")

    assert load_so101_calibration(as_type(path)) == mock_so101_calibration


def test_load_so101_calibration_passes_through_dict_and_none(mock_so101_calibration) -> None:
    assert load_so101_calibration(mock_so101_calibration) is mock_so101_calibration
    assert load_so101_calibration(None) is None
