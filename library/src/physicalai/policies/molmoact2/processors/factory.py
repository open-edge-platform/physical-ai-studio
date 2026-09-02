# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory helpers for MolmoAct2 preprocessors."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from physicalai.data.observation import Feature, FeatureType
from physicalai.policies.utils.features import get_feature_by_type

from .image import MolmoAct2ImageProcessor
from .inputs import MolmoAct2InputLayout
from .normalization import MolmoAct2NormalizeTransform
from .postprocessor import MolmoAct2Postprocessor
from .preprocess_steps import (
    ActionPadder,
    ImagePacker,
    ImageResizeNormalizer,
    RobotPromptEncoder,
    StateTaskImageExtractor,
)
from .preprocessor import MolmoAct2Preprocessor
from .tokenizers import MolmoAct2Tokenizers

if TYPE_CHECKING:
    from physicalai.policies import MolmoAct2Config


def _check_missing_tokens(required_tokens: dict[str, int | None]) -> None:
    missing_tokens = [name for name, value in required_tokens.items() if value is None]
    if missing_tokens:
        msg = f"MolmoAct2 requires configured image token IDs: {', '.join(missing_tokens)}"
        raise ValueError(msg)


def _check_missing_state_feature(state_feature: Feature | None) -> None:
    if state_feature is None or state_feature.shape is None:
        msg = "MolmoAct2 requires a state input feature with a resolved shape."
        raise ValueError(msg)


def _check_missing_action_feature(action_feature: Feature | None) -> None:
    if action_feature is None or action_feature.shape is None:
        msg = "MolmoAct2 requires an action output feature with a resolved shape."
        raise ValueError(msg)


def make_molmoact2_preprocessors(config: MolmoAct2Config) -> tuple[MolmoAct2Preprocessor, MolmoAct2Postprocessor]:
    """Build matched MolmoAct2 normalization processors.

    Args:
        config: MolmoAct2 config describing model.

    Returns:
        Forward preprocessing and inverse postprocessing modules.

    Raises:
        ValueError: If input or output features are unresolved.
    """
    if (config.input_features is None) or (config.output_features is None):
        msg = "Input and output features must be set; please initialize the model first."
        raise ValueError(msg)

    # shallow copy features
    input_features = list(config.input_features)
    output_features = list(config.output_features)

    # check required state/action features and tokens
    state_feature = get_feature_by_type(input_features, FeatureType.STATE)
    action_feature = get_feature_by_type(output_features, FeatureType.ACTION)
    _check_missing_state_feature(state_feature)
    _check_missing_action_feature(action_feature)
    _check_missing_tokens(
        {
            "image_start_token_id": config.image_start_token_id,
            "image_end_token_id": config.image_end_token_id,
            "image_patch_id": config.image_patch_id,
        },
    )

    # Normalize state and action features using the resolved dataset statistics.
    normalizer = MolmoAct2NormalizeTransform(
        input_features=input_features,
        output_features=output_features,
        normalization_mode=config.normalization_mode,
    )

    # Extract state, task text, and ordered camera images from input batches.
    extractor = StateTaskImageExtractor(
        image_keys=[feature.name for feature in input_features if feature.ftype == FeatureType.VISUAL and feature.name],
    )

    # Encode task and discretized state values into the model prompt format.
    prompt_encoder = RobotPromptEncoder(
        num_state_tokens=max(int(config.num_state_tokens), 1),
        setup_type=config.setup_type,
        control_mode=config.control_mode,
        add_setup_tokens=config.add_setup_tokens,
        add_control_tokens=config.add_control_tokens,
    )

    # determine image size from config
    image_size = (int(config.image_processor_size["height"]), int(config.image_processor_size["width"]))

    # Resize camera images and convert their values to the expected input range.
    image_resize = ImageResizeNormalizer(image_size=image_size)

    # Pack per-example camera images into the model's batched image layout.
    image_packer = ImagePacker(image_size=image_size)

    # Patchify and normalize packed images for the vision backbone.
    image_processor = MolmoAct2ImageProcessor(
        crop_mode=config.image_processor_crop_mode,
        size=config.image_processor_size,
        patch_size=config.image_processor_patch_size,
        pooling_size=config.image_processor_pooling_size,
        image_mean=config.image_processor_mean,
        image_std=config.image_processor_std,
    )

    # Tokenize prompts with the checkpoint-compatible local tokenizer assets.
    tokenizers = MolmoAct2Tokenizers(
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        max_token_len=config.tokenizer_max_length,
        padding=config.tokenizer_padding,
        tokenizer_config=config.tokenizer_config,
    )

    # Pad optional training actions to the model's fixed action dimension.
    action_padder = ActionPadder(max_action_dim=config.max_action_dim)

    # Describe action dimensions and special-token layout for model input assembly.
    input_layout = MolmoAct2InputLayout(
        env_action_dim=int(cast("tuple[int, ...]", cast("Feature", action_feature).shape)[-1]),
        max_action_dim=config.max_action_dim,
        image_placeholder_token_id=config.image_placeholder_token_id,
        image_patch_id=cast("int", config.image_patch_id),
        image_start_token_id=cast("int", config.image_start_token_id),
        image_end_token_id=cast("int", config.image_end_token_id),
        image_col_id=config.image_col_id,
        low_res_image_start_token_id=config.low_res_image_start_token_id,
        frame_start_token_id=config.frame_start_token_id,
        frame_end_token_id=config.frame_end_token_id,
        image_low_res_id=config.image_low_res_id,
        image_use_col_tokens=config.image_use_col_tokens,
        use_single_crop_col_tokens=config.use_single_crop_col_tokens,
        use_single_crop_start_token=config.use_single_crop_start_token,
    )

    # Assemble the matched preprocessor only after all components are ready.
    return (
        MolmoAct2Preprocessor(
            normalizer=normalizer,
            extractor=extractor,
            prompt_encoder=prompt_encoder,
            image_resize=image_resize,
            image_packer=image_packer,
            image_processor=image_processor,
            tokenizers=tokenizers,
            action_padder=action_padder,
            input_layout=input_layout,
            adapt_to_so101=config.adapt_to_so101,
        ),
        MolmoAct2Postprocessor(
            output_features=config.output_features,
            normalization_mode=config.normalization_mode,
            adapt_to_so101=config.adapt_to_so101,
        ),
    )
