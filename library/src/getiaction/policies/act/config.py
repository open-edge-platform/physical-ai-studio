# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ACT policy config."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Self

from getiaction.data import Feature, FeatureType, NormalizationParameters


@dataclass(frozen=True)
class ACTConfig:
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        input_features: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_features: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    input_features: dict[str, Feature] = field(default_factory=dict)
    output_features: dict[str, Feature] = field(default_factory=dict)

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        """Convert ACTConfig to a plain dict for safe serialization.

        Uses dataclasses.asdict() which recursively converts all nested dataclasses
        (Feature, NormalizationParameters) to dicts, making the result pickle-safe
        and compatible with weights_only=True in torch.load().

        Returns:
            Dictionary representation of the config, safe for checkpoint serialization.
        """
        config_dict = dataclasses.asdict(self)
        # Convert FeatureType enums to strings for serialization
        for features_key in ("input_features", "output_features"):
            for feature_dict in config_dict[features_key].values():
                if feature_dict.get("ftype") is not None:
                    feature_dict["ftype"] = str(feature_dict["ftype"])
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """Reconstruct ACTConfig from a plain dict.

        Rebuilds the nested dataclass structure (Feature, NormalizationParameters)
        from the dict representation saved in checkpoints.

        Args:
            config_dict: Dictionary representation of ACTConfig.

        Returns:
            Reconstructed ACTConfig instance.
        """
        # Reconstruct Feature objects for input_features
        input_features = {}
        for name, feat_dict in config_dict.get("input_features", {}).items():
            norm_data = feat_dict.get("normalization_data")
            if norm_data is not None:
                norm_data = NormalizationParameters(**norm_data)
            ftype = feat_dict.get("ftype")
            if ftype is not None:
                ftype = FeatureType(ftype)
            input_features[name] = Feature(
                normalization_data=norm_data,
                ftype=ftype,
                shape=tuple(feat_dict["shape"]) if feat_dict.get("shape") else None,
                name=feat_dict.get("name"),
            )

        # Reconstruct Feature objects for output_features
        output_features = {}
        for name, feat_dict in config_dict.get("output_features", {}).items():
            norm_data = feat_dict.get("normalization_data")
            if norm_data is not None:
                norm_data = NormalizationParameters(**norm_data)
            ftype = feat_dict.get("ftype")
            if ftype is not None:
                ftype = FeatureType(ftype)
            output_features[name] = Feature(
                normalization_data=norm_data,
                ftype=ftype,
                shape=tuple(feat_dict["shape"]) if feat_dict.get("shape") else None,
                name=feat_dict.get("name"),
            )

        # Build ACTConfig with reconstructed features
        return cls(
            input_features=input_features,
            output_features=output_features,
            n_obs_steps=config_dict.get("n_obs_steps", 1),
            chunk_size=config_dict.get("chunk_size", 100),
            n_action_steps=config_dict.get("n_action_steps", 100),
            vision_backbone=config_dict.get("vision_backbone", "resnet18"),
            pretrained_backbone_weights=config_dict.get("pretrained_backbone_weights"),
            replace_final_stride_with_dilation=config_dict.get("replace_final_stride_with_dilation", False),
            pre_norm=config_dict.get("pre_norm", False),
            dim_model=config_dict.get("dim_model", 512),
            n_heads=config_dict.get("n_heads", 8),
            dim_feedforward=config_dict.get("dim_feedforward", 3200),
            feedforward_activation=config_dict.get("feedforward_activation", "relu"),
            n_encoder_layers=config_dict.get("n_encoder_layers", 4),
            n_decoder_layers=config_dict.get("n_decoder_layers", 1),
            use_vae=config_dict.get("use_vae", True),
            latent_dim=config_dict.get("latent_dim", 32),
            n_vae_encoder_layers=config_dict.get("n_vae_encoder_layers", 4),
            temporal_ensemble_coeff=config_dict.get("temporal_ensemble_coeff"),
            dropout=config_dict.get("dropout", 0.1),
            kl_weight=config_dict.get("kl_weight", 10.0),
        )
