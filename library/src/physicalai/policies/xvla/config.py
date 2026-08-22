# Copyright 2025 2toINF (https://github.com/2toINF) and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the XVLA policy.

XVLA ("cross-embodiment vision-language-action") pairs a Florence-2 encoder with a
soft-prompted, domain-aware transformer that denoises an action chunk by flow matching.
One checkpoint serves many robots: a per-sample ``domain_id`` selects the domain-aware
projections and soft prompts, and an :mod:`~physicalai.policies.xvla.action_hub` action
space describes how the fixed-width action vector maps onto a given embodiment.

Example (CLI):
    physicalai fit --config configs/physicalai/xvla.yaml

Example (API):
    >>> from physicalai.policies.xvla import XVLAConfig
    >>> config = XVLAConfig()
    >>> config.chunk_size
    32
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from physicalai.config import Config

from .action_hub import ACTION_REGISTRY

if TYPE_CHECKING:
    from transformers import Florence2Config


def translate_vision_config(vision_config: dict[str, Any]) -> dict[str, Any]:
    """Translate a Florence-2 vision config from the legacy remote-code format.

    Published XVLA checkpoints store the vision config in Microsoft's original
    remote-code layout (``dim_embed``, ``image_pos_embed``, ...). This maps it onto the
    native ``transformers`` field names; a config already in the native format passes
    through unchanged.

    Args:
        vision_config: The checkpoint's vision config.

    Returns:
        The vision config in native ``transformers`` form.

    Raises:
        ValueError: If the config selects a backbone or feature combination that the
            native ``transformers`` implementation cannot reproduce.
    """
    vision = dict(vision_config)
    model_type = vision.pop("model_type", None)
    if model_type not in {None, "davit", "florence_vision"}:
        msg = f"Unsupported Florence-2 vision backbone: {model_type!r}"
        raise ValueError(msg)
    vision.pop("enable_checkpoint", None)

    image_pos_embed = vision.pop("image_pos_embed", None)
    if image_pos_embed is not None:
        if image_pos_embed.get("type") != "learned_abs_2d":
            msg = f"Unsupported image_pos_embed type: {image_pos_embed.get('type')!r}"
            raise ValueError(msg)
        vision["max_position_embeddings"] = image_pos_embed["max_pos_embeddings"]

    temporal_embedding = vision.pop("visual_temporal_embedding", None)
    if temporal_embedding is not None:
        if temporal_embedding.get("type") != "COSINE":
            msg = f"Unsupported visual_temporal_embedding type: {temporal_embedding.get('type')!r}"
            raise ValueError(msg)
        vision["max_temporal_embeddings"] = temporal_embedding["max_temporal_embeddings"]

    feature_source = vision.pop("image_feature_source", None)
    if feature_source is not None and list(feature_source) != ["spatial_avg_pool", "temporal_avg_pool"]:
        # The native Florence2MultiModalProjector hardcodes this feature combination.
        msg = f"Unsupported image_feature_source: {feature_source!r}"
        raise ValueError(msg)

    if "dim_embed" in vision:
        vision["embed_dim"] = vision.pop("dim_embed")
    return vision


@dataclass(frozen=True)
class XVLAConfig(Config):
    """Configuration for the XVLA flow-matching vision-language-action model.

    Attributes:
        florence_config: Architecture of the Florence-2 backbone, in either the native
            ``transformers`` format or the legacy remote-code format used by published
            XVLA checkpoints. Empty means the ``transformers`` defaults (Florence-2 base).
        tokenizer_name: HuggingFace tokenizer for the language prompt. Defaults to
            "facebook/bart-large", the tokenizer Florence-2's BART text stack was trained with.
        tokenizer_max_length: Prompt length, padded and truncated to a fixed size so the
            sequence length the transformer sees does not vary. Defaults to 64.
        dtype: Precision of the model weights. Defaults to "float32".

        n_obs_steps: Number of observation steps consumed per inference call. Defaults to 1.
        chunk_size: Number of action steps predicted per forward pass. Defaults to 32.
        n_action_steps: Number of action steps executed before the chunk is refilled.
            Must not exceed ``chunk_size``. Defaults to 32.

        hidden_size: Width of the action transformer. Defaults to 1024.
        depth: Number of transformer blocks. Defaults to 24.
        num_heads: Number of attention heads. Defaults to 16.
        mlp_ratio: Feed-forward expansion factor. Defaults to 4.0.
        num_domains: Number of embodiments the domain-aware layers can serve. Defaults to 30.
        len_soft_prompts: Learned prompt tokens per domain (``0`` disables them). Defaults to 32.
        dim_time: Width of the sinusoidal flow-matching timestep features. Defaults to 32.
        max_len_seq: Longest sequence the learned positional embedding covers. The sequence
            is ``chunk_size`` action tokens + the Florence-2 encoder output (image tokens of
            the primary view + ``tokenizer_max_length``) + the pooled auxiliary views.
            Defaults to 512.
        use_hetero_proj: Project the visual streams per domain rather than globally.
            Defaults to False.

        action_mode: Action space from :data:`~physicalai.policies.xvla.action_hub.ACTION_REGISTRY`.
            ``"auto"`` (the default) keeps the model's ``max_action_dim`` width but supervises
            and emits exactly the dataset's action width, so it fits any embodiment. The
            published checkpoints were trained with ``"ee6d"``.
        num_denoising_steps: Euler steps used to denoise a chunk at inference. Defaults to 10.
        use_proprio: Feed the proprioceptive state into the action tokens. Defaults to True.
        max_state_dim: Proprioceptive state is zero-padded (or truncated) to this width.
            Defaults to 32.
        max_action_dim: Action width the model predicts under ``action_mode="auto"``.
            Defaults to 20, matching the published checkpoints.
        domain_id: Domain index used when the batch carries none. Defaults to 0.
        domain_feature_key: Batch key holding a per-sample domain index. ``None`` looks for
            ``"domain_id"`` and ``"extra.domain_id"``. Defaults to None.

        resize_imgs_with_padding: Resize camera images to ``(height, width)``, preserving
            aspect ratio and padding the remainder. ``None`` passes them through at their
            dataset resolution. Defaults to None.
        num_image_views: Number of camera slots the model expects. ``None`` derives it from
            the dataset's cameras plus ``empty_cameras``. Defaults to None.
        empty_cameras: Masked-out camera slots appended to the real ones, so a checkpoint
            trained with more cameras than the dataset provides still lines up. Defaults to 0.

        freeze_vision_encoder: Freeze the Florence-2 vision tower. Defaults to False.
        freeze_language_encoder: Freeze the Florence-2 text encoder and its embeddings.
            Defaults to False.
        train_policy_transformer: Train the action transformer's backbone. Defaults to True.
        train_soft_prompts: Train the per-domain soft prompts. Defaults to True.

        normalization_mode: State/action normalization. XVLA's action spaces carry their own
            per-channel loss scaling and the published checkpoints are trained on raw units,
            so this defaults to ``"IDENTITY"``. Use ``"MEAN_STD"`` or ``"QUANTILES"`` when
            training from scratch on a dataset whose units are far from those checkpoints.

        optimizer_lr: Base learning rate. Defaults to 1e-4.
        optimizer_betas: Adam beta coefficients. Defaults to (0.9, 0.99).
        optimizer_eps: Optimizer epsilon. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay coefficient. Defaults to 0.0.
        optimizer_grad_clip_norm: Maximum gradient norm. Defaults to 10.0.
        optimizer_vlm_lr_scale: Learning-rate (and weight-decay) multiplier for the
            Florence-2 parameters. Upstream trains the VLM at a tenth of the base rate,
            which is what keeps finetuning stable. Defaults to 0.1.
        optimizer_soft_prompt_lr_scale: Learning-rate multiplier for the soft prompts.
            Defaults to 1.0.
        scheduler_warmup_steps: Linear warmup steps. Defaults to 1000.
        scheduler_decay_steps: Cosine decay horizon in steps. ``None`` auto-scales to the
            total training steps. Defaults to 30000.
        scheduler_decay_lr: Final learning rate after decay. Defaults to 2.5e-6.
    """

    # Florence-2 backbone and tokenizer
    florence_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_name: str = "facebook/bart-large"
    tokenizer_max_length: int = 64
    dtype: Literal["bfloat16", "float32"] = "float32"

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32

    # Action transformer
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_domains: int = 30
    len_soft_prompts: int = 32
    dim_time: int = 32
    max_len_seq: int = 512
    use_hetero_proj: bool = False

    # Action space and proprioception
    action_mode: str = "auto"
    num_denoising_steps: int = 10
    use_proprio: bool = True
    max_state_dim: int = 32
    max_action_dim: int = 20
    domain_id: int = 0
    domain_feature_key: str | None = None

    # Vision preprocessing
    resize_imgs_with_padding: tuple[int, int] | None = None
    num_image_views: int | None = None
    empty_cameras: int = 0

    # Finetuning
    freeze_vision_encoder: bool = False
    freeze_language_encoder: bool = False
    train_policy_transformer: bool = True
    train_soft_prompts: bool = True

    # Normalization
    normalization_mode: Literal["IDENTITY", "MEAN_STD", "QUANTILES"] = "IDENTITY"

    # Optimizer / scheduler
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.99)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 10.0
    optimizer_vlm_lr_scale: float = 0.1
    optimizer_soft_prompt_lr_scale: float = 1.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int | None = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        """Validate the configuration and normalize its sequence fields to tuples.

        A config built from YAML or JSON yields lists where the defaults are tuples;
        coercing them here keeps such a config equal to one built from the defaults.

        Raises:
            ValueError: If the chunk geometry, dtype, action space, or camera counts are
                inconsistent.
        """
        if not isinstance(self.optimizer_betas, tuple):
            object.__setattr__(self, "optimizer_betas", tuple(self.optimizer_betas))
        if self.resize_imgs_with_padding is not None and not isinstance(self.resize_imgs_with_padding, tuple):
            object.__setattr__(self, "resize_imgs_with_padding", tuple(self.resize_imgs_with_padding))

        if self.chunk_size <= 0:
            msg = f"chunk_size must be strictly positive, got {self.chunk_size}"
            raise ValueError(msg)

        if self.n_action_steps > self.chunk_size:
            msg = f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            raise ValueError(msg)

        if self.dtype not in {"bfloat16", "float32"}:
            msg = f"Invalid dtype: {self.dtype}"
            raise ValueError(msg)

        if self.action_mode.lower() not in ACTION_REGISTRY:
            msg = f"Unknown action_mode {self.action_mode!r}. Available: {sorted(ACTION_REGISTRY)}"
            raise ValueError(msg)

        if self.num_image_views is not None and self.num_image_views <= 0:
            msg = f"num_image_views must be > 0 when set, got {self.num_image_views}"
            raise ValueError(msg)

        if self.empty_cameras < 0:
            msg = f"empty_cameras cannot be negative, got {self.empty_cameras}"
            raise ValueError(msg)

        if self.max_action_dim <= 0:
            msg = f"max_action_dim must be strictly positive, got {self.max_action_dim}"
            raise ValueError(msg)

    @property
    def dim_proprio(self) -> int:
        """Width of the proprioceptive vector the action tokens carry (``0`` when disabled)."""
        return self.max_state_dim if self.use_proprio else 0

    def build_florence_config(self) -> Florence2Config:
        """Build the ``transformers`` Florence-2 config that backs the VLM.

        ``florence_config`` may be given in the native ``transformers`` format or in the
        legacy remote-code format published with existing XVLA checkpoints; the latter is
        translated field by field. An empty ``florence_config`` uses the ``transformers``
        defaults, which is what makes a randomly initialized ``XVLA()`` constructible
        without touching the network.

        Returns:
            The Florence-2 configuration.
        """
        from transformers import Florence2Config  # noqa: PLC0415

        raw = dict(self.florence_config)
        if not raw:
            return Florence2Config()

        vision_config = raw.get("vision_config")
        text_config = dict(raw.get("text_config") or {})
        if text_config.get("model_type", "florence2_language") == "florence2_language":
            # The legacy remote-code language config is BART, field for field.
            text_config["model_type"] = "bart"

        kwargs: dict[str, Any] = {
            key: raw[key]
            for key in (
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "image_token_id",
                "is_encoder_decoder",
                "tie_word_embeddings",
            )
            if key in raw
        }
        if vision_config:
            kwargs["vision_config"] = translate_vision_config(vision_config)
        if text_config:
            kwargs["text_config"] = text_config
        return Florence2Config(**kwargs)


__all__ = ["XVLAConfig", "translate_vision_config"]
