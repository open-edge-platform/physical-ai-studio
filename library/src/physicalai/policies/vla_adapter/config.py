# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the VLA-Adapter model.

Upstream reference: https://github.com/OpenHelix-Team/VLA-Adapter (MIT),
arXiv:2509.09372.

Upstream keeps chunk length, action dimension and proprio dimension as
*module-level globals* in ``prismatic/vla/constants.py``, selected from
``sys.argv`` at import. That is incompatible with Studio's frozen dataclass
configs, so they are typed fields here; defaults reproduce the LIBERO preset.

CLI: ``physicalai fit --config configs/physicalai/vla_adapter.yaml``

Example:
    >>> from physicalai.policies.vla_adapter import VLAAdapterConfig
    >>> config = VLAAdapterConfig(chunk_size=8, max_action_dim=7)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from physicalai.config import Config


@dataclass(frozen=True)
class VLAAdapterConfig(Config):
    """Configuration for the VLA-Adapter vision-language-action model.

    Attributes:
        n_obs_steps: Number of observation steps to use. Defaults to 1.
        chunk_size: Size of action chunks for prediction. Upstream
            ``NUM_ACTIONS_CHUNK``. Defaults to 8 (the LIBERO preset).
        n_action_steps: Number of action steps executed per model invocation.
            Must not exceed ``chunk_size``. Defaults to 8, matching upstream's
            ``num_open_loop_steps``.
        max_state_dim: Proprioceptive state dimension. Upstream ``PROPRIO_DIM``.
            Defaults to 8 (LIBERO: xyz + axis-angle + gripper).
        max_action_dim: Action vector dimension. Upstream ``ACTION_DIM``.
            Defaults to 7.
        image_size: Target (height, width). Checkpoints record
            ``resize-naive`` — a plain stretch, no pad or crop. Default (224, 224).
        image_key_reorder_map: Optional mapping from dataset camera keys to
            policy camera slot indices, applied during preprocessing. Defaults to {}.
        num_cameras: Camera slots expected; uncovered slots are filled with
            masked empty cameras. Values <= 0 keep only batch cameras.
        num_images_in_input: Number of camera views the backbone consumes.
            Defaults to 2 (third-person + wrist).
        tokenizer_max_length: Maximum language token length. Fixed rather than
            dynamic so exported graphs keep static input shapes. Defaults to 48.
        num_task_tokens: Leading positions treated as *task* features by the
            head; the rest are action-query features. Derived from the backbone
            at build time; this documents the LIBERO layout. Defaults to 512.
        num_action_queries: Number of learned action-query tokens appended to
            the LLM input sequence. Upstream ``NUM_TOKENS``. Defaults to 64.
        head_num_heads: Number of attention heads inside each action-head block.
            Defaults to 8.
        llm_model_name: HuggingFace id supplying both the tokenizer and the
            pretrained language-model weights. Its geometry must match the
            ``llm_*`` fields below, or pretrained loading is skipped.
        load_pretrained_backbone: Whether to initialise the vision towers and
            language model from pretrained weights. Defaults to True: both are
            frozen, so random initialisation would leave the head reading noise.
            Set False for tests, or when a full VLA-Adapter checkpoint will
            supply every weight anyway.
        vision_backbone_ids: timm model ids for the fused DINOv2 and SigLIP
            towers.
        arch_specifier: Projector architecture, as recorded in the checkpoint's
            ``config.json``. Defaults to "no-align+fused-gelu-mlp".
        use_proprio: Whether to feed proprioceptive state to the action head.
            Defaults to True.
        train_vision_backbone: Whether the fused DINOv2 + SigLIP towers train.
            Defaults to False, as upstream freezes them.
        train_llm: Whether the language model trains. Defaults to False, as
            upstream freezes it and adapts through LoRA instead.
        optimizer_lr: Learning rate. Defaults to 5e-4, matching upstream.
        optimizer_betas: Adam beta coefficients. Defaults to (0.9, 0.95).
        optimizer_eps: Adam epsilon. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay coefficient. Defaults to 1e-10.
        optimizer_grad_clip_norm: Maximum gradient norm. Defaults to 1.0.
        scheduler_warmup_steps: LR warmup steps. Defaults to 1000.
        scheduler_decay_steps: LR decay steps. Defaults to 30000.
        scheduler_decay_lr: Final learning rate after decay. Defaults to 2.5e-6.
    """

    n_obs_steps: int = 1
    chunk_size: int = 8
    n_action_steps: int = 8

    max_state_dim: int = 8
    max_action_dim: int = 7

    image_size: tuple[int, int] = (224, 224)
    image_key_reorder_map: dict[str, int] = field(default_factory=dict)
    num_cameras: int = 0
    num_images_in_input: int = 2

    # NOTE: fixed for export compatibility (avoids dynamic input shapes).
    # Masking ignores unused tokens, so this should not affect accuracy.
    tokenizer_max_length: int = 48

    num_task_tokens: int = 512
    num_action_queries: int = 64
    head_num_heads: int = 8

    # The Prismatic checkpoint defines the topology but ships no usable fast
    # tokenizer, so both tokenizer and LLM weights come from Qwen directly.
    llm_model_name: str = "Qwen/Qwen2.5-0.5B"
    load_pretrained_backbone: bool = True
    # timm ids taken verbatim from the checkpoint's config.json (timm_model_ids).
    vision_backbone_ids: tuple[str, str] = (
        "vit_large_patch14_reg4_dinov2.lvd142m",
        "vit_so400m_patch14_siglip_224",
    )
    arch_specifier: str = "no-align+fused-gelu-mlp"

    use_proprio: bool = True

    # Only the two large pretrained backbones are configurable. Everything else
    # — visual projector, action queries, action head, proprio projector —
    # always trains: each is randomly or zero-initialised, so freezing any of
    # them would leave the policy unable to learn.
    train_vision_backbone: bool = False
    train_llm: bool = False

    optimizer_lr: float = 5e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If ``n_action_steps`` exceeds ``chunk_size``, or if any
                count that must be positive is not.
        """
        if self.n_action_steps > self.chunk_size:
            msg = (
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
            raise ValueError(msg)

        for name, value in (
            ("num_images_in_input", self.num_images_in_input),
            ("num_action_queries", self.num_action_queries),
        ):
            if value <= 0:
                msg = f"`{name}` must be positive, got {value}."
                raise ValueError(msg)
