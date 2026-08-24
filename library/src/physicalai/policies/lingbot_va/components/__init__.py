# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""First-party Wan2.2 components backing the LingBot-VA policy.

The modules here are a port of the upstream LingBot-VA / Wan2.2 model code: the dual-stream
transformer and its attention backends, the flow-matching scheduler, and the frozen
VAE / text-encoder plumbing. Nothing in this package imports the policy itself, so the
dependency direction stays one-way.
"""

from .attention import FlexAttnFunc, WanAttention, WanRotaryPosEmbed, custom_sdpa
from .scheduler import FlowMatchScheduler, sample_timestep_id
from .text import clean_prompt, encode_prompt, load_text_encoder, load_tokenizer
from .transformer import (
    WanTimeTextImageEmbedding,
    WanTransformer3DModel,
    WanTransformerBlock,
    data_seq_to_patch,
    get_mesh_id,
)
from .vae import (
    WanVAEStreamingWrapper,
    denormalize_latents,
    load_vae,
    normalize_latents,
    patchify,
)

__all__ = [
    "FlexAttnFunc",
    "FlowMatchScheduler",
    "WanAttention",
    "WanRotaryPosEmbed",
    "WanTimeTextImageEmbedding",
    "WanTransformer3DModel",
    "WanTransformerBlock",
    "WanVAEStreamingWrapper",
    "clean_prompt",
    "custom_sdpa",
    "data_seq_to_patch",
    "denormalize_latents",
    "encode_prompt",
    "get_mesh_id",
    "load_text_encoder",
    "load_tokenizer",
    "load_vae",
    "normalize_latents",
    "patchify",
    "sample_timestep_id",
]
