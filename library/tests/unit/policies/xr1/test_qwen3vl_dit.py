# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the XR1 DiT action expert."""

from __future__ import annotations

import pytest
import torch
from physicalai.policies.xr1.qwen3vl_dit import (
    DiT,
    DiTAttention,
    MLPProjector,
    RMSNorm,
    TimestepEmbedder,
    modulate,
    repeat_batch,
    repeat_kv,
    rotate_half,
)

HIDDEN_SIZE = 256
HEAD_DIM = 128
KV_HEADS = 2
NUM_LAYERS = 3
CACHE_LEN = 5
QUERY_LEN = 6
BATCH = 2


def make_cache(
    num_layers: int = NUM_LAYERS,
    batch: int = BATCH,
    cache_len: int = CACHE_LEN,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build a stand-in for the VLM key/value cache.

    Args:
        num_layers: Number of cache layers.
        batch: Batch size.
        cache_len: Cached sequence length.

    Returns:
        Per-layer ``(key, value)`` pairs.
    """
    return [
        (torch.randn(batch, KV_HEADS, cache_len, HEAD_DIM), torch.randn(batch, KV_HEADS, cache_len, HEAD_DIM))
        for _ in range(num_layers)
    ]


def make_dit(layer_num: int = NUM_LAYERS) -> DiT:
    """Build a small DiT stack.

    Args:
        layer_num: Number of DiT layers.

    Returns:
        The DiT module.
    """
    return DiT(hidden_size=HIDDEN_SIZE, layer_num=layer_num, head_dim=HEAD_DIM, kv_heads=KV_HEADS)


def make_inputs(
    batch: int = BATCH,
    query_len: int = QUERY_LEN,
    cache_len: int = CACHE_LEN,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build hidden states, rotary embeddings, timestep conditioning and a mask.

    Args:
        batch: Batch size.
        query_len: DiT query length.
        cache_len: Cached sequence length.

    Returns:
        ``(hidden_states, (cos, sin), timestep, attn_mask)``.
    """
    hidden_states = torch.randn(batch, query_len, HIDDEN_SIZE)
    position_embeds = (torch.randn(batch, query_len, HEAD_DIM), torch.randn(batch, query_len, HEAD_DIM))
    timestep = torch.randn(batch, 6, HIDDEN_SIZE)
    attn_mask = torch.ones(batch, 1, query_len, cache_len + query_len, dtype=torch.bool)
    return hidden_states, position_embeds, timestep, attn_mask


class TestHelpers:
    """Tensor helpers ported from the reference implementation."""

    def test_modulate(self) -> None:
        """Modulation is an affine transform with a unit-offset scale."""
        x = torch.ones(1, 2, 3)
        shift = torch.full((1, 1, 3), 0.5)
        scale = torch.full((1, 1, 3), 1.0)

        assert torch.allclose(modulate(x, shift, scale), torch.full((1, 2, 3), 2.5))

    def test_repeat_kv_is_identity_for_single_group(self) -> None:
        """A single query head per kv head needs no repetition."""
        keys = torch.randn(1, 2, 3, 4)

        assert repeat_kv(keys, 1) is keys

    def test_repeat_kv_expands_head_dimension(self) -> None:
        """Grouped-query attention repeats each kv head."""
        keys = torch.randn(1, 2, 3, 4)
        repeated = repeat_kv(keys, 3)

        assert repeated.shape == (1, 6, 3, 4)
        assert torch.equal(repeated[:, 0], repeated[:, 1])

    def test_repeat_batch_expands(self) -> None:
        """The cache is expanded when training repeats samples."""
        cached = torch.randn(2, 1, 1, 1)

        assert repeat_batch(cached, 6).shape == (6, 1, 1, 1)

    def test_repeat_batch_rejects_non_multiple(self) -> None:
        """A non-divisible batch would silently misalign samples."""
        with pytest.raises(ValueError, match="Cannot repeat batch"):
            repeat_batch(torch.randn(4, 1, 1, 1), 6)


class TestMLPProjector:
    """Projector layer construction."""

    @pytest.mark.parametrize(("num_layers", "expected_linear"), [(1, 1), (2, 2), (4, 4)])
    def test_layer_count(self, num_layers: int, expected_linear: int) -> None:
        """Requested depth is materialized as that many linear layers."""
        projector = MLPProjector(8, HIDDEN_SIZE, num_layers=num_layers)
        linear_layers = [module for module in projector.layers if isinstance(module, torch.nn.Linear)]

        assert len(linear_layers) == expected_linear

    def test_output_shape(self) -> None:
        """Projection maps the last dimension only."""
        projector = MLPProjector(8, HIDDEN_SIZE, num_layers=2)

        assert projector(torch.randn(BATCH, 1, 8)).shape == (BATCH, 1, HIDDEN_SIZE)

    def test_rejects_zero_layers(self) -> None:
        """An empty projector is a configuration error."""
        with pytest.raises(ValueError, match="must be positive"):
            MLPProjector(8, HIDDEN_SIZE, num_layers=0)


class TestTimestepEmbedder:
    """Timestep conditioning."""

    def test_output_shape(self) -> None:
        """Embeddings carry a singleton sequence axis for broadcasting."""
        embedder = TimestepEmbedder(HIDDEN_SIZE)

        assert embedder(torch.rand(BATCH)).shape == (BATCH, 1, HIDDEN_SIZE)

    def test_follows_module_dtype(self) -> None:
        """The reference hardcodes bfloat16; we follow the module dtype instead.

        Without this, float32 runs and graph export both break.
        """
        embedder = TimestepEmbedder(HIDDEN_SIZE).to(torch.float32)
        assert embedder(torch.rand(BATCH)).dtype == torch.float32

        embedder = embedder.to(torch.bfloat16)
        assert embedder(torch.rand(BATCH)).dtype == torch.bfloat16

    def test_rejects_odd_frequency_size(self) -> None:
        """Sine and cosine halves must be equal in width."""
        with pytest.raises(ValueError, match="must be even"):
            TimestepEmbedder(HIDDEN_SIZE, frequency_embedding_size=127)


class TestDiTAttention:
    """Head geometry validation."""

    def test_rejects_indivisible_hidden_size(self) -> None:
        """Head dim must tile the hidden size."""
        with pytest.raises(ValueError, match="must be divisible by head_dim"):
            DiTAttention(hidden_size=300, head_dim=128)

    def test_rejects_indivisible_kv_heads(self) -> None:
        """Each kv head must serve a whole number of query heads."""
        with pytest.raises(ValueError, match="must be divisible by kv_heads"):
            DiTAttention(hidden_size=384, head_dim=128, kv_heads=2)

    def test_attends_over_cache_and_query(self) -> None:
        """Output length matches the query, not the cache."""
        attention = DiTAttention(hidden_size=HIDDEN_SIZE, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        hidden_states, position_embeds, _, attn_mask = make_inputs()
        cache = make_cache(num_layers=1)[0]

        output = attention(hidden_states, cache, position_embeds, attn_mask)

        assert output.shape == (BATCH, QUERY_LEN, HIDDEN_SIZE)

    def test_expands_cache_to_repeated_batch(self) -> None:
        """Training repeats queries while the cache stays at one entry per sample."""
        attention = DiTAttention(hidden_size=HIDDEN_SIZE, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        repeated_batch = BATCH * 2
        hidden_states, position_embeds, _, attn_mask = make_inputs(batch=repeated_batch)
        cache = make_cache(num_layers=1, batch=BATCH)[0]

        output = attention(hidden_states, cache, position_embeds, attn_mask)

        assert output.shape == (repeated_batch, QUERY_LEN, HIDDEN_SIZE)


class TestDiT:
    """The stacked action expert."""

    def test_output_shape(self) -> None:
        """The DiT preserves the query shape."""
        dit = make_dit()
        hidden_states, position_embeds, timestep, attn_mask = make_inputs()

        output = dit(hidden_states, make_cache(), attn_mask, position_embeds, timestep)

        assert output.shape == (BATCH, QUERY_LEN, HIDDEN_SIZE)

    def test_reads_deepest_cache_layers(self) -> None:
        """A shallow DiT aligns to the last VLM layers, not the first."""
        dit = make_dit(layer_num=2)
        hidden_states, position_embeds, timestep, attn_mask = make_inputs()
        cache = make_cache(num_layers=5)

        deep_only = dit(hidden_states, cache[-2:], attn_mask, position_embeds, timestep)
        full_cache = dit(hidden_states, cache, attn_mask, position_embeds, timestep)

        assert torch.allclose(deep_only, full_cache)

    def test_rejects_shallow_cache(self) -> None:
        """A cache shallower than the DiT is a config error, caught with a hint."""
        dit = make_dit(layer_num=4)
        hidden_states, position_embeds, timestep, attn_mask = make_inputs()

        with pytest.raises(ValueError, match="fewer than the DiT"):
            dit(hidden_states, make_cache(num_layers=2), attn_mask, position_embeds, timestep)

    def test_gradients_reach_every_layer(self) -> None:
        """Every layer participates, so the whole stack trains."""
        dit = make_dit()
        hidden_states, position_embeds, timestep, attn_mask = make_inputs()

        dit(hidden_states, make_cache(), attn_mask, position_embeds, timestep).sum().backward()

        for index, layer in enumerate(dit.layers):
            assert layer.attn.qkv_proj.weight.grad is not None, f"layer {index} received no gradient"


class TestLibraryEquivalence:
    """The locally implemented primitives must match transformers exactly.

    ``rotate_half`` and ``RMSNorm`` are implemented in this package rather than
    imported from ``transformers.models.qwen2.modeling_qwen2``, which is another
    model's private module and would version-lock the action expert. These tests are
    what make that trade safe.
    """

    def test_rotate_half_matches_transformers(self) -> None:
        """Rotation agrees bit-for-bit with the library implementation."""
        from transformers.models.qwen2.modeling_qwen2 import rotate_half as reference

        x = torch.randn(2, 4, 6, HEAD_DIM)

        assert torch.equal(rotate_half(x), reference(x))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_rms_norm_matches_transformers(self, dtype: torch.dtype) -> None:
        """Normalization agrees bit-for-bit, including the float32 upcast."""
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as Reference

        ours = RMSNorm(HIDDEN_SIZE).to(dtype)
        reference = Reference(HIDDEN_SIZE).to(dtype)
        with torch.no_grad():
            weight = torch.randn(HIDDEN_SIZE)
            ours.weight.copy_(weight)
            reference.weight.copy_(weight)
        x = torch.randn(BATCH, QUERY_LEN, HIDDEN_SIZE, dtype=dtype)

        assert torch.equal(ours(x), reference(x))

    def test_rms_norm_preserves_input_dtype(self) -> None:
        """The float32 upcast must not leak into the output dtype."""
        norm = RMSNorm(HIDDEN_SIZE).to(torch.bfloat16)

        assert norm(torch.randn(1, 2, HIDDEN_SIZE, dtype=torch.bfloat16)).dtype == torch.bfloat16

    def test_rms_norm_supports_autograd_through_views(self) -> None:
        """The norm runs on views from ``qkv.unbind(2)``, so it must not write in place."""
        norm = RMSNorm(HEAD_DIM)
        qkv = torch.randn(BATCH, QUERY_LEN, 3, 2, HEAD_DIM, requires_grad=True)
        query, _, _ = qkv.unbind(2)

        norm(query).sum().backward()

        assert qkv.grad is not None
