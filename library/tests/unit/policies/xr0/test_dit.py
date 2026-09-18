# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 DiT action head and rectified-flow model (``dit``).
"""

from __future__ import annotations

import pytest
import torch

from physicalai.policies.xr0.dit import (
    DiT,
    DecoderLayer,
    DiTAttention,
    DiTMLP,
    MLPProjector,
    TimestepEmbedder,
    XR0FlowModel,
    apply_rotary_pos_emb,
    modulate,
    repeat_kv,
)

# Small structural config shared across module tests.
HIDDEN = 64
HEAD_DIM = 16
NUM_HEADS = HIDDEN // HEAD_DIM  # 4
KV_HEADS = 2
BATCH = 2
SEQ = 5
CACHE = 3

# Tolerance for pinned reference outputs.
TOL = {"atol": 1e-4, "rtol": 1e-4}


def _attn_inputs(seq: int = SEQ, cache: int = CACHE):
    """Build small inputs for a DiTAttention / DecoderLayer forward."""
    hidden = torch.randn(BATCH, seq, HIDDEN)
    past_key = torch.randn(BATCH, KV_HEADS, cache, HEAD_DIM)
    past_value = torch.randn(BATCH, KV_HEADS, cache, HEAD_DIM)
    cos = torch.randn(BATCH, seq, HEAD_DIM)
    sin = torch.randn(BATCH, seq, HEAD_DIM)
    # Boolean mask over [cache | causal-query] keys.
    mask = torch.ones(BATCH, 1, seq, cache + seq, dtype=torch.bool)
    return hidden, (past_key, past_value), (cos, sin), mask


# ============================================================================ #
# Helper Functions                                                             #
# ============================================================================ #


class TestHelpers:
    """Tests for the DiT tensor helper functions."""

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([-0.21684796, 0.37115434, 1.17266655, 0.0787273, 1.57226944])],
    )
    def test_modulate_reference(self, reference: torch.Tensor) -> None:
        """A seeded modulate pins a slice of x * (1 + scale) + shift."""
        torch.manual_seed(0)
        x = torch.randn(2, 3, HIDDEN)
        shift = torch.randn(2, 1, HIDDEN)
        scale = torch.randn(2, 1, HIDDEN)
        out = modulate(x, shift, scale)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()

    def test_repeat_kv_identity(self) -> None:
        """n_rep=1 is a no-op."""
        kv = torch.randn(BATCH, KV_HEADS, CACHE, HEAD_DIM)
        torch.testing.assert_close(repeat_kv(kv, 1), kv)

    def test_repeat_kv_interleaves(self) -> None:
        """Repeated heads are interleaved (head i -> positions i*n_rep .. )."""
        kv = torch.randn(1, 2, CACHE, HEAD_DIM)
        out = repeat_kv(kv, 2)
        torch.testing.assert_close(out[:, 0], kv[:, 0])
        torch.testing.assert_close(out[:, 1], kv[:, 0])
        torch.testing.assert_close(out[:, 2], kv[:, 1])
        torch.testing.assert_close(out[:, 3], kv[:, 1])

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.64037418, 0.58257341, -0.39190835, -0.41302168, -0.85603261])],
    )
    def test_apply_rotary_reference(self, reference: torch.Tensor) -> None:
        """A seeded rotary embedding pins a slice of the rotated query."""
        torch.manual_seed(0)
        q = torch.randn(BATCH, NUM_HEADS, SEQ, HEAD_DIM)
        k = torch.randn(BATCH, NUM_HEADS, SEQ, HEAD_DIM)
        cos = torch.randn(BATCH, SEQ, HEAD_DIM)
        sin = torch.randn(BATCH, SEQ, HEAD_DIM)
        q_out, _ = apply_rotary_pos_emb(q, k, cos, sin)
        out = q_out[0, 0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


# ============================================================================ #
# Projectors & Embedders                                                       #
# ============================================================================ #


class TestMLPProjector:
    """Tests for MLPProjector."""
    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.06443175, -0.08235656, -0.01462553, -0.19436540, -0.07179737])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded 2-layer projector pins a slice of its output."""
        torch.manual_seed(0)
        proj = MLPProjector(32, HIDDEN, num_layers=2)
        out = proj(torch.randn(BATCH, SEQ, 32))[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


class TestTimestepEmbedder:
    """Tests for TimestepEmbedder."""

    def test_dtype_respected(self) -> None:
        """The frequency embedding is cast to the configured dtype."""
        emb = TimestepEmbedder(HIDDEN, dtype=torch.float32)
        assert emb(torch.tensor([1.0])).dtype == torch.float32

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.34102309, 0.21420282, 0.16651681, 0.06136062])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded embedder pins the first channel across timesteps."""
        torch.manual_seed(0)
        emb = TimestepEmbedder(HIDDEN, dtype=torch.float32)
        out = emb(torch.tensor([0.0, 200.0, 500.0, 999.0]))[:, 0, 0]
        assert torch.allclose(out, reference, **TOL), out.tolist()


# ============================================================================ #
# DiT Components                                                               #
# ============================================================================ #


class TestDiTMLP:
    """Tests for DiTMLP (SwiGLU)."""

    def test_intermediate_size(self) -> None:
        """Intermediate size is 4x the hidden size."""
        assert DiTMLP(HIDDEN).intermediate_size == 4 * HIDDEN

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.14671412, 0.22900799, 0.19239995, 0.04836469, 0.22137016])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded SwiGLU MLP pins a slice of its output."""
        torch.manual_seed(0)
        mlp = DiTMLP(HIDDEN)
        out = mlp(torch.randn(BATCH, SEQ, HIDDEN))[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


class TestDiTAttention:
    """Tests for DiTAttention."""

    def test_head_configuration(self) -> None:
        """num_heads and kv_group derive from hidden/head_dim/kv_heads."""
        attn = DiTAttention(hidden_size=HIDDEN, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        assert attn.num_heads == NUM_HEADS
        assert attn.kv_group == NUM_HEADS // KV_HEADS

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.13715474, -0.13632900, 0.11728425, 0.05819567, -0.16545057])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded attention forward pins a slice of its output."""
        torch.manual_seed(0)
        attn = DiTAttention(hidden_size=HIDDEN, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        hidden, kv, pos, mask = _attn_inputs()
        out = attn(hidden, kv, pos, attn_mask=mask)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


class TestDecoderLayer:
    """Tests for DecoderLayer."""

    def test_adaln_table_shape(self) -> None:
        """AdaLN table produces 6 modulation vectors of hidden size."""
        layer = DecoderLayer(hidden_size=HIDDEN, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        assert layer.adaln_table.shape == (6, HIDDEN)

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.26495734, -0.48088220, 1.40348279, -0.17601027, -1.48331332])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded decoder-layer forward pins a slice of its output."""
        torch.manual_seed(0)
        layer = DecoderLayer(hidden_size=HIDDEN, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        hidden, kv, pos, mask = _attn_inputs()
        t_embeds = torch.randn(BATCH, 6, HIDDEN)
        out = layer(hidden, kv, pos, t_embeds, attn_mask=mask)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


class TestDiT:
    """Tests for the DiT decoder stack."""
    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([1.01393712, 0.45370287, -0.99828362, -0.52321780, -0.47535068])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded DiT forward pins a slice of its output."""
        torch.manual_seed(0)
        dit = DiT(hidden_size=HIDDEN, layer_num=2, head_dim=HEAD_DIM, kv_heads=KV_HEADS)
        hidden, kv, pos, mask = _attn_inputs()
        past_key_values = [kv, (kv[0].clone(), kv[1].clone())]
        t_embeds = torch.randn(BATCH, 6, HIDDEN)
        out = dit(hidden, past_key_values, mask, pos, t_embeds)
        assert out.shape == (BATCH, SEQ, HIDDEN)
        assert torch.allclose(out[0, 0, :5], reference, **TOL), out.tolist()


# ============================================================================ #
# Rectified-flow action model (XR0FlowModel)                                   #
# ============================================================================ #

# Structural config for the XR0FlowModel tests.
FLOW_HIDDEN = 128
FLOW_HEAD_DIM = 128  # DiT default; hidden must be divisible by it.
FLOW_KV_HEADS = 1
LAYERS = 2
STATE_LEN = 1
ACTION_LEN = 4
STATE_DIM = 8
ACTION_DIM = 8
NUM_STEPS = 3


def _build_flow_model() -> XR0FlowModel:
    """Build a small fp32 flow model matching the shared config."""
    return XR0FlowModel(
        state_shape=(STATE_LEN, STATE_DIM),
        action_shape=(ACTION_LEN, ACTION_DIM),
        dit_num_layers=LAYERS,
        dit_hidden_size=FLOW_HIDDEN,
        dit_kv_heads=FLOW_KV_HEADS,
        num_steps=NUM_STEPS,
        dtype=torch.float32,
    )


def _flow_dit_inputs() -> dict:
    """Build small inputs for ``dit_forward`` / ``_flow_generate``."""
    q_len = 1 + STATE_LEN + ACTION_LEN  # sink + state + action
    ang = torch.randn(BATCH, q_len, FLOW_HEAD_DIM)
    q_causal = torch.tril(torch.ones(q_len, q_len, dtype=torch.bool))
    cache_ones = torch.ones(q_len, CACHE, dtype=torch.bool)
    mask = torch.cat([cache_ones, q_causal], dim=-1)[None, None].expand(BATCH, 1, q_len, CACHE + q_len)
    return {
        "noisy_action": torch.randn(BATCH, ACTION_LEN, ACTION_DIM),
        "t": torch.ones(BATCH, 1, 1) * 0.3,
        "action_mask": torch.ones(BATCH, ACTION_LEN, ACTION_DIM),
        "state_embed": torch.randn(BATCH, STATE_LEN, FLOW_HIDDEN),
        "cos": torch.cos(ang),
        "sin": torch.sin(ang),
        "past_key_values": [
            (
                torch.randn(BATCH, FLOW_KV_HEADS, CACHE, FLOW_HEAD_DIM),
                torch.randn(BATCH, FLOW_KV_HEADS, CACHE, FLOW_HEAD_DIM),
            )
            for _ in range(LAYERS)
        ],
        "attn_mask": mask.contiguous(),
    }


def _flow_dit_kwargs(i: dict) -> dict:
    """Assemble the ``dit_forward`` keyword args from an input bundle."""
    return {
        "action_mask": i["action_mask"],
        "state_embed": i["state_embed"],
        "position_embeds": (i["cos"], i["sin"]),
        "past_key_values": i["past_key_values"],
        "attn_mask": i["attn_mask"],
        "prefix_length": 0,
    }


class TestFlowMath:
    """Tests for the pure rectified-flow helpers."""

    def test_interpolate_endpoints(self) -> None:
        """t=0 returns x0 and t=1 returns x1."""
        model = _build_flow_model()
        x0 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        x1 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        torch.testing.assert_close(model._flow_interpolate(x0, x1, torch.zeros(BATCH, 1, 1)), x0)
        torch.testing.assert_close(model._flow_interpolate(x0, x1, torch.ones(BATCH, 1, 1)), x1)

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.39629608, -0.72900736, -0.29863209, -0.36008668, 0.27356121])],
    )
    def test_interpolate_formula(self, reference: torch.Tensor) -> None:
        """A seeded interpolation pins a slice of z_t = (1 - t)*x0 + t*x1."""
        torch.manual_seed(0)
        model = _build_flow_model()
        x0 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        x1 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        t = torch.rand(BATCH, 1, 1)
        out = model._flow_interpolate(x0, x1, t)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.48739055, -0.00960857, -0.06220679, -1.28131175, 2.19818282])],
    )
    def test_velocity_target(self, reference: torch.Tensor) -> None:
        """A seeded velocity target pins a slice of v = x1 - x0."""
        torch.manual_seed(0)
        model = _build_flow_model()
        x0 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        x1 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        out = model._flow_velocity_target(x0, x1)[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()

    def test_sample_timestep_beta_range(self) -> None:
        """Beta sampling yields shape (batch,) values in (0, 0.999]."""
        t = _build_flow_model()._sample_timestep(16, dtype=torch.float32)
        assert t.shape == (16,)
        assert torch.all(t > 0) and torch.all(t <= 0.999)

    def test_sample_timestep_uniform_fallback(self) -> None:
        """A non-beta/logit sampler falls back to Uniform(0, 1)."""
        model = _build_flow_model()
        model.flow_sampling = "uniform"
        t = model._sample_timestep(16, dtype=torch.float32)
        assert torch.all(t >= 0) and torch.all(t < 1)


class TestDitForward:
    """Tests for a single DiT velocity prediction."""

    def test_output_shape(self) -> None:
        """dit_forward returns (B, action_len, action_dim)."""
        model = _build_flow_model()
        i = _flow_dit_inputs()
        out = model.dit_forward(i["noisy_action"], i["t"], **_flow_dit_kwargs(i))
        assert out.shape == (BATCH, ACTION_LEN, ACTION_DIM)

    def test_prefix_length_zeroes_leading_actions(self) -> None:
        """A positive prefix_length forces the leading action tokens to zero."""
        model = _build_flow_model()
        i = _flow_dit_inputs()
        kwargs = _flow_dit_kwargs(i)
        kwargs["prefix_length"] = 2
        out = model.dit_forward(i["noisy_action"], i["t"], **kwargs)
        torch.testing.assert_close(out[:, :2], torch.zeros(BATCH, 2, ACTION_DIM))

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([0.15467618, 0.23847848, 0.09961469, -0.11562672, 0.05315172])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded dit_forward pins a slice of its velocity output."""
        torch.manual_seed(0)
        model = _build_flow_model()
        i = _flow_dit_inputs()
        out = model.dit_forward(i["noisy_action"], i["t"], **_flow_dit_kwargs(i))[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()


class TestFlowGenerate:
    """Tests for the Euler integration inference loop."""

    @pytest.mark.parametrize(
        "reference",
        [torch.tensor([-1.96027327, -0.91721338, 0.38879472, -0.24504544, 0.8044914])],
    )
    def test_reference(self, reference: torch.Tensor) -> None:
        """A seeded _flow_generate pins a slice of its Euler-integrated output."""
        torch.manual_seed(0)
        model = _build_flow_model()
        i = _flow_dit_inputs()
        x0 = torch.randn(BATCH, ACTION_LEN, ACTION_DIM)
        out = model._flow_generate(x0, _flow_dit_kwargs(i))[0, 0, :5]
        assert torch.allclose(out, reference, **TOL), out.tolist()
