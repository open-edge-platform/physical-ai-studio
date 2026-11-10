# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RLDX-1 export-time f16 workarounds on a tiny action head."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import BatchFeature

from physicalai.policies.rldx1.constants import ACTION_PRED, BACKBONE_FEATURES
from physicalai.policies.rldx1.model import RLDXActionModel
from physicalai.policies.rldx1.utils.export import f16_safe_residual_stream, untie_first_denoising_step

# Denoising steps per forward in the tiny head; two forwards are run.
_NUM_STEPS = 3


class _Head(nn.Module):
    def __init__(self, action_model: RLDXActionModel) -> None:
        super().__init__()
        self.action_model = action_model

    def forward(self, backbone_features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        embodiment_id = torch.zeros(1, dtype=torch.long)
        action_input = self.action_model.prepare_input({"state": state, "embodiment_id": embodiment_id})
        out = self.action_model.get_action(BatchFeature(data={BACKBONE_FEATURES: backbone_features}), action_input)
        return out[ACTION_PRED]


@pytest.fixture
def head() -> _Head:
    """Tiny RLDX-1 action head with random weights.

    Returns:
        The head in eval mode.
    """
    torch.manual_seed(0)
    action_model = RLDXActionModel(
        hidden_size=32,
        input_embedding_dim=32,
        backbone_embedding_dim=48,
        max_action_dim=4,
        action_horizon=3,
        num_inference_timesteps=_NUM_STEPS,
        max_num_embodiments=2,
        max_state_dim=4,
        use_vlln=False,
        add_pos_embed=True,
        max_seq_len=16,
        state_dropout_prob=0.0,
        state_additive_noise_scale=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        diffusion_model_cfg={
            "attention_head_dim": 8,
            "num_attention_heads": 4,
            "depth_multi_stream": 2,
            "depth_single_stream": 2,
            "output_dim": 32,
            "dropout": 0.0,
            "action_model_max_seq_len": 64,
        },
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        backbone_trainable_params_fp32=False,
    )
    return _Head(action_model).eval()


def _run(head: _Head) -> torch.Tensor:
    torch.manual_seed(1)
    inputs = torch.randn(1, 5, 48), torch.randn(1, 1, 4)
    torch.manual_seed(2)
    with torch.no_grad():
        return head(*inputs)


def test_f16_safe_residual_stream_is_exact_and_restores(head: _Head) -> None:
    """Rescaling shrinks the residual stream, keeps the output, and is fully undone."""
    msat = head.action_model.model
    state_before = {k: v.clone() for k, v in msat.state_dict().items()}
    eps_before = [m.eps for m in msat.modules() if isinstance(m, nn.LayerNorm)]
    captured: list[float] = []
    w3 = msat.double_blocks[0].sa_mlp.w3
    hook = w3.register_forward_hook(lambda _m, _a, out: captured.append(out.abs().max().item()))
    reference = _run(head)

    with f16_safe_residual_stream(msat, scale=16.0):
        rescaled = _run(head)
    hook.remove()

    torch.testing.assert_close(rescaled, reference, rtol=1e-5, atol=1e-5)
    assert captured[len(captured) // 2] == pytest.approx(captured[0] / 16.0, rel=1e-5)
    assert all(torch.equal(v, msat.state_dict()[k]) for k, v in state_before.items())
    assert [m.eps for m in msat.modules() if isinstance(m, nn.LayerNorm)] == eps_before
    assert not msat._forward_pre_hooks  # noqa: SLF001
    assert not msat.time_token_proj._forward_hooks  # noqa: SLF001


def test_untie_first_denoising_step_is_exact_and_restores(head: _Head) -> None:
    """Only the first denoising step per forward reads the copied weights."""
    msat = head.action_model.model
    reference = _run(head)
    used: list[bool] = []

    with untie_first_denoising_step(head, msat):
        wrapper = msat.double_blocks[0].vl_mlp.w3
        assert wrapper.first_weight is not wrapper.linear.weight
        hook = wrapper.linear.register_forward_hook(lambda *_: used.append(True))
        untied = _run(head)
        untied_again = _run(head)
        hook.remove()

    torch.testing.assert_close(untied, reference)
    torch.testing.assert_close(untied_again, reference)
    assert len(used) == 2 * (_NUM_STEPS - 1)
    assert all(type(block.vl_mlp.w3) is nn.Linear for block in msat.double_blocks)
