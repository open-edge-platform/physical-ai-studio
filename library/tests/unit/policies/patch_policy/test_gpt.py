# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small, real block-causal transformer tests."""

import pytest
import torch

from physicalai.policies.patch_policy.components.gpt import (
    CausalSelfAttention,
    GPTConfig,
    PatchGPT,
    generate_mask_matrix,
)


def _make_small_gpt() -> PatchGPT:
    torch.manual_seed(7)
    return PatchGPT(2, 3, n_patches=2, n_obs_steps=2, n_layer=1, n_head=2, n_embd=4).eval()


def test_mask_allows_all_current_patches_but_no_future_windows() -> None:
    expected = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    torch.testing.assert_close(generate_mask_matrix(2, 2), expected[None, None])


def test_attention_averages_visible_patches_with_uniform_scores() -> None:
    attention = CausalSelfAttention(GPTConfig(block_size=2, n_patches=2, n_embd=2, n_head=1, dropout=0.0))
    with torch.no_grad():
        attention.c_attn.weight.zero_()
        attention.c_attn.weight[4:].copy_(torch.eye(2))
        attention.c_attn.bias.zero_()
        attention.c_proj.weight.copy_(torch.eye(2))
        attention.c_proj.bias.zero_()
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    expected = torch.tensor([[[2.0, 3.0], [2.0, 3.0], [4.0, 5.0], [4.0, 5.0]]])
    torch.testing.assert_close(attention(tokens), expected)


def test_future_tokens_cannot_change_past_outputs() -> None:
    gpt = _make_small_gpt()
    tokens = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[2.0, 3.0], [4.0, 5.0]]]])
    changed = tokens.clone()
    changed[:, 1] = torch.tensor([[[-4.0, 2.0], [3.0, -5.0]]])
    original_output = gpt(tokens)
    changed_output = gpt(changed)
    torch.testing.assert_close(original_output[:, 0], changed_output[:, 0])
    assert not torch.allclose(original_output[:, 1], changed_output[:, 1])


def test_output_selects_last_patch_and_backpropagates() -> None:
    gpt = _make_small_gpt()
    tokens = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], requires_grad=True)
    output = gpt(tokens)
    positions = gpt.transformer.wpe(torch.tensor([[0, 1, 2, 3]])).reshape(1, 2, 2, 4)
    hidden = (gpt.transformer.wte(tokens) + positions).reshape(1, 4, 4)
    hidden = gpt.transformer.h[0](hidden)
    all_patches = gpt.lm_head(gpt.transformer.ln_f(hidden)).reshape(1, 2, 2, 3)
    assert output.shape == (1, 2, 3)
    torch.testing.assert_close(output, all_patches[:, :, -1])
    output.square().sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert tokens.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("shape", "message"),
    [((1, 2, 2), "Expected tokens"), ((1, 3, 2, 2), "block size"), ((1, 2, 1, 2), "patches per timestep")],
)
def test_rejects_incompatible_token_shapes(shape: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _make_small_gpt()(torch.zeros(shape))


def test_crop_preserves_prefix_outputs_and_trims_masks() -> None:
    gpt = _make_small_gpt()
    tokens = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    expected = gpt(tokens)
    gpt.crop_block_size(1)
    assert gpt.config.block_size == 1
    assert gpt.transformer.wpe.weight.shape == (2, 4)
    assert gpt.transformer.h[0].attn.bias.shape == (1, 1, 2, 2)
    torch.testing.assert_close(gpt(tokens), expected)
    with pytest.raises(ValueError, match="larger block size"):
        gpt.crop_block_size(2)
