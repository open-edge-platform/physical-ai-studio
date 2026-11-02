# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Check memory-efficient attention against the original PATCH equations."""

import copy
import math

import pytest
import torch

from physicalai.policies.patch_policy.components.gpt import CausalSelfAttention, GPTConfig


@pytest.mark.parametrize("training", [False, True])
def test_attention_matches_original_outputs_and_gradients(training: bool) -> None:
    torch.manual_seed(7)
    attention = CausalSelfAttention(GPTConfig(n_embd=8, n_head=2, n_patches=3, block_size=4, dropout=0.0))
    attention.train(training)
    reference = copy.deepcopy(attention)
    inputs = torch.randn(2, 9, 8, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()

    actual = attention(inputs)
    q, k, v = reference.c_attn(reference_inputs).split(8, dim=2)
    q, k, v = [value.reshape(2, 9, 2, 4).transpose(1, 2) for value in (q, k, v)]
    weights = (q @ k.transpose(-2, -1)) / math.sqrt(4)
    weights = weights.masked_fill(reference.bias[:, :, :9, :9] == 0, float("-inf")).softmax(-1)
    expected = reference.c_proj((weights @ v).transpose(1, 2).contiguous().reshape(2, 9, 8))

    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    for actual_parameter, expected_parameter in zip(attention.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


def test_attention_keeps_same_timestep_visible_and_future_hidden() -> None:
    torch.manual_seed(7)
    attention = CausalSelfAttention(GPTConfig(n_embd=8, n_head=2, n_patches=3, block_size=2, dropout=0.5)).eval()
    inputs = torch.randn(1, 6, 8)
    expected = attention(inputs)
    torch.testing.assert_close(attention(inputs), expected)
    changed = inputs.clone()
    changed[:, 3:] += 10
    torch.testing.assert_close(attention(changed)[:, :3], expected[:, :3])
    changed = inputs.clone()
    changed[:, 2] += 10
    assert not torch.allclose(attention(changed)[:, 0], expected[:, 0])
