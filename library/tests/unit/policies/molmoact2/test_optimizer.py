# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MolmoAct2 optimizer."""

from __future__ import annotations

import torch

from physicalai.policies.molmoact2.optimizer import MolmoAct2AdamW


def test_updates_float32_parameters() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([1.0])
    optimizer = MolmoAct2AdamW([parameter], lr=0.1, group_grad_clip_norm=1.0)

    optimizer.step()

    assert parameter.item() < 1.0


def test_clips_each_parameter_group_independently() -> None:
    first = torch.nn.Parameter(torch.zeros(1))
    second = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([10.0])
    second.grad = torch.tensor([20.0])
    optimizer = MolmoAct2AdamW(
        [{"params": [first]}, {"params": [second]}],
        lr=0.1,
        group_grad_clip_norm=2.0,
    )

    optimizer._clip_grad_groups()

    torch.testing.assert_close(first.grad, torch.tensor([2.0]))
    torch.testing.assert_close(second.grad, torch.tensor([2.0]))


def test_bfloat16_updates_keep_compensation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    parameter.grad = torch.tensor([1.0], dtype=torch.bfloat16)
    optimizer = MolmoAct2AdamW([parameter], lr=0.1, group_grad_clip_norm=1.0)

    optimizer.step()

    assert "compensation" in optimizer.state[parameter]
    assert optimizer.state[parameter]["compensation"].dtype == torch.bfloat16


def test_nonfinite_gradient_skips_update_and_clears_grad() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])
    optimizer = MolmoAct2AdamW([parameter], lr=0.1, group_grad_clip_norm=1.0)

    optimizer.step()

    torch.testing.assert_close(parameter, torch.tensor([1.0]))
    assert parameter.grad is None
    assert not optimizer.state[parameter]
