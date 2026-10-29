# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared optimizer factories."""

import pytest
import torch
from torchao.optim import _AdamW

from physicalai.train.optimizers import _bf16_adamw_class, adamw

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="torchao compiles its step; CUDA keeps this fast")


def _params(dtype: torch.dtype, device: str = "cpu") -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(torch.full((256, 256), 0.02, dtype=dtype, device=device))]


class TestAdamW:
    def test_float32_uses_torch_adamw(self) -> None:
        optimizer = adamw(_params(torch.float32), lr=1e-5, fused=False)
        assert type(optimizer) is torch.optim.AdamW

    def test_bfloat16_uses_stochastic_rounding(self) -> None:
        bf16, fp32 = _params(torch.bfloat16), _params(torch.float32)
        optimizer = adamw(
            [{"params": fp32, "weight_decay": 0.0}, {"params": bf16}],
            lr=1e-5,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            fused=True,
        )
        assert isinstance(optimizer, _AdamW)
        assert optimizer.bf16_stochastic_round
        assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0, 0.01]
        assert all(group["betas"] == (0.9, 0.95) for group in optimizer.param_groups)

    def test_non_bfloat16_params_match_torch_adamw(self) -> None:
        def train(make: type) -> list[torch.Tensor]:
            torch.manual_seed(0)
            params = [torch.nn.Parameter(torch.randn(8, 8)), torch.nn.Parameter(torch.randn(8))]
            optimizer = make(
                [{"params": params[:1]}, {"params": params[1:], "weight_decay": 0.0}],
                lr=1e-2,
                betas=(0.9, 0.95),
                weight_decay=0.1,
                amsgrad=True,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
            for _ in range(3):
                for p in params:
                    p.grad = torch.randn_like(p)
                optimizer.step()
                scheduler.step()
            return [p.detach() for p in params]

        for actual, expected in zip(train(_bf16_adamw_class()), train(torch.optim.AdamW), strict=True):
            torch.testing.assert_close(actual, expected)

    @cuda
    def test_mixed_dtypes_update_both(self) -> None:
        bf16, fp32 = _params(torch.bfloat16, "cuda"), _params(torch.float32, "cuda")
        optimizer = adamw([*bf16, *fp32], lr=1e-5)
        for p in [*bf16, *fp32]:
            p.grad = torch.ones_like(p)
        optimizer.step()
        assert (bf16[0] != 0.02).any()
        assert (fp32[0] < 0.02).all()
        assert all(optimizer.state[p]["step"].item() == 1 for p in [*bf16, *fp32])

    @cuda
    def test_small_bfloat16_updates_survive(self) -> None:
        def train(make: type | None) -> torch.Tensor:
            params = _params(torch.bfloat16, "cuda")
            optimizer = adamw(params, lr=1e-5) if make is None else make(params, lr=1e-5)
            for _ in range(5):
                params[0].grad = torch.ones_like(params[0])
                optimizer.step()
            return params[0].detach().float()

        start = torch.full((256, 256), 0.02, dtype=torch.bfloat16, device="cuda").float()
        assert torch.equal(train(torch.optim.AdamW), start), "Plain AdamW should round these updates away"
        moved = (train(None) != start).float().mean().item()
        assert moved > 0.1

    @cuda
    def test_resumes_from_torch_adamw_state(self) -> None:
        params = _params(torch.bfloat16, "cuda")
        source = torch.optim.AdamW(params, lr=1e-5)
        params[0].grad = torch.ones_like(params[0])
        source.step()

        optimizer = adamw(params, lr=1e-3)
        optimizer.load_state_dict(source.state_dict())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
        optimizer.step()
        scheduler.step()

        assert isinstance(optimizer.param_groups[0]["lr"], torch.Tensor)
        assert optimizer.param_groups[0]["lr"].item() == pytest.approx(5e-6)
        assert optimizer.state[params[0]]["step"].item() == 2
