# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared optimizer factories."""

import pytest
import torch

from physicalai.train import optimizers
from physicalai.train.optimizers import BF16AdamW, adamw


def _params(dtype: torch.dtype) -> list[torch.nn.Parameter]:
    # Start from a bf16-representable value so every dtype begins at the same weights.
    return [torch.nn.Parameter(torch.full((64, 64), 0.02).bfloat16().to(dtype))]


def _train(params: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer, steps: int) -> None:
    generator = torch.Generator().manual_seed(0)
    for _ in range(steps):
        for p in params:
            p.grad = torch.randn(p.shape, generator=generator).bfloat16().to(p.dtype)  # same values for every dtype
        optimizer.step()


class TestAdamW:
    def test_float32_uses_torch_adamw(self) -> None:
        optimizer = adamw(_params(torch.float32), lr=1e-5, fused=False)
        assert type(optimizer) is torch.optim.AdamW

    def test_bfloat16_uses_bf16_adamw(self) -> None:
        bf16, fp32 = _params(torch.bfloat16), _params(torch.float32)
        optimizer = adamw(
            [{"params": fp32, "weight_decay": 0.0}, {"params": bf16}],
            lr=1e-5,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            fused=True,
        )
        assert isinstance(optimizer, BF16AdamW)
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
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
            for _ in range(3):
                for p in params:
                    p.grad = torch.randn_like(p)
                optimizer.step()
                scheduler.step()
            return [p.detach() for p in params]

        for actual, expected in zip(train(BF16AdamW), train(torch.optim.AdamW), strict=True):
            torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("chunk_numel", [None, 1000])
    def test_bfloat16_tracks_float32_weights(self, monkeypatch: pytest.MonkeyPatch, chunk_numel: int | None) -> None:
        if chunk_numel is not None:
            monkeypatch.setattr(optimizers, "_CHUNK_NUMEL", chunk_numel)
        kwargs = {"lr": 1e-5, "betas": (0.9, 0.95), "weight_decay": 0.01}
        reference, plain, compensated = _params(torch.float32), _params(torch.bfloat16), _params(torch.bfloat16)
        _train(reference, torch.optim.AdamW(reference, **kwargs), steps=50)
        _train(plain, torch.optim.AdamW(plain, **kwargs), steps=50)
        optimizer = adamw(compensated, **kwargs)
        _train(compensated, optimizer, steps=50)

        start = torch.tensor(0.02).bfloat16().float()
        assert (plain[0].float() == start).all(), "plain bf16 AdamW should round these updates away"
        effective = compensated[0].float() + optimizer.state[compensated[0]]["compensation"].float()
        # Moments are stored in bf16, so allow 1% of the fp32 run's total movement.
        moved = (reference[0].detach() - start).abs().max()
        assert (effective - reference[0].detach()).abs().max() < 0.01 * moved
        assert (compensated[0].float() != start).any()

    def test_resumes_from_torch_adamw_state(self) -> None:
        params = _params(torch.bfloat16)
        source = torch.optim.AdamW(params, lr=1e-5)
        _train(params, source, steps=1)

        optimizer = adamw(params, lr=1e-3)
        optimizer.load_state_dict(source.state_dict())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
        _train(params, optimizer, steps=1)
        scheduler.step()

        assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)
        assert optimizer.state[params[0]]["step"].item() == 2
        assert optimizer.state[params[0]]["compensation"].dtype == torch.bfloat16
