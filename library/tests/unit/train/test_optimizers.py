# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared optimizer factories."""

import pytest
import torch

from physicalai.train.optimizers import AdamWBF16Kahan, adamw_bf16_kahan, build_adamw_auto_precision

_ACCELERATOR = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else None


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
        optimizer = build_adamw_auto_precision(_params(torch.float32), lr=1e-5, fused=False)
        assert type(optimizer) is torch.optim.AdamW

    def test_bfloat16_uses_bf16_adamw(self) -> None:
        bf16, fp32 = _params(torch.bfloat16), _params(torch.float32)
        optimizer = build_adamw_auto_precision(
            [{"params": fp32, "weight_decay": 0.0}, {"params": bf16}],
            lr=1e-5,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            fused=True,
        )
        assert isinstance(optimizer, AdamWBF16Kahan)
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

        for actual, expected in zip(train(AdamWBF16Kahan), train(torch.optim.AdamW), strict=True):
            torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("chunk_numel", [None, 1000])
    def test_bfloat16_tracks_float32_weights(self, monkeypatch: pytest.MonkeyPatch, chunk_numel: int | None) -> None:
        if chunk_numel is not None:
            monkeypatch.setattr(adamw_bf16_kahan, "_CHUNK_NUMEL", chunk_numel)
        kwargs = {"lr": 1e-5, "betas": (0.9, 0.95), "weight_decay": 0.01}
        reference, plain, compensated = _params(torch.float32), _params(torch.bfloat16), _params(torch.bfloat16)
        _train(reference, torch.optim.AdamW(reference, **kwargs), steps=50)
        _train(plain, torch.optim.AdamW(plain, **kwargs), steps=50)
        optimizer = build_adamw_auto_precision(compensated, **kwargs)
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

        optimizer = build_adamw_auto_precision(params, lr=1e-3)
        optimizer.load_state_dict(source.state_dict())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
        _train(params, optimizer, steps=1)
        scheduler.step()

        assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)
        assert optimizer.state[params[0]]["step"].item() == 2
        assert optimizer.state[params[0]]["compensation"].dtype == torch.bfloat16

    def test_generator_param_groups_keep_all_params(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4).bfloat16())
        optimizer = build_adamw_auto_precision([{"params": model[0].parameters()}, {"params": model[1].parameters()}], lr=1e-3)
        assert [len(group["params"]) for group in optimizer.param_groups] == [2, 2]

        fp32_only = build_adamw_auto_precision([{"params": model[0].parameters()}], lr=1e-3)
        assert len(fp32_only.param_groups[0]["params"]) == 2

    def test_unsupported_options_raise_for_bfloat16(self) -> None:
        assert isinstance(build_adamw_auto_precision(_params(torch.bfloat16), lr=1e-3, amsgrad=False, foreach=False), AdamWBF16Kahan)
        with pytest.raises(ValueError, match="amsgrad"):
            build_adamw_auto_precision(_params(torch.bfloat16), lr=1e-3, amsgrad=True)

    def test_default_beta2_tracks_float32_second_moment(self) -> None:
        params = _params(torch.bfloat16)
        optimizer = build_adamw_auto_precision(params, lr=1e-5)  # default betas (0.9, 0.999)
        for _ in range(2000):
            params[0].grad = torch.ones_like(params[0])
            optimizer.step()

        state = optimizer.state[params[0]]
        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.float32
        torch.testing.assert_close(state["exp_avg_sq"], torch.full_like(state["exp_avg_sq"], 1 - 0.999**2000))

        restored = build_adamw_auto_precision(params, lr=1e-5)
        restored.load_state_dict(optimizer.state_dict())
        assert torch.equal(restored.state[params[0]]["exp_avg_sq"], state["exp_avg_sq"])

    @pytest.mark.skipif(_ACCELERATOR is None, reason="Triton kernel runs on CUDA or XPU")
    @pytest.mark.parametrize("betas", [(0.9, 0.95), (0.9, 0.999)])
    def test_fused_kernel_matches_eager(self, monkeypatch: pytest.MonkeyPatch, betas: tuple[float, float]) -> None:
        pytest.importorskip("triton")
        init = (torch.randn(1031, 517, device=_ACCELERATOR) * 0.02).bfloat16()

        def train(fused: bool) -> tuple[torch.Tensor, dict]:
            if not fused:
                monkeypatch.setattr(adamw_bf16_kahan, "_fused_update", lambda: None)
            params = [torch.nn.Parameter(init.clone())]
            optimizer = build_adamw_auto_precision(params, lr=1e-5, betas=betas, weight_decay=0.01)
            generator = torch.Generator(device=_ACCELERATOR).manual_seed(0)
            for _ in range(20):
                params[0].grad = torch.randn(init.shape, device=_ACCELERATOR, generator=generator).bfloat16()
                optimizer.step()
            return params[0].detach(), optimizer.state[params[0]]

        fused_weight, fused_state = train(fused=True)
        eager_weight, eager_state = train(fused=False)
        fused_effective = fused_weight.float() + fused_state["compensation"].float()
        eager_effective = eager_weight.float() + eager_state["compensation"].float()
        moved = (eager_effective - init.float()).abs().max()
        assert (fused_effective - eager_effective).abs().max() < 0.02 * moved
        assert fused_state["exp_avg_sq"].dtype == eager_state["exp_avg_sq"].dtype
        assert (fused_state["compensation"] != 0).any()
