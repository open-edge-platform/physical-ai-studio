# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the diffusion action head.

Most tests use an oracle denoiser that knows the clean chunk ``x_0``. With a
perfect prediction the maths of diffusion gives exact answers: the forward
process has a closed form, a DDIM step lands on the forward process at the
next timestep, a DDPM step follows the posterior ``q(x_{t-1} | x_t, x_0)``, and
sampling recovers ``x_0``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import nn

from physicalai.policies.components import DiffusionActionHead
from physicalai.policies.components.action_heads import Context, make_betas

CHUNK_SIZE, ACTION_DIM, CONTEXT_DIM, NUM_TRAIN_TIMESTEPS = 4, 3, 8, 100


def alpha_bar() -> torch.Tensor:
    """Reference alpha_bar_t = prod_{s <= t} (1 - beta_s) of the default cosine schedule, in float64."""
    return torch.cumprod(1 - make_betas("squaredcos_cap_v2", NUM_TRAIN_TIMESTEPS).double(), dim=0)


class ToyDiffusionHead(DiffusionActionHead):
    """Diffusion head with a single linear layer as the denoiser."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(CHUNK_SIZE, ACTION_DIM, num_train_timesteps=NUM_TRAIN_TIMESTEPS, **kwargs)
        self.net = nn.Linear(ACTION_DIM + CONTEXT_DIM + 1, ACTION_DIM)

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, context: Context) -> torch.Tensor:
        pooled = context["tokens"].mean(dim=1, keepdim=True).expand(-1, x_t.shape[1], -1)
        time = (t.to(x_t.dtype) / self.num_train_timesteps)[:, None, None].expand(-1, x_t.shape[1], 1)
        return self.net(torch.cat([x_t, pooled, time], dim=-1))


class OracleDiffusionHead(DiffusionActionHead):
    """Perfect denoiser: reads the clean chunk from ``context["target"]`` and predicts exactly."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(CHUNK_SIZE, ACTION_DIM, num_train_timesteps=NUM_TRAIN_TIMESTEPS, **kwargs)
        self.reference_alpha_bar = alpha_bar().float()

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, context: Context) -> torch.Tensor:
        x0 = context["target"]
        if self.prediction_type == "sample":
            return x0
        ab = self.reference_alpha_bar[t][:, None, None]
        return (x_t - ab.sqrt() * x0) / (1 - ab).sqrt()  # the noise that produced x_t


def forward_process(x0: torch.Tensor, noise: torch.Tensor, t: int) -> torch.Tensor:
    """x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps, with x_{-1} = x_0."""
    ab = 1.0 if t < 0 else alpha_bar()[t].item()
    return math.sqrt(ab) * x0 + math.sqrt(1 - ab) * noise


@pytest.fixture
def context() -> Context:
    """Context with a batch of 2 and 6 tokens."""
    return {"tokens": torch.randn(2, 6, CONTEXT_DIM)}


@pytest.fixture
def target() -> torch.Tensor:
    """A clean chunk inside the clip range."""
    return torch.rand(2, CHUNK_SIZE, ACTION_DIM) * 1.6 - 0.8


class TestSchedule:
    """The noise schedule follows its definitions."""

    def test_linear_betas(self) -> None:
        """Test linear betas run evenly from beta_start to beta_end."""
        betas = make_betas("linear", NUM_TRAIN_TIMESTEPS, beta_start=1e-4, beta_end=0.02)
        torch.testing.assert_close(betas, torch.linspace(1e-4, 0.02, NUM_TRAIN_TIMESTEPS))

    def test_scaled_linear_betas(self) -> None:
        """Test sqrt(beta) runs evenly from sqrt(beta_start) to sqrt(beta_end)."""
        betas = make_betas("scaled_linear", NUM_TRAIN_TIMESTEPS, beta_start=1e-4, beta_end=0.02)
        torch.testing.assert_close(betas.sqrt(), torch.linspace(1e-2, 0.02**0.5, NUM_TRAIN_TIMESTEPS))

    def test_cosine_alpha_bar(self) -> None:
        """Test alpha_bar_t = f(t + 1) / f(0) with f(s) = cos^2((s / T + 0.008) / 1.008 * pi / 2)."""
        betas = make_betas("squaredcos_cap_v2", NUM_TRAIN_TIMESTEPS)
        assert (betas > 0).all()
        assert (betas <= 0.999).all()

        def f(s: torch.Tensor) -> torch.Tensor:
            return torch.cos((s / NUM_TRAIN_TIMESTEPS + 0.008) / 1.008 * math.pi / 2) ** 2

        steps = torch.arange(1, NUM_TRAIN_TIMESTEPS + 1, dtype=torch.float64)
        expected = f(steps) / f(torch.zeros(1, dtype=torch.float64))
        uncapped = betas < 0.999  # the cap only changes the last step
        torch.testing.assert_close(alpha_bar()[uncapped], expected[uncapped], rtol=1e-5, atol=1e-7)

    def test_buffers(self) -> None:
        """Test the buffers store sqrt(alpha_bar) and sqrt(1 - alpha_bar), with the clean data at index 0."""
        head = ToyDiffusionHead()
        torch.testing.assert_close(head.sqrt_alpha_bar[1:].double() ** 2, alpha_bar())
        torch.testing.assert_close(head.sqrt_alpha_bar**2 + head.sqrt_one_minus_alpha_bar**2, torch.ones(101))
        assert head.sqrt_alpha_bar[0] == 1
        assert head.sqrt_one_minus_alpha_bar[0] == 0
        assert (head.sqrt_alpha_bar.diff() < 0).all()  # noise grows with t

    def test_timesteps(self) -> None:
        """Test the schedule runs from noise to the clean-data sentinel -1."""
        head = ToyDiffusionHead()
        timesteps = head.timesteps(10, device=torch.device("cpu"), dtype=torch.float32)
        assert timesteps.tolist() == [90, 80, 70, 60, 50, 40, 30, 20, 10, 0, -1]
        assert head.timesteps(NUM_TRAIN_TIMESTEPS, torch.device("cpu"), torch.float32)[0] == NUM_TRAIN_TIMESTEPS - 1

    def test_invalid_arguments(self) -> None:
        """Test out-of-range arguments are rejected."""
        with pytest.raises(ValueError, match="num_steps"):
            ToyDiffusionHead(num_inference_steps=NUM_TRAIN_TIMESTEPS + 1)
        with pytest.raises(ValueError, match="prediction_type"):
            ToyDiffusionHead(prediction_type="v_prediction")
        with pytest.raises(ValueError, match="beta_schedule"):
            ToyDiffusionHead(beta_schedule="sigmoid")


class TestForwardProcess:
    """add_noise samples q(x_t | x_0)."""

    def test_closed_form(self, target: torch.Tensor) -> None:
        """Test add_noise matches sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps."""
        head = ToyDiffusionHead()
        noise = torch.randn_like(target)
        for t in (0, 1, 50, 99):
            actual = head.add_noise(target, noise, torch.full((2,), t))
            torch.testing.assert_close(actual, forward_process(target, noise, t))

    def test_preserves_variance(self) -> None:
        """Test unit-variance data stays unit variance at every noise level."""
        head = ToyDiffusionHead()
        x0, noise = torch.randn(20000, CHUNK_SIZE, ACTION_DIM), torch.randn(20000, CHUNK_SIZE, ACTION_DIM)
        for t in (0, 50, 99):
            x_t = head.add_noise(x0, noise, torch.full((20000,), t))
            assert abs(x_t.var().item() - 1) < 0.02


class TestSampling:
    """With a perfect denoiser, sampling follows the diffusion maths exactly."""

    @pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
    def test_ddim_step_lands_on_forward_process(self, target: torch.Tensor, prediction_type: str) -> None:
        """Test a DDIM step from x_t = q(x_t | x_0, eps) lands on the same x_0 and eps at t_next."""
        head = OracleDiffusionHead(prediction_type=prediction_type, eta=0.0)
        noise = torch.randn_like(target)
        for t, t_next in [(99, 89), (50, 40), (10, 0), (0, -1)]:
            x_t = forward_process(target, noise, t)
            output = head.denoise(x_t, torch.full((2,), t), {"target": target})
            actual = head.step(x_t, output, torch.tensor(t), torch.tensor(t_next))
            torch.testing.assert_close(actual, forward_process(target, noise, t_next), rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("t", [99, 50, 1])
    def test_ddpm_step_is_the_posterior(self, target: torch.Tensor, t: int) -> None:
        """Test eta = 1 samples q(x_{t-1} | x_t, x_0) from Ho et al. (2020), equations 6 and 7."""
        head = OracleDiffusionHead(eta=1.0)
        ab = alpha_bar()
        ab_t, ab_prev = ab[t].item(), ab[t - 1].item()
        beta_t = 1 - ab_t / ab_prev
        x_t = forward_process(target, torch.randn_like(target), t)

        mean = (math.sqrt(ab_prev) * beta_t / (1 - ab_t)) * target + (
            math.sqrt(1 - beta_t) * (1 - ab_prev) / (1 - ab_t)
        ) * x_t
        std = math.sqrt((1 - ab_prev) / (1 - ab_t) * beta_t)

        output = head.denoise(x_t, torch.full((2,), t), {"target": target})
        torch.manual_seed(0)
        actual = head.step(x_t, output, torch.tensor(t), torch.tensor(t - 1))
        torch.manual_seed(0)
        expected = mean + std * torch.randn_like(x_t)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("num_steps", [10, NUM_TRAIN_TIMESTEPS])
    @pytest.mark.parametrize("eta", [0.0, 1.0])
    @pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
    def test_sample_recovers_target(
        self,
        target: torch.Tensor,
        num_steps: int,
        eta: float,
        prediction_type: str,
    ) -> None:
        """Test DDPM and DDIM both end exactly at x_0 when the denoiser is perfect."""
        head = OracleDiffusionHead(num_inference_steps=num_steps, eta=eta, prediction_type=prediction_type)
        torch.testing.assert_close(head.sample({"target": target}), target, rtol=1e-4, atol=1e-4)

    def test_clip_sample(self, target: torch.Tensor) -> None:
        """Test a predicted x_0 outside the clip range is clamped, so sampling ends at the bound."""
        head = OracleDiffusionHead(prediction_type="sample", eta=0.0, clip_sample_range=0.5)
        out_of_range = torch.full_like(target, 2.0)
        torch.testing.assert_close(head.sample({"target": out_of_range}), torch.full_like(target, 0.5))

    def test_ddim_is_deterministic(self, context: Context) -> None:
        """Test eta = 0 sampling is a deterministic function of the initial noise."""
        head = ToyDiffusionHead(num_inference_steps=10, eta=0.0)
        noise = torch.randn(2, CHUNK_SIZE, ACTION_DIM)
        torch.testing.assert_close(head.sample(context, noise=noise), head.sample(context, noise=noise))

    def test_zero_input_noise(self, context: Context) -> None:
        """Test use_random_input_noise=False starts from zeros."""
        head = ToyDiffusionHead(num_inference_steps=10, eta=0.0, use_random_input_noise=False)
        zeros = torch.zeros(2, CHUNK_SIZE, ACTION_DIM)
        torch.testing.assert_close(head.sample(context), head.sample(context, noise=zeros))


class TestTraining:
    """compute_loss regresses the noise (or x_0) of the forward process."""

    @pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
    def test_perfect_denoiser_has_zero_loss(self, target: torch.Tensor, prediction_type: str) -> None:
        """Test the loss target is eps (or x_0), so a perfect prediction scores zero."""
        head = OracleDiffusionHead(prediction_type=prediction_type)
        losses = head.compute_loss(target, {"target": target})
        assert losses.shape == (2, CHUNK_SIZE, ACTION_DIM)
        torch.testing.assert_close(losses, torch.zeros_like(losses), rtol=0, atol=1e-8)

    def test_loss_backpropagates(self, context: Context) -> None:
        """Test the loss is unreduced and backpropagates into the denoiser."""
        head = ToyDiffusionHead()
        losses = head.compute_loss(torch.rand(2, CHUNK_SIZE, ACTION_DIM) * 2 - 1, context)
        assert losses.shape == (2, CHUNK_SIZE, ACTION_DIM)
        losses.mean().backward()
        assert head.net.weight.grad is not None

    def test_schedule_not_in_state_dict(self) -> None:
        """Test the schedule buffers stay out of checkpoints."""
        assert set(ToyDiffusionHead().state_dict()) == {"net.weight", "net.bias"}


class TestDtypes:
    """The head runs in the module's dtype."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("eta", [0.0, 1.0])
    @pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
    def test_low_precision(self, context: Context, dtype: torch.dtype, prediction_type: str, eta: float) -> None:
        """Test a low-precision head samples in its dtype and stays finite near t = 0, where 1 - alpha_bar is tiny."""
        head = ToyDiffusionHead(num_inference_steps=NUM_TRAIN_TIMESTEPS, prediction_type=prediction_type, eta=eta)
        head = head.to(dtype)
        assert head.sqrt_alpha_bar.dtype == head.sqrt_one_minus_alpha_bar.dtype == dtype
        assert head.sqrt_one_minus_alpha_bar[1] > 0

        actions = head.sample({"tokens": context["tokens"].to(dtype)})
        assert actions.dtype == dtype
        assert torch.isfinite(actions).all()


GRAPH_DEVICES = [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    pytest.param("xpu", marks=pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")),
]


class TestGraphReplay:
    """enable_graph_replay replays sampling at inference only."""

    @pytest.mark.parametrize("device", GRAPH_DEVICES)
    def test_replay(self, device: str) -> None:
        """Test graph replay matches eager sampling, reuses graphs and only runs at inference."""
        head = ToyDiffusionHead(num_inference_steps=10, eta=0.0, use_random_input_noise=False).to(device).eval()
        head.enable_graph_replay()
        contexts = [{"tokens": torch.randn(2, 6, CONTEXT_DIM, device=device)} for _ in range(2)]
        with torch.inference_mode():
            for context in contexts:
                torch.testing.assert_close(head.sample(context), head._sample(context, None, 10))  # noqa: SLF001
            assert head._graphs is not None  # noqa: SLF001
            assert len(head._graphs) == 1  # noqa: SLF001
            head.sample(contexts[0], num_steps=5)
            assert len(head._graphs) == 2  # noqa: SLF001

        head.train()
        with torch.inference_mode():
            head.sample(contexts[0], num_steps=3)
        assert len(head._graphs) == 2  # noqa: SLF001

        head.double()
        assert head._graphs == {}  # noqa: SLF001

    @pytest.mark.parametrize("device", GRAPH_DEVICES)
    def test_ddpm_draws_new_noise(self, device: str) -> None:
        """Test a replayed DDPM graph draws fresh noise on every call."""
        head = ToyDiffusionHead(num_inference_steps=10).to(device).eval()
        head.enable_graph_replay()
        context = {"tokens": torch.randn(2, 6, CONTEXT_DIM, device=device)}
        with torch.inference_mode():
            assert not torch.allclose(head.sample(context), head.sample(context))

    def test_falls_back_on_cpu(self, context: Context) -> None:
        """Test enable_graph_replay leaves CPU sampling eager and unchanged."""
        head = ToyDiffusionHead(num_inference_steps=10, eta=0.0, use_random_input_noise=False).eval()
        expected = head.sample(context)
        head.enable_graph_replay()
        with torch.inference_mode():
            torch.testing.assert_close(head.sample(context), expected)
        assert head._graphs == {}  # noqa: SLF001
