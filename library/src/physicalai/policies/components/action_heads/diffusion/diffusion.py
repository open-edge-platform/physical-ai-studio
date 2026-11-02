# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Denoising diffusion (DDPM / DDIM) action head.

The forward process adds Gaussian noise to a clean action chunk ``x_0``::

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,    eps ~ N(0, I)

where ``alpha_bar_t`` falls from ~1 at ``t = 0`` to ~0 at ``t = T - 1``. The
network learns to predict ``eps`` (or ``x_0``) from ``x_t``. Sampling walks
back from pure noise with the DDIM update::

    x_0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)
    x_prev  = sqrt(alpha_bar_prev) * x_0_hat
              + sqrt(1 - alpha_bar_prev - sigma^2) * eps_hat
              + sigma * z

    sigma^2 = eta^2 * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)

``eta = 1`` is DDPM ancestral sampling and ``eta = 0`` is deterministic DDIM.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from itertools import pairwise
from typing import Literal

import torch

from physicalai.policies.components.action_heads.base import Context, IterativeActionHead

# Device type -> (torch backend module, graph class name) for ``enable_graph_replay``.
_GRAPH_BACKENDS = {"cuda": (torch.cuda, "CUDAGraph"), "xpu": (torch.xpu, "XPUGraph")}

type BetaSchedule = Literal["linear", "scaled_linear", "squaredcos_cap_v2"]
type PredictionType = Literal["epsilon", "sample"]


def make_betas(
    schedule: BetaSchedule,
    num_train_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Build a noise schedule.

    Args:
        schedule: ``"linear"``, ``"scaled_linear"`` or ``"squaredcos_cap_v2"`` (cosine).
        num_train_timesteps: Number of forward diffusion steps (T).
        beta_start: First beta of the linear schedules.
        beta_end: Last beta of the linear schedules.

    Returns:
        Betas of shape (T,) in float32.

    Raises:
        ValueError: If ``schedule`` is unknown.
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    if schedule == "scaled_linear":
        return torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
    if schedule == "squaredcos_cap_v2":
        # Nichol & Dhariwal (2021): alpha_bar(s) = cos^2((s + 0.008) / 1.008 * pi / 2), beta capped at 0.999.
        def alpha_bar(s: float) -> float:
            return math.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2

        steps = [i / num_train_timesteps for i in range(num_train_timesteps + 1)]
        betas = [min(1 - alpha_bar(s_next) / alpha_bar(s), 0.999) for s, s_next in pairwise(steps)]
        return torch.tensor(betas, dtype=torch.float32)
    msg = f"Unknown beta_schedule {schedule!r}; expected 'linear', 'scaled_linear' or 'squaredcos_cap_v2'."
    raise ValueError(msg)


class DiffusionActionHead(IterativeActionHead):
    """Iterative action head trained with denoising diffusion.

    Implements the noise schedule, the training loss and the DDPM / DDIM
    update. Subclasses only implement ``denoise``: the network that maps a
    noisy chunk, integer timesteps of shape (B,) in ``[0, T)`` and the
    context to a noise (``"epsilon"``) or clean-sample (``"sample"``)
    prediction.

    Timesteps are integers from ``T - 1`` (noise) down to ``0``, followed by
    ``-1`` which stands for the clean data (``alpha_bar = 1``).

    Args:
        chunk_size: Number of actions predicted per chunk.
        action_dim: Dimension of each action.
        num_train_timesteps: Number of forward diffusion steps (T).
        num_inference_steps: Default number of sampling steps. Defaults to
            ``num_train_timesteps``.
        beta_schedule: Noise schedule, see ``make_betas``.
        beta_start: First beta of the linear schedules.
        beta_end: Last beta of the linear schedules.
        prediction_type: What ``denoise`` predicts: ``"epsilon"`` or ``"sample"``.
        clip_sample: Clamp the predicted clean sample to ``[-clip_sample_range, clip_sample_range]``.
        clip_sample_range: Clamp bound, matching the action normalization range.
        eta: Sampling noise scale. ``1.0`` is DDPM, ``0.0`` is deterministic DDIM.
        use_random_input_noise: Start sampling from Gaussian noise. If False,
            start from zeros. With ``eta = 0`` sampling is then fully
            deterministic and has no random ops, e.g. for OpenVINO export.

    The schedule buffers follow the module's dtype, so ``step`` runs in the
    dtype of ``x_t``. They store ``sqrt(alpha_bar)`` and ``sqrt(1 - alpha_bar)``
    directly, because ``1 - alpha_bar`` near ``t = 0`` rounds to 0 in bf16.

    Raises:
        ValueError: If an argument is out of range.
    """

    def __init__(
        self,
        chunk_size: int,
        action_dim: int,
        *,
        num_train_timesteps: int = 100,
        num_inference_steps: int | None = None,
        beta_schedule: BetaSchedule = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: PredictionType = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        eta: float = 1.0,
        use_random_input_noise: bool = True,
    ) -> None:
        """Initialize the diffusion action head.

        Raises:
            ValueError: If an argument is out of range.
        """
        super().__init__(chunk_size, action_dim, num_inference_steps or num_train_timesteps)
        if prediction_type not in {"epsilon", "sample"}:
            msg = f"prediction_type must be 'epsilon' or 'sample', got {prediction_type!r}."
            raise ValueError(msg)
        if eta < 0:
            msg = f"eta must be non-negative, got {eta}."
            raise ValueError(msg)
        self._check_num_steps(self.num_inference_steps, num_train_timesteps)

        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.eta = eta
        self.use_random_input_noise = use_random_input_noise
        self._graphs: dict[tuple, tuple] | None = None  # set by ``enable_graph_replay``

        alpha_bar = torch.cumprod(1 - make_betas(beta_schedule, num_train_timesteps, beta_start, beta_end), dim=0)
        # Prepend alpha_bar = 1 for the clean data, so timestep t is stored at index t + 1 and t = -1 at index 0.
        alpha_bar = torch.cat([alpha_bar.new_ones(1), alpha_bar])
        # Store both square roots: 1 - alpha_bar is tiny near t = 0 and would round to 0 if derived in low precision.
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt(), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt(), persistent=False)

    def _apply(self, fn, recurse=True):  # noqa: ANN001, ANN202, FBT002
        # Moving or casting replaces parameters, so captured graphs would read stale memory.
        if self._graphs:
            self._graphs = {}
        return super()._apply(fn, recurse)

    def _schedule_at(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # index_select keeps the lookup on the device; indexing with a 0-d tensor calls .item(),
        # which syncs and breaks CUDA / XPU graph capture.
        index = (t + 1).reshape(1)
        return self.sqrt_alpha_bar.index_select(0, index), self.sqrt_one_minus_alpha_bar.index_select(0, index)

    @staticmethod
    def _check_num_steps(num_steps: int, num_train_timesteps: int) -> None:
        if not 0 < num_steps <= num_train_timesteps:
            msg = f"num_steps must be in [1, {num_train_timesteps}], got {num_steps}."
            raise ValueError(msg)

    @abstractmethod
    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, context: Context) -> torch.Tensor:
        """Predict the noise or the clean sample.

        Args:
            x_t: Noisy actions of shape (B, chunk_size, action_dim).
            t: Integer timesteps of shape (B,) in ``[0, num_train_timesteps)``.
            context: Prepared conditioning tensors.

        Returns:
            ``eps_hat`` or ``x_0_hat`` (see ``prediction_type``), shape (B, chunk_size, action_dim).
        """

    def add_noise(self, actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample the forward process ``q(x_t | x_0)``.

        Args:
            actions: Clean actions ``x_0`` of shape (B, chunk_size, action_dim).
            noise: Gaussian noise of the same shape.
            t: Integer timesteps of shape (B,).

        Returns:
            ``sqrt(alpha_bar_t) * actions + sqrt(1 - alpha_bar_t) * noise``.
        """
        sqrt_ab = self.sqrt_alpha_bar[t + 1][:, None, None]
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bar[t + 1][:, None, None]
        return sqrt_ab * actions + sqrt_1m_ab * noise

    def compute_loss(self, actions: torch.Tensor, context: Context) -> torch.Tensor:
        """Noise the actions at a random timestep and regress the prediction target.

        Args:
            actions: Target actions of shape (B, chunk_size, action_dim).
            context: Conditioning tensors.

        Returns:
            Unreduced squared error of shape (B, chunk_size, action_dim).
        """
        noise = torch.randn_like(actions)
        t = torch.randint(0, self.num_train_timesteps, (actions.shape[0],), device=actions.device)
        prediction = self.denoise(self.add_noise(actions, noise, t), t, self.prepare_context(context))
        target = noise if self.prediction_type == "epsilon" else actions
        return (prediction - target) ** 2

    def timesteps(self, num_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:  # noqa: ARG002
        """Return evenly spaced integer timesteps, e.g. ``[90, 80, ..., 0, -1]`` for T = 100 and 10 steps.

        Uses the ``"leading"`` spacing of ``diffusers``. The schedule is always
        ``torch.long`` because it indexes the noise schedule; ``dtype`` is ignored.

        Args:
            num_steps: Number of sampling steps, at most ``num_train_timesteps``.
            device: Device of the returned tensor.
            dtype: Ignored.

        Returns:
            Timesteps of shape (num_steps + 1,).
        """
        self._check_num_steps(num_steps, self.num_train_timesteps)
        stride = self.num_train_timesteps // num_steps
        steps = torch.arange(num_steps - 1, -1, -1, device=device) * stride
        return torch.cat([steps, steps.new_full((1,), -1)])

    def sample(
        self,
        context: Context,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate an action chunk, starting from zeros if ``use_random_input_noise`` is False.

        Replays a CUDA or XPU graph instead when ``enable_graph_replay`` was
        called, the head is in ``eval()`` mode and gradients are disabled.

        Args:
            context: Conditioning tensors.
            noise: Optional initial sample of shape (B, chunk_size, action_dim).
            num_steps: Number of sampling steps. Defaults to ``num_inference_steps``.

        Returns:
            Actions of shape (B, chunk_size, action_dim).
        """
        num_steps = num_steps or self.num_inference_steps
        graphs = self._graphs
        if graphs is not None and not self.training and not torch.is_grad_enabled():
            backend = _GRAPH_BACKENDS.get(next(iter(context.values())).device.type)
            if backend is not None:
                return self._replay(graphs, backend, context, noise, num_steps)
        return self._sample(context, noise, num_steps)

    def _sample(self, context: Context, noise: torch.Tensor | None, num_steps: int) -> torch.Tensor:
        if noise is None and not self.use_random_input_noise:
            reference = next(iter(context.values()))
            noise = reference.new_zeros(reference.shape[0], self.chunk_size, self.action_dim)
        return super().sample(context, noise=noise, num_steps=num_steps)

    def enable_graph_replay(self, enabled: bool = True) -> None:  # noqa: FBT001, FBT002
        """Capture ``sample`` in a CUDA or XPU graph on first use and replay it afterwards.

        The backend follows the device of the context. Replay only runs in
        ``eval()`` mode with gradients disabled (``torch.inference_mode()`` or
        ``torch.no_grad()``); otherwise, and on other devices, ``sample`` runs
        eagerly. One graph is captured per context shape and dtype, number of
        steps and whether ``noise`` is given. Graphs are dropped when the module
        is moved or cast; call this method again after replacing parameters
        (for example ``load_state_dict(..., assign=True)``).

        Args:
            enabled: Whether to replay graphs. ``False`` drops captured graphs.
        """
        self._graphs = {} if enabled else None

    def _replay(
        self,
        graphs: dict[tuple, tuple],
        backend: tuple,
        context: Context,
        noise: torch.Tensor | None,
        num_steps: int,
    ) -> torch.Tensor:
        key = (num_steps, noise is None, *((name, v.shape, v.dtype, v.device) for name, v in context.items()))
        if key not in graphs:
            graphs[key] = self._capture(backend, context, noise, num_steps)
        graph, static_context, static_noise, static_actions = graphs[key]
        # Static buffers are normal tensors, so update them outside inference mode.
        with torch.inference_mode(mode=False), torch.no_grad():
            for name, value in context.items():
                static_context[name].copy_(value)
            if noise is not None:
                static_noise.copy_(noise)
            graph.replay()
            return static_actions.clone()

    def _capture(self, backend: tuple, context: Context, noise: torch.Tensor | None, num_steps: int) -> tuple:
        api, graph_type = backend
        device = next(iter(context.values())).device
        with torch.inference_mode(mode=False), torch.no_grad(), api.device(device):
            static_context = {name: value.clone() for name, value in context.items()}
            static_noise = None if noise is None else noise.clone()
            # Warm up on a side stream so lazy allocations happen outside the graph.
            stream = api.Stream()
            stream.wait_stream(api.current_stream())
            with api.stream(stream):
                for _ in range(3):
                    self._sample(static_context, static_noise, num_steps)
            api.current_stream().wait_stream(stream)
            graph = getattr(api, graph_type)()
            with api.graph(graph):
                static_actions = self._sample(static_context, static_noise, num_steps)
        return graph, static_context, static_noise, static_actions

    def step(
        self,
        x_t: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one DDIM update from ``t`` to ``t_next`` (DDPM when ``eta = 1``).

        Args:
            x_t: Current sample of shape (B, chunk_size, action_dim).
            model_output: Output of ``denoise`` at ``t``.
            t: Current integer timestep (scalar tensor).
            t_next: Next integer timestep (scalar tensor), ``-1`` for the clean data.

        Returns:
            The sample at ``t_next``.
        """
        sqrt_ab, sqrt_1m_ab = self._schedule_at(t)
        sqrt_ab_next, sqrt_1m_ab_next = self._schedule_at(t_next)

        x0 = (x_t - sqrt_1m_ab * model_output) / sqrt_ab if self.prediction_type == "epsilon" else model_output
        if self.clip_sample:
            x0 = x0.clamp(-self.clip_sample_range, self.clip_sample_range)
        # Noise consistent with x_t and the (clipped) x0, so eta = 1 reproduces the DDPM posterior exactly.
        eps = (x_t - sqrt_ab * x0) / sqrt_1m_ab

        # sigma^2 = eta^2 * (1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next)
        variance = (sqrt_1m_ab_next / sqrt_1m_ab) ** 2 * (1 - (sqrt_ab / sqrt_ab_next) ** 2)
        sigma = self.eta * variance.clamp(min=0).sqrt()
        direction = (sqrt_1m_ab_next**2 - sigma**2).clamp(min=0).sqrt()
        x_next = sqrt_ab_next * x0 + direction * eps
        if self.eta > 0:
            x_next += sigma * torch.randn_like(x_t)
        return x_next
