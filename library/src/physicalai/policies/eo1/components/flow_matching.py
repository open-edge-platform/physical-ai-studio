# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""Flow-matching primitives used by the EO-1 action head.

Ported from LeRobot's ``lerobot.policies.common.flow_matching`` and
``lerobot.policies.common.vla_utils``, which are themselves exact copies of the openpi originals.
Every function here is stateless and parameter-free, so carrying a local copy has no effect on
checkpoint compatibility.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

_TIME_NDIM_CHOICES = (1, 2)


def safe_float64(device: torch.device) -> torch.dtype:
    """Pick the widest float dtype the device supports.

    Args:
        device: Target device.

    Returns:
        ``torch.float32`` on MPS, which has no float64 kernels, otherwise ``torch.float64``.
    """
    return torch.float32 if device.type == "mps" else torch.float64


def sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> Tensor:
    """Draw ``bsize`` samples from ``Beta(alpha, beta)``.

    Args:
        alpha: Alpha parameter of the Beta distribution.
        beta: Beta parameter of the Beta distribution.
        bsize: Number of samples.
        device: Device the samples are moved to.

    Returns:
        Tensor of shape ``(bsize,)``.
    """
    # Beta sampling goes through _sample_dirichlet, which has no MPS kernel; sample on CPU.
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def sample_noise(shape: tuple[int, ...], device: torch.device) -> Tensor:
    """Draw the standard-normal flow-matching sample ``x_1``.

    Args:
        shape: Shape of the noise tensor.
        device: Device to allocate on.

    Returns:
        Float32 noise tensor of the requested shape.
    """
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)


def sample_time_beta(
    bsize: int,
    device: torch.device,
    *,
    alpha: float,
    beta: float,
    scale: float,
    offset: float,
) -> Tensor:
    """Draw flow-matching timesteps as ``Beta(alpha, beta) * scale + offset``.

    Args:
        bsize: Number of timesteps.
        device: Device to allocate on.
        alpha: Alpha parameter of the Beta distribution.
        beta: Beta parameter of the Beta distribution.
        scale: Multiplier applied to the sample.
        offset: Offset added to the scaled sample.

    Returns:
        Float32 tensor of shape ``(bsize,)``.
    """
    time_beta = sample_beta(alpha, beta, bsize, device)
    time = time_beta * scale + offset
    return time.to(dtype=torch.float32, device=device)


def create_sinusoidal_pos_embedding(
    time: Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    """Compute sine-cosine embeddings of scalar flow-matching timesteps.

    Args:
        time: Timesteps of shape ``(batch_size,)`` or ``(batch_size, action_horizon)``.
        dimension: Embedding width, which must be even.
        min_period: Shortest period in the geometric period schedule.
        max_period: Longest period in that schedule.
        device: Device to allocate the schedule on.

    Returns:
        Embedding of shape ``(*time.shape, dimension)``.

    Raises:
        ValueError: If `dimension` is odd or `time` has an unsupported rank.
    """
    if dimension % 2 != 0:
        msg = f"dimension ({dimension}) must be divisible by 2"
        raise ValueError(msg)
    if time.ndim not in _TIME_NDIM_CHOICES:
        msg = "The time tensor must have shape (batch_size,) or (batch_size, action_horizon)."
        raise ValueError(msg)

    dtype = safe_float64(device)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = time[..., None] * scaling_factor
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Zero-pad the last dimension of a vector up to `new_dim`.

    Vectors already at least `new_dim` wide are returned untouched, matching openpi.

    Args:
        vector: Tensor of shape ``(..., dim)``.
        new_dim: Target width of the last dimension.

    Returns:
        The padded tensor.
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def euler_integrate(denoise_fn: Callable[[Tensor, Tensor], Tensor], noise: Tensor, num_steps: int) -> Tensor:
    """Integrate a velocity field from ``t=1`` (noise) to ``t=0`` (actions) with forward Euler.

    This is the openpi sampling loop: ``dt = -1/num_steps``, ``time = 1.0 + step * dt`` and
    ``x_t <- x_t + dt * v_t``. LeRobot's real-time-chunking hooks are not ported, since EO-1 does
    not use them.

    Args:
        denoise_fn: Callable computing the velocity ``v_t`` from ``(x_t, time_tensor)``, where
            `time_tensor` is a float32 tensor of shape ``(batch_size,)``. The returned velocity must
            match `x_t` in shape and dtype.
        noise: Initial sample ``x_1`` of shape ``(batch_size, ...)``.
        num_steps: Number of Euler steps.

    Returns:
        The integrated sample ``x_0``.
    """
    bsize = noise.shape[0]
    device = noise.device

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
        v_t = denoise_fn(x_t, time_tensor)
        # Deliberately out of place: `+=` would mutate the caller's noise tensor in place.
        x_t = x_t + dt * v_t  # noqa: PLR6104
    return x_t
