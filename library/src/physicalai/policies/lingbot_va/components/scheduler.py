# Copyright 2024-2025 The Robbyant Team Authors.
# Copyright 2026 The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Flow-matching scheduler for the LingBot-VA dual-stream sampler.

LingBot-VA runs two independent instances of :class:`FlowMatchScheduler` at inference,
one for the video-latent stream and one for the action stream, each with its own
``shift`` (SNR shift) and number of denoising steps.
"""

from __future__ import annotations

import math

import torch


def sample_timestep_id(
    batch_size: int = 1,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Sample per-frame flow-matching timestep ids.

    Args:
        batch_size: Number of timestep ids to draw (one per latent frame).
        min_timestep_bd: Lower bound of the sampled uniform range, in ``[0, 1]``.
        max_timestep_bd: Upper bound of the sampled uniform range, in ``[0, 1]``.
        num_train_timesteps: Size of the discrete training timestep grid.

    Returns:
        Integer tensor of shape ``[batch_size]`` with values in
        ``[0, num_train_timesteps - 1]``.
    """
    u = torch.rand(size=[batch_size]) * (max_timestep_bd - min_timestep_bd) + min_timestep_bd
    return (u * num_train_timesteps).clamp(min=0, max=num_train_timesteps - 1).to(torch.int64)


class FlowMatchScheduler:
    """Rectified-flow (flow-matching) noise scheduler.

    The sigma grid is a shifted linear interpolation between ``sigma_max`` and
    ``sigma_min``; ``shift`` moves the schedule towards the noisy end (larger values
    spend more steps at high noise). Timesteps are ``sigma * num_train_timesteps``.

    Args:
        num_inference_steps: Number of denoising steps for the initial grid.
        num_train_timesteps: Size of the discrete training timestep grid.
        shift: SNR shift applied to the sigma grid.
        sigma_max: Largest sigma (pure noise).
        sigma_min: Smallest sigma (clean sample).
        inverse_timesteps: Reverse the sigma order (noise-prediction convention).
        extra_one_step: Build ``num_inference_steps + 1`` sigmas and drop the last,
            so the grid never reaches ``sigma_min`` exactly.
        reverse_sigmas: Use ``1 - sigma`` instead of ``sigma``.
        exponential_shift: Apply an exponential (``mu``-based) shift instead of the
            linear ``shift``.
        exponential_shift_mu: Fixed ``mu`` for the exponential shift.
        shift_terminal: Rescale the schedule so the final sigma equals this value.
    """

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        *,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
        exponential_shift: bool = False,
        exponential_shift_mu: float | None = None,
        shift_terminal: float | None = None,
    ) -> None:
        """Initialize the scheduler and build the initial sigma / timestep grid."""
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.training = False
        self.linear_timesteps_weights: torch.Tensor | None = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        *,
        training: bool = False,
        shift: float | None = None,
        dynamic_shift_len: int | None = None,
    ) -> None:
        """Rebuild the sigma / timestep grid for a given number of steps.

        Args:
            num_inference_steps: Number of denoising steps.
            denoising_strength: Fraction of the noise range to traverse (1.0 = full).
            training: Also compute the per-timestep loss weights used during training.
            shift: Override the instance ``shift``.
            dynamic_shift_len: Sequence length used to derive ``mu`` when
                ``exponential_shift`` is enabled.

        Raises:
            ValueError: If ``exponential_shift`` is enabled without a usable ``mu``.
        """
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            if mu is None:
                msg = "exponential_shift needs either exponential_shift_mu or dynamic_shift_len."
                raise ValueError(msg)
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())
            self.training = True
        else:
            self.training = False

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        *,
        to_final: bool = False,
    ) -> torch.Tensor:
        """Take one Euler step along the probability-flow ODE.

        Args:
            model_output: Predicted velocity for ``sample`` at ``timestep``.
            timestep: Current timestep (scalar tensor or float).
            sample: Current (noisy) sample.
            to_final: Jump straight to the clean end of the schedule.

        Returns:
            The sample advanced to the next timestep of the grid.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_ - sigma)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        t_dim: int = 2,
    ) -> torch.Tensor:
        """Interpolate between a clean sample and noise at the given per-frame timesteps.

        Args:
            original_samples: Clean sample tensor.
            noise: Noise tensor with the same shape as ``original_samples``.
            timestep: Per-frame timesteps, broadcast along ``t_dim``.
            t_dim: Axis of ``original_samples`` that ``timestep`` indexes.

        Returns:
            The noised sample ``(1 - sigma) * x + sigma * noise``.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep = timestep[None]
        timestep_id = torch.argmin((self.timesteps[:, None] - timestep).abs(), dim=0)
        shape = [1] * noise.ndim
        shape[t_dim] = timestep_id.shape[0]
        sigma = self.sigmas[timestep_id].to(original_samples).view(shape)
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Return the flow-matching regression target ``noise - sample``.

        Args:
            sample: Clean sample tensor.
            noise: Noise tensor.
            timestep: Unused; kept for signature parity with diffusers schedulers.

        Returns:
            The velocity target.
        """
        del timestep
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """Look up the per-timestep training loss weights.

        Args:
            timestep: Timestep values to look up.

        Returns:
            Weight tensor with the same shape as ``timestep``.

        Raises:
            RuntimeError: If ``set_timesteps(..., training=True)`` was never called.
        """
        if self.linear_timesteps_weights is None:
            msg = "training_weight() requires set_timesteps(..., training=True) first."
            raise RuntimeError(msg)
        timestep_id = torch.argmin((self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0)
        return self.linear_timesteps_weights.to(timestep.device)[timestep_id].to(timestep.device)

    @staticmethod
    def calculate_shift(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ) -> float:
        """Interpolate the exponential-shift ``mu`` from the sequence length.

        Args:
            image_seq_len: Token count of the sample being denoised.
            base_seq_len: Sequence length mapped to ``base_shift``.
            max_seq_len: Sequence length mapped to ``max_shift``.
            base_shift: Shift at ``base_seq_len``.
            max_shift: Shift at ``max_seq_len``.

        Returns:
            The interpolated ``mu``.
        """
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b
