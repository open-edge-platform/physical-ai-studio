# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Flow-matching objective and sampler for the XR1 policy.

XR1 learns a velocity field that transports Gaussian noise to the action chunk.
Training interpolates between noise and the target action at a sampled timestep and
regresses the velocity; inference integrates the field with a fixed number of Euler
steps.

Two terms make up the loss, following the reference implementation:

* a weighted MSE on the velocity, where the weight is derived from the error of a
  full rollout when an action prefix is present (hard samples count for more);
* a frequency-domain term on the real FFT of the predicted chunk, which penalizes
  jerky trajectories. Dimensions listed in ``freq_excluded_dims`` — gripper-like
  channels that are meant to switch abruptly — are left out of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Callable

WEIGHT_MIN = 0.5
WEIGHT_MAX = 5.0


class XR1FlowModel(nn.Module):
    """Flow-matching schedule, sampler and loss.

    Holds no parameters; it is an :class:`~torch.nn.Module` only so it can live in
    the model tree and follow ``.to(...)`` calls.
    """

    def __init__(
        self,
        num_inference_steps: int = 5,
        flow_sampling: Literal["beta", "logit_normal", "uniform"] = "beta",
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        *,
        enable_freq: bool = True,
        freq_coefficient: float = 1.0,
        freq_excluded_dims: tuple[int, ...] = (17, 18, 19),
    ) -> None:
        """Initialize the flow model.

        Args:
            num_inference_steps: Number of Euler steps at inference.
            flow_sampling: Distribution used to sample training timesteps.
            beta_alpha: Alpha of the Beta prior, used when ``flow_sampling`` is
                ``"beta"``.
            beta_beta: Beta of the Beta prior.
            enable_freq: Whether to compute the frequency-domain loss term.
            freq_coefficient: Weight of the frequency term in the total loss.
            freq_excluded_dims: Action dimensions excluded from the frequency term.
        """
        super().__init__()
        self.num_inference_steps = num_inference_steps
        self.flow_sampling = flow_sampling
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.enable_freq = enable_freq
        self.freq_coefficient = freq_coefficient
        self.freq_excluded_dims = tuple(freq_excluded_dims)
        self._beta = Beta(
            torch.tensor(float(beta_alpha)),
            torch.tensor(float(beta_beta)),
        )

    def sample_timestep(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample training timesteps in ``[0, 1)``.

        Args:
            batch_size: Number of timesteps to draw.
            device: Device for the result.
            dtype: Dtype for the result.

        Returns:
            Tensor of shape ``(batch_size, 1, 1)`` for broadcasting over an action
            chunk.

        Raises:
            ValueError: If ``flow_sampling`` is not a supported distribution.
        """
        if self.flow_sampling == "beta":
            # The reference draws (1 - Beta(1.5, 1.0)), whose mean is 0.4, so
            # timesteps concentrate toward the noisy end of the path where the
            # velocity is hardest to predict. The 0.999 scale keeps t < 1.
            samples = (1 - self._beta.sample((batch_size,)).to(device)) * 0.999
        elif self.flow_sampling == "uniform":
            samples = torch.rand(batch_size, device=device) * 0.999
        elif self.flow_sampling == "logit_normal":
            samples = torch.sigmoid(torch.randn(batch_size, device=device))
        else:
            msg = f"Unsupported flow_sampling: {self.flow_sampling}"
            raise ValueError(msg)
        return samples.to(dtype)[:, None, None]

    @staticmethod
    def interpolate(noise: torch.Tensor, action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Interpolate between noise and the target action.

        Args:
            noise: Gaussian sample shaped like ``action``.
            action: Target action chunk.
            timestep: Broadcastable timestep in ``[0, 1]``; ``1`` is the action.

        Returns:
            The interpolated (noisy) action.
        """
        return (1 - timestep) * noise + timestep * action

    @staticmethod
    def velocity_target(noise: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return the velocity that transports noise to the action.

        Args:
            noise: Gaussian sample shaped like ``action``.
            action: Target action chunk.

        Returns:
            The target velocity field.
        """
        return action - noise

    @torch.no_grad()
    def generate(
        self,
        noise: torch.Tensor,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Integrate the velocity field with uniform Euler steps.

        Args:
            noise: Starting sample of shape ``(batch, horizon, action_dim)``.
            velocity_fn: Callable mapping ``(sample, timestep)`` to a velocity.

        Returns:
            The integrated action chunk, same shape as ``noise``.
        """
        sample = noise.clone()
        step_size = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            timestep = torch.full(
                (sample.shape[0], 1, 1),
                step * step_size,
                device=sample.device,
                dtype=sample.dtype,
            )
            sample += velocity_fn(sample, timestep) * step_size
        return sample

    def flow_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        action_mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the weighted velocity MSE and the frequency-domain term.

        Args:
            pred: Predicted velocity of shape ``(batch, horizon, action_dim)``.
            target: Target velocity of the same shape.
            action_mask: Float or boolean mask of the same shape marking supervised
                entries.
            weight: Optional per-entry weight; defaults to ones.

        Returns:
            ``(loss_mse, loss_freq)``. Both are differentiable zeros when the mask
            selects nothing, so an unlucky batch cannot break the step.
        """
        pred = pred.float()
        target = target.float()
        weight = torch.ones_like(pred) if weight is None else weight.float()
        mask = action_mask.bool()

        if not torch.any(mask):
            zero = (pred.sum() + target.sum()) * 0.0
            return zero, zero

        with torch.no_grad():
            weight = weight.clone()
            weight[mask] /= weight[mask].mean()
            weight.clamp_(WEIGHT_MIN, WEIGHT_MAX)

        loss_mse = (F.mse_loss(pred, target, reduction="none") * weight)[mask].mean()

        if not self.enable_freq:
            return loss_mse, loss_mse.new_zeros(())

        return loss_mse, self._frequency_loss(pred, target, mask, weight)

    def _frequency_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize spectral differences between predicted and target chunks.

        Args:
            pred: Predicted velocity.
            target: Target velocity.
            mask: Boolean supervision mask.
            weight: Per-entry weights used to scale each sample's contribution.

        Returns:
            Scalar frequency loss, or a differentiable zero when no sample has a
            fully valid final timestep.
        """
        spectrum = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()

        # Only score samples whose chunk runs to the end; a truncated chunk has a
        # meaningless spectrum.
        valid_batch = mask[:, -1].any(dim=1)
        if not torch.any(valid_batch):
            return spectrum.sum() * 0.0

        freq_mask = mask[valid_batch, : spectrum.shape[1]].clone()
        excluded = [dim for dim in self.freq_excluded_dims if dim < freq_mask.shape[-1]]
        if excluded:
            freq_mask[:, :, excluded] = False
        if not torch.any(freq_mask):
            return spectrum.sum() * 0.0

        freq_weight = weight.mean(dim=(1, 2)).view(-1, 1, 1)
        return (spectrum * freq_weight)[valid_batch][freq_mask].mean()
