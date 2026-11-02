# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base classes for policy action heads.

An action head turns a conditioning context (e.g. image or VLM tokens) into a
chunk of actions of shape (B, chunk_size, action_dim).

- ``ActionHead`` is the minimal interface: a training loss and a sampler.
- ``IterativeActionHead`` adds the shared denoising loop used by flow matching
  and diffusion heads. Subclasses only define the network call, the time
  schedule and the update rule.

Both classes register no parameters, buffers or submodules. A module's
``state_dict`` keys therefore depend only on its own attributes, so existing
policies can adopt these base classes without breaking pretrained checkpoints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

type Context = dict[str, torch.Tensor]
"""Conditioning tensors for an action head, e.g. ``{"tokens": (B, S, D), "mask": (B, S)}``."""


class ActionHead(nn.Module, ABC):
    """Interface for modules that map a conditioning context to an action chunk.

    Subclasses implement ``compute_loss`` for training and ``sample`` for
    inference. ``forward`` is an alias for ``sample``.

    Args:
        chunk_size: Number of actions predicted per chunk (T).
        action_dim: Dimension of each action (A).

    Raises:
        ValueError: If ``chunk_size`` or ``action_dim`` is not positive.

    Examples:
        >>> class LinearHead(ActionHead):
        ...     def __init__(self, context_dim: int, chunk_size: int, action_dim: int) -> None:
        ...         super().__init__(chunk_size, action_dim)
        ...         self.proj = nn.Linear(context_dim, chunk_size * action_dim)
        ...
        ...     def sample(self, context, *, noise=None, num_steps=None):
        ...         pooled = context["tokens"].mean(dim=1)
        ...         return self.proj(pooled).view(-1, self.chunk_size, self.action_dim)
        ...
        ...     def compute_loss(self, actions, context):
        ...         return (self.sample(context) - actions).abs()
    """

    def __init__(self, chunk_size: int, action_dim: int) -> None:
        """Initialize the action head.

        Raises:
            ValueError: If ``chunk_size`` or ``action_dim`` is not positive.
        """
        super().__init__()
        if chunk_size <= 0 or action_dim <= 0:
            msg = f"chunk_size and action_dim must be positive, got {chunk_size} and {action_dim}."
            raise ValueError(msg)
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    @abstractmethod
    def compute_loss(self, actions: torch.Tensor, context: Context) -> torch.Tensor:
        """Compute the training loss for a batch of target actions.

        Args:
            actions: Target actions of shape (B, chunk_size, action_dim).
            context: Conditioning tensors.

        Returns:
            Unreduced per-element loss of shape (B, chunk_size, action_dim).
            Reduce it with ``physicalai.policies.utils.loss.reduce_losses`` to
            ignore padded action steps.
        """

    @abstractmethod
    def sample(
        self,
        context: Context,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate an action chunk.

        Args:
            context: Conditioning tensors.
            noise: Optional initial sample of shape (B, chunk_size, action_dim).
                Ignored by deterministic heads.
            num_steps: Optional number of sampling steps. Ignored by one-shot heads.

        Returns:
            Actions of shape (B, chunk_size, action_dim).
        """

    def forward(
        self,
        context: Context,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate an action chunk. Same as ``sample``; use ``compute_loss`` for training.

        Args:
            context: Conditioning tensors.
            noise: Optional initial sample of shape (B, chunk_size, action_dim).
            num_steps: Optional number of sampling steps.

        Returns:
            Actions of shape (B, chunk_size, action_dim).
        """
        return self.sample(context, noise=noise, num_steps=num_steps)


class IterativeActionHead(ActionHead):
    """Action head that refines noise into actions over several steps.

    Sampling runs the loop::

        x = noise
        for t, t_next in zip(timesteps[:-1], timesteps[1:]):
            x = step(x, denoise(x, t, context), t, t_next)

    Subclasses implement ``denoise`` (the network call), ``timesteps`` (the
    schedule) and ``step`` (the update rule, e.g. Euler or DDIM). They may also
    override ``prepare_context`` to precompute work that does not change
    between steps, such as cross-attention keys and values.

    ``integrate`` only uses tensor operations on fixed shapes, so the whole
    loop can be captured in a CUDA or XPU graph.

    Args:
        chunk_size: Number of actions predicted per chunk (T).
        action_dim: Dimension of each action (A).
        num_inference_steps: Default number of sampling steps.

    Raises:
        ValueError: If ``num_inference_steps`` is not positive.
    """

    def __init__(self, chunk_size: int, action_dim: int, num_inference_steps: int) -> None:
        """Initialize the iterative action head.

        Raises:
            ValueError: If ``num_inference_steps`` is not positive.
        """
        super().__init__(chunk_size, action_dim)
        if num_inference_steps <= 0:
            msg = f"num_inference_steps must be positive, got {num_inference_steps}."
            raise ValueError(msg)
        self.num_inference_steps = num_inference_steps

    @abstractmethod
    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, context: Context) -> torch.Tensor:
        """Run the network on a noisy action chunk.

        Args:
            x_t: Noisy actions of shape (B, chunk_size, action_dim).
            t: Timesteps of shape (B,).
            context: Prepared conditioning tensors.

        Returns:
            The network output, e.g. a velocity or noise prediction, of shape
            (B, chunk_size, action_dim).
        """

    @abstractmethod
    def timesteps(self, num_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the sampling schedule.

        Args:
            num_steps: Number of sampling steps.
            device: Device of the returned tensor.
            dtype: Dtype of the returned tensor.

        Returns:
            Timesteps of shape (num_steps + 1,), from the noise end to the data end.
        """

    @abstractmethod
    def step(
        self,
        x_t: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """Move the sample from time ``t`` to ``t_next``.

        Args:
            x_t: Current sample of shape (B, chunk_size, action_dim).
            model_output: Output of ``denoise`` at time ``t``.
            t: Current timestep (scalar tensor).
            t_next: Next timestep (scalar tensor).

        Returns:
            The sample at ``t_next``.
        """

    def prepare_context(self, context: Context) -> Context:  # noqa: PLR6301
        """Precompute step-independent work on the context. Called once per ``sample``.

        Args:
            context: Conditioning tensors.

        Returns:
            The prepared context passed to ``denoise`` at every step.
        """
        return context

    def integrate(self, x: torch.Tensor, context: Context, timesteps: torch.Tensor) -> torch.Tensor:
        """Run the sampling loop from ``timesteps[0]`` to ``timesteps[-1]``.

        Args:
            x: Initial sample of shape (B, chunk_size, action_dim).
            context: Prepared conditioning tensors.
            timesteps: Schedule of shape (num_steps + 1,).

        Returns:
            The final sample of shape (B, chunk_size, action_dim).
        """
        batch_size = x.shape[0]
        for i in range(timesteps.shape[0] - 1):
            t, t_next = timesteps[i], timesteps[i + 1]
            model_output = self.denoise(x, t.expand(batch_size), context)
            x = self.step(x, model_output, t, t_next)
        return x

    def sample(
        self,
        context: Context,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate an action chunk by integrating from noise.

        Args:
            context: Conditioning tensors. The batch size, device and dtype are
                taken from its first tensor when ``noise`` is not given.
            noise: Optional initial sample of shape (B, chunk_size, action_dim).
                Defaults to standard Gaussian noise.
            num_steps: Number of sampling steps. Defaults to ``num_inference_steps``.

        Returns:
            Actions of shape (B, chunk_size, action_dim).
        """
        if noise is None:
            reference = next(iter(context.values()))
            noise = torch.randn(
                reference.shape[0],
                self.chunk_size,
                self.action_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
        timesteps = self.timesteps(num_steps or self.num_inference_steps, device=noise.device, dtype=noise.dtype)
        return self.integrate(noise, self.prepare_context(context), timesteps)
