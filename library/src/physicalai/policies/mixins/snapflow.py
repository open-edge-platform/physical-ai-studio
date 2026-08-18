# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared SnapFlow self-distillation surface for flow-matching policies.

SnapFlow ([arXiv:2604.05656](https://arxiv.org/abs/2604.05656)) compresses the
multi-step Euler denoising loop of a flow-matching VLA into a single forward
pass by self-distillation. It is trained in two phases: standard flow matching
first, then a short distillation phase with the VLM backbone frozen.

The target-time embedding and the ``nn.Module`` forward paths that are
``torch.compile``-wrapped and traced during export still live in each policy's
model. What this module owns is the surface that was otherwise duplicated
verbatim between :class:`~physicalai.policies.Pi05` and
:class:`~physicalai.policies.SmolVLA`:

- :class:`SnapFlowConfigMixin` — the four config flags and their validation.
- :class:`SnapFlowPolicyMixin` — the ``enable_snapflow()`` phase-2 entry point
  used by :class:`~physicalai.train.callbacks.SnapFlowPhaseCallback`.
- :class:`SnapFlowModelMixin` — the mixed FM/consistency-distillation training
  loss and the inference-time step-count/target-time selection, shared by each
  policy's flow-matching ``nn.Module``. A new flow matcher only needs to
  implement ``_predict_velocity`` (source-time, target-time conditioned
  velocity prediction) and a ``sample_noise`` callable to reuse it.

Example:
    >>> from physicalai.policies import Pi05
    >>> policy = Pi05(pretrained_name_or_path="lerobot/pi05_base")
    >>> policy.enable_snapflow(alpha=0.5, lambda_=0.1, num_inference_steps=1)  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

# Paper defaults (arXiv:2604.05656 §3.6 + Appendix J). Note that the config
# default for lambda is 1.0 for backwards compatibility with checkpoints trained
# before the two-phase recipe landed; the phase-2 entry point uses 0.1.
SNAPFLOW_DEFAULT_ALPHA = 0.5
SNAPFLOW_DEFAULT_LAMBDA = 0.1
SNAPFLOW_DEFAULT_NUM_INFERENCE_STEPS = 1


@dataclass(frozen=True)
class SnapFlowConfigMixin:
    """SnapFlow self-distillation flags shared by flow-matching policy configs.

    Mix into a policy config ahead of :class:`~physicalai.config.Config` and call
    :meth:`_validate_snapflow` from the config's ``__post_init__``.

    Attributes:
        snapflow_enabled: Enable SnapFlow self-distillation training mode for 1-NFE inference.
            When True, training mixes standard flow-matching with consistency objectives.
            See: arxiv.org/abs/2604.05656. Defaults to False.
        snapflow_alpha: Mixing ratio between FM and consistency objectives. ``alpha`` fraction of samples
            use standard flow-matching loss, ``1-alpha`` use the two-step Euler shortcut consistency loss.
            Must be in [0, 1]. Defaults to 0.5.
        snapflow_lambda: Weight for the consistency (shortcut) loss component. Balances gradient magnitudes
            between FM and consistency objectives. Defaults to 0.1.
        snapflow_num_inference_steps: Number of denoising steps at inference when SnapFlow is enabled.
            Set to 1 for single-step (1-NFE) generation. Defaults to 1.

    Example:
        >>> from dataclasses import dataclass
        >>> from physicalai.config import Config
        >>> from physicalai.policies.mixins import SnapFlowConfigMixin
        >>> @dataclass(frozen=True)
        ... class MyConfig(SnapFlowConfigMixin, Config):
        ...     hidden_dim: int = 256
        ...
        ...     def __post_init__(self) -> None:
        ...         self._validate_snapflow()
        >>> MyConfig().snapflow_enabled
        False
    """

    # SnapFlow self-distillation (arxiv.org/abs/2604.05656)
    snapflow_enabled: bool = False
    snapflow_alpha: float = SNAPFLOW_DEFAULT_ALPHA
    snapflow_lambda: float = SNAPFLOW_DEFAULT_LAMBDA
    snapflow_num_inference_steps: int = SNAPFLOW_DEFAULT_NUM_INFERENCE_STEPS

    def _validate_snapflow(self) -> None:
        """Validate the SnapFlow flags.

        Raises:
            ValueError: If ``snapflow_alpha`` falls outside ``[0, 1]`` or
                ``snapflow_num_inference_steps`` is below 1.
        """
        if not 0.0 <= self.snapflow_alpha <= 1.0:
            msg = f"snapflow_alpha must be in [0, 1], got {self.snapflow_alpha}"
            raise ValueError(msg)

        if self.snapflow_num_inference_steps < 1:
            msg = f"snapflow_num_inference_steps must be >= 1, got {self.snapflow_num_inference_steps}"
            raise ValueError(msg)


class SnapFlowPolicyMixin:
    """Give a flow-matching policy the SnapFlow phase-2 entry point.

    Implements :meth:`enable_snapflow`, which switches a policy that has been
    trained with standard flow matching into SnapFlow self-distillation and
    freezes the VLM backbone so only the action expert and the zero-initialised
    target-time embedding keep training (~10% of parameters).

    The mixin depends on two capabilities that are not SnapFlow-specific — they
    are ordinary parts of a VLA policy's API, and each policy family implements
    them differently because the VLM wrapper is named and frozen differently:

    - :attr:`inner_model` — the unwrapped flow-matching ``nn.Module``.
    - :meth:`freeze_vlm` — freeze the VLM backbone so only the action expert
      trains.

    Attributes:
        config: The policy's frozen config dataclass, which must mix in
            :class:`SnapFlowConfigMixin` and carry a ``train_expert_only`` flag.
        _set_hparam_keys: Policy hook that re-syncs checkpoint hparams from
            ``config``.

    Example:
        >>> class MyPolicy(SnapFlowPolicyMixin, Policy):  # doctest: +SKIP
        ...     @property
        ...     def inner_model(self):
        ...         return self.model
        ...
        ...     def freeze_vlm(self):
        ...         object.__setattr__(self.config, "train_expert_only", True)
        ...         self.model.vlm.train_expert_only = True
        ...         self.model.vlm.set_requires_grad()
        ...         self.model.train()
    """

    # Declared for type checkers only; provided by the host policy.
    config: Any
    _set_hparam_keys: Callable[[], None]

    @property
    def inner_model(self) -> nn.Module:
        """The unwrapped flow-matching module.

        Implementations return the module that owns the velocity field and the
        target-time embedding, and should raise ``RuntimeError`` when it has not
        been built yet.

        Raises:
            NotImplementedError: If the host policy does not implement the hook.
        """
        msg = f"{type(self).__name__} must implement the inner_model property."
        raise NotImplementedError(msg)

    def freeze_vlm(self) -> None:
        """Freeze the VLM backbone so only the action expert keeps training.

        Implementations set ``config.train_expert_only``, flip ``requires_grad``
        on the backbone, and re-apply train/eval modes.

        Raises:
            NotImplementedError: If the host policy does not implement the hook.
        """
        msg = f"{type(self).__name__} must implement freeze_vlm()."
        raise NotImplementedError(msg)

    def enable_snapflow(
        self,
        alpha: float = SNAPFLOW_DEFAULT_ALPHA,
        lambda_: float = SNAPFLOW_DEFAULT_LAMBDA,
        num_inference_steps: int = SNAPFLOW_DEFAULT_NUM_INFERENCE_STEPS,
    ) -> None:
        """Enable SnapFlow self-distillation and freeze the VLM backbone.

        Activates the SnapFlow mixed FM/consistency objective and freezes the VLM
        so only the action expert and target-time embedding are trained. This is
        the phase-2 entry point used by
        :class:`~physicalai.train.callbacks.SnapFlowPhaseCallback` and can also
        be called manually before ``trainer.fit()``.

        Warm-starting from a well-trained flow-matching checkpoint is a
        precondition: the shortcut target is bootstrapped from the model's own
        marginal-velocity predictions, so distilling an undertrained model
        distills noise.

        Args:
            alpha: Weight for the flow-matching loss branch (``L_FM``).
                Paper default: ``0.5``.
            lambda_: Scaling factor for the shortcut consistency loss
                (``L_shortcut``). Paper default: ``0.1``.
            num_inference_steps: Number of denoising steps at inference time.
                Set to ``1`` for the full single-step SnapFlow speedup.

        Raises:
            ValueError: If ``alpha`` falls outside ``[0, 1]`` or
                ``num_inference_steps`` is below 1.

        Note:
            :attr:`inner_model` raises ``RuntimeError`` when accessed before the
            model has been initialized (i.e. before ``setup()`` runs).
        """
        if not 0.0 <= alpha <= 1.0:
            msg = f"alpha must be in [0, 1], got {alpha}"
            raise ValueError(msg)
        if num_inference_steps < 1:
            msg = f"num_inference_steps must be >= 1, got {num_inference_steps}"
            raise ValueError(msg)

        inner = self.inner_model
        inner._snapflow_enabled = True  # type: ignore[assignment]  # noqa: SLF001
        inner._snapflow_alpha = alpha  # type: ignore[assignment]  # noqa: SLF001
        inner._snapflow_lambda = lambda_  # type: ignore[assignment]  # noqa: SLF001
        inner._snapflow_num_inference_steps = num_inference_steps  # type: ignore[assignment]  # noqa: SLF001

        # Config is a frozen dataclass — bypass the immutability check so the
        # updated flags are included in checkpoint hparams. train_expert_only is
        # set by freeze_vlm(), which owns that half of the state.
        for key, value in (
            ("snapflow_enabled", True),
            ("snapflow_alpha", alpha),
            ("snapflow_lambda", lambda_),
            ("snapflow_num_inference_steps", num_inference_steps),
        ):
            object.__setattr__(self.config, key, value)  # noqa: PLC2801

        self.freeze_vlm()

        # Keep top-level checkpoint hparams in sync with the mutated config so
        # checkpoints saved after this phase transition reload as SnapFlow
        # policies. hparams["config"] is a to_dict() snapshot, not a live view,
        # so it must be refreshed explicitly.
        self._set_hparam_keys()


class SnapFlowModelMixin:
    """Give a flow-matching ``nn.Module`` the SnapFlow training/inference math.

    Implements the pieces of SnapFlow that were duplicated verbatim between
    :class:`~physicalai.policies.smolvla.model.VLAFlowMatching` and
    :class:`~physicalai.policies.pi05.model.Pi05Model`:

    - :meth:`snapflow_mixed_loss` — the mixed flow-matching / consistency
      distillation training loss (self-distillation via a two-step Euler
      shortcut target).
    - :meth:`snapflow_num_inference_steps` and :meth:`snapflow_target_time` —
      the inference-time step-count and target-time overrides used by the
      1-NFE sampling loop.

    This mixin does not define ``__init__`` (mixing into an ``nn.Module``
    subclass makes constructor chaining brittle). Call
    :meth:`init_snapflow_state` explicitly at the end of the host model's own
    ``__init__``, mirroring how :class:`SnapFlowConfigMixin` expects
    ``_validate_snapflow()`` to be called from ``__post_init__``.

    A host model must additionally provide:

    - A ``_predict_velocity(x_t, timestep, target_time, *cond) -> Tensor``
      method (or equivalent callable passed to :meth:`snapflow_mixed_loss`)
      that predicts the velocity field conditioned on both the source
      timestep and the SnapFlow target time.

    Example:
        >>> class MyFlowMatching(SnapFlowModelMixin, nn.Module):  # doctest: +SKIP
        ...     def __init__(self, *, snapflow_enabled=False, snapflow_alpha=0.5,
        ...                  snapflow_lambda=1.0, snapflow_num_inference_steps=1):
        ...         super().__init__()
        ...         self.init_snapflow_state(
        ...             enabled=snapflow_enabled,
        ...             alpha=snapflow_alpha,
        ...             lambda_=snapflow_lambda,
        ...             num_inference_steps=snapflow_num_inference_steps,
        ...         )
    """

    # Declared for type checkers only; set by init_snapflow_state().
    _snapflow_enabled: bool
    _snapflow_alpha: float
    _snapflow_lambda: float
    _snapflow_num_inference_steps: int

    def init_snapflow_state(
        self,
        *,
        enabled: bool,
        alpha: float,
        lambda_: float,
        num_inference_steps: int,
    ) -> None:
        """Store the SnapFlow flags on the host model.

        Call once from the host model's ``__init__``, after ``super().__init__()``.
        Values are expected to already be validated (config-level validation is
        done by :meth:`SnapFlowConfigMixin._validate_snapflow`).

        Args:
            enabled: Whether SnapFlow self-distillation is active.
            alpha: Mixing ratio between FM and consistency objectives.
            lambda_: Weight for the consistency (shortcut) loss component.
            num_inference_steps: Number of denoising steps at inference when enabled.
        """
        self._snapflow_enabled = enabled
        self._snapflow_alpha = alpha
        self._snapflow_lambda = lambda_
        self._snapflow_num_inference_steps = num_inference_steps

    def snapflow_mixed_loss(  # noqa: PLR0914
        self,
        *,
        u_t: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
        actions: torch.Tensor,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        sample_noise: Callable[[tuple[int, ...], torch.device], torch.Tensor],
        predict_velocity: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """Compute the SnapFlow mixed FM/consistency-distillation loss.

        Splits the batch into a flow-matching (FM) fraction (``alpha``) and a
        consistency-distillation (CD) fraction (``1 - alpha``). The FM fraction
        uses the standard flow-matching objective (matching ``predict_velocity``
        against ``u_t`` at matching source/target time). The CD fraction
        bootstraps a two-step Euler shortcut target from the model's own
        velocity predictions (with gradients detached) and trains a single-step
        shortcut to match it, scaled by ``snapflow_lambda``.

        Args:
            u_t: Target velocity ``noise - actions``, shape ``(B, T, D)``.
            x_t: Noisy interpolated actions ``time * noise + (1 - time) * actions``.
            time: Sampled diffusion time, shape ``(B,)``.
            actions: Ground-truth action tensor, used only for shape/device/dtype.
            prefix_embs: Precomputed prefix (vision+language(+state)) embeddings.
            prefix_pad_masks: Padding mask for the prefix sequence.
            prefix_att_masks: Attention mask for the prefix sequence.
            sample_noise: Callable ``(shape, device) -> Tensor`` sampling noise
                with the given shape.
            predict_velocity: Callable
                ``(x_t, timestep, target_time, prefix_embs, prefix_pad_masks, prefix_att_masks) -> Tensor``
                predicting the velocity field.

        Returns:
            Per-element loss tensor, same shape as ``actions``.
        """
        bsize = actions.shape[0]
        device = actions.device
        fm_mask = torch.rand(bsize, device=device) < self._snapflow_alpha
        fm_idx = fm_mask.nonzero(as_tuple=True)[0]
        cd_idx = (~fm_mask).nonzero(as_tuple=True)[0]

        losses = torch.zeros_like(actions)

        if fm_idx.numel() > 0:
            v_fm = predict_velocity(
                x_t[fm_idx],
                time[fm_idx],
                time[fm_idx],
                prefix_embs[fm_idx],
                prefix_pad_masks[fm_idx],
                prefix_att_masks[fm_idx],
            )
            losses[fm_idx] = F.mse_loss(u_t[fm_idx], v_fm, reduction="none")

        if cd_idx.numel() > 0:
            cd_bsize = cd_idx.numel()
            cd_actions_shape = (cd_bsize, *actions.shape[1:])
            x_1 = sample_noise(cd_actions_shape, device)
            cd_prefix_embs = prefix_embs[cd_idx]
            cd_prefix_pad_masks = prefix_pad_masks[cd_idx]
            cd_prefix_att_masks = prefix_att_masks[cd_idx]

            with torch.no_grad():
                t1 = torch.ones(cd_bsize, dtype=torch.float32, device=device)
                v_1 = predict_velocity(x_1, t1, t1, cd_prefix_embs, cd_prefix_pad_masks, cd_prefix_att_masks)
                x_half = x_1 - 0.5 * v_1
                t_half = torch.full((cd_bsize,), 0.5, dtype=torch.float32, device=device)
                v_half = predict_velocity(
                    x_half,
                    t_half,
                    t_half,
                    cd_prefix_embs,
                    cd_prefix_pad_masks,
                    cd_prefix_att_masks,
                )
                v_target = 0.5 * (v_1 + v_half)

            t1 = torch.ones(cd_bsize, dtype=torch.float32, device=device)
            t_zero = torch.zeros(cd_bsize, dtype=torch.float32, device=device)
            v_pred = predict_velocity(x_1, t1, t_zero, cd_prefix_embs, cd_prefix_pad_masks, cd_prefix_att_masks)
            losses[cd_idx] = self._snapflow_lambda * F.mse_loss(v_pred, v_target.detach(), reduction="none")

        return losses

    def snapflow_num_inference_steps(self, default: int) -> int:
        """Return the number of inference denoising steps to use.

        Args:
            default: Number of steps to use when SnapFlow is not enabled.

        Returns:
            ``snapflow_num_inference_steps`` when SnapFlow is enabled, else ``default``.
        """
        return self._snapflow_num_inference_steps if self._snapflow_enabled else default

    def snapflow_target_time(self, bsize: int, time_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Return the target-time tensor to condition the velocity prediction on.

        Args:
            bsize: Batch size.
            time_tensor: The current source-time tensor, used verbatim when
                SnapFlow is not enabled.
            device: Device to allocate the zero target-time tensor on.

        Returns:
            A zero tensor of shape ``(bsize,)`` when SnapFlow is enabled (the
            1-NFE shortcut always targets ``t=0``), else ``time_tensor``.
        """
        if not self._snapflow_enabled:
            return time_tensor

        return torch.zeros(bsize, dtype=torch.float32, device=device)
