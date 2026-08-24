# Copyright 2025 2toINF (https://github.com/2toINF) and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action spaces for the XVLA policy.

XVLA is a *multi-embodiment* model: the transformer head always predicts a fixed-width
action vector, and an :class:`ActionSpace` describes how that vector maps onto a concrete
robot -- which channels are grippers, which loss each channel gets, and how a prediction is
turned back into a control vector.

Spaces are registered by name so ``XVLAConfig.action_mode`` selects one from a config file:

    >>> from physicalai.policies.xvla.action_hub import build_action_space
    >>> space = build_action_space("auto", real_dim=7, max_dim=20)
    >>> space.dim_action
    20

``"auto"`` is the embodiment-agnostic default: the model keeps its pretrained 20-channel
width, the loss is taken only over the dataset's real channels, and predictions are trimmed
back to that width. The other spaces reproduce the fixed layouts published with the
upstream XVLA checkpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

ACTION_REGISTRY: dict[str, type[BaseActionSpace]] = {}
"""Registry of action spaces, keyed by their lowercase name."""


def register_action(name: str) -> Callable[[type[BaseActionSpace]], type[BaseActionSpace]]:
    """Register an action space under ``name``.

    Args:
        name: Name used by ``XVLAConfig.action_mode``; matched case-insensitively.

    Returns:
        A class decorator that records the class in :data:`ACTION_REGISTRY`.
    """

    def _wrap(cls: type[BaseActionSpace]) -> type[BaseActionSpace]:
        key = name.lower()
        if key in ACTION_REGISTRY:
            msg = f"ActionSpace {key!r} is already registered to {ACTION_REGISTRY[key]}"
            raise KeyError(msg)
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls

    return _wrap


def build_action_space(name: str, **kwargs: int) -> BaseActionSpace:
    """Instantiate a registered action space by name.

    Args:
        name: Registered name, matched case-insensitively.
        **kwargs: Constructor arguments (``"auto"`` takes ``real_dim`` and ``max_dim``).

    Returns:
        The instantiated action space.

    Raises:
        KeyError: If no action space is registered under ``name``.
    """
    key = name.lower()
    if key not in ACTION_REGISTRY:
        msg = f"Unknown action space {name!r}. Available: {sorted(ACTION_REGISTRY)}"
        raise KeyError(msg)
    return ACTION_REGISTRY[key](**kwargs)


def _ensure_indices_valid(dim_action: int, idx: Iterable[int], name: str) -> None:
    """Check that channel indices fall inside the action vector.

    Args:
        dim_action: Width of the action vector.
        idx: Channel indices to validate.
        name: Name of the index group, used in the error message.

    Raises:
        IndexError: If any index is out of range.
    """
    bad = [i for i in idx if i < 0 or i >= dim_action]
    if bad:
        msg = f"{name} contains out-of-range indices {bad} for action dim {dim_action}"
        raise IndexError(msg)


class BaseActionSpace(nn.Module):
    """Base class for XVLA action-space definitions.

    Subclasses declare the width of the model-facing action vector and how it is
    supervised and decoded:

    - ``dim_action``: width of the vector the transformer head predicts.
    - ``gripper_idx``: channels holding gripper commands.
    - :meth:`compute_loss`: the supervised loss, returned per component so each term can be
      logged separately.
    - :meth:`preprocess`: adjusts proprioception and the (noised) action before they enter
      the transformer -- padding narrow vectors, masking channels that are not supervised.
    - :meth:`postprocess`: turns a raw prediction into a control vector (gripper logits to
      probabilities, model width back to robot width).
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: tuple[int, ...] = ()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the supervised loss for this action space.

        Args:
            pred: Predicted actions of shape ``[B, T, dim_action]``.
            target: Ground-truth actions, broadcastable to ``pred``.

        Returns:
            Loss components keyed by name; the training loss is their sum.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Alias for :meth:`compute_loss` so the space can be called as a module.

        Args:
            pred: Predicted actions.
            target: Ground-truth actions.

        Returns:
            Loss components keyed by name.
        """
        return self.compute_loss(pred, target)

    def preprocess(  # noqa: PLR6301
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adjust proprioception and the noised action before the transformer sees them.

        Args:
            proprio: Proprioceptive state of shape ``[B, dim_proprio]``.
            action: Noised actions of shape ``[B, T, D]``.

        Returns:
            Tuple of ``(proprio, action)``; unchanged by default.
        """
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Turn a raw prediction into a control vector.

        Args:
            action: Predicted actions of shape ``[B, T, dim_action]``.

        Returns:
            The decoded actions; unchanged by default.
        """
        return action


def _zero_grippers(
    proprio: torch.Tensor,
    action: torch.Tensor,
    gripper_idx: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask the gripper channels out of proprioception and the noised action.

    The gripper is supervised as a binary logit rather than a continuous value, so feeding
    its noised value back into the transformer would leak an uncalibrated scale.

    Args:
        proprio: Proprioceptive state.
        action: Noised actions.
        gripper_idx: Gripper channel indices.

    Returns:
        Tuple of masked ``(proprio, action)`` copies.
    """
    proprio_masked = proprio.clone()
    action_masked = action.clone()
    if proprio_masked.shape[-1] > max(gripper_idx):
        proprio_masked[..., gripper_idx] = 0.0
    action_masked[..., gripper_idx] = 0.0
    return proprio_masked, action_masked


@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    """Bimanual end-effector layout: xyz, 6D rotation and a gripper logit per arm.

    Channels ``0-9`` are the left arm and ``10-19`` the right arm, each laid out as
    ``[xyz (3), rot6d (6), gripper (1)]``. Position and rotation are supervised with a
    scaled MSE and the gripper with a binary cross-entropy, so all three terms reach the
    same order of magnitude.
    """

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self) -> None:
        """Build the MSE and BCE criteria."""
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the position, rotation and gripper losses.

        Args:
            pred: Predicted actions of shape ``[B, T, 20]``.
            target: Ground-truth actions of shape ``[B, T, 20]``.

        Returns:
            Dict with ``position_loss``, ``rotate6D_loss`` and ``gripper_loss``.

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
        """
        if pred.shape != target.shape:
            msg = f"pred/target shapes must match, got {tuple(pred.shape)} vs {tuple(target.shape)}"
            raise ValueError(msg)
        _ensure_indices_valid(pred.shape[-1], self.gripper_idx, "gripper_idx")

        gripper_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = torch.stack(gripper_losses).mean() * self.GRIPPER_SCALE

        position_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1])
            + self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        rotation_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1])
            + self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": position_loss,
            "rotate6D_loss": rotation_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero the gripper channels of proprioception and the noised action.

        Args:
            proprio: Proprioceptive state.
            action: Noised actions.

        Returns:
            Tuple of masked ``(proprio, action)``.
        """
        return _zero_grippers(proprio, action, self.gripper_idx)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Map the gripper logits through a sigmoid.

        Args:
            action: Predicted actions.

        Returns:
            Actions with gripper channels in ``[0, 1]``.
        """
        if action.size(-1) <= max(self.gripper_idx):
            return action
        action = action.clone()
        action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("joint")
class JointActionSpace(BaseActionSpace):
    """Bimanual joint-space layout: six joints plus a gripper logit per arm."""

    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self) -> None:
        """Build the MSE and BCE criteria."""
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the joint and gripper losses.

        Args:
            pred: Predicted actions of shape ``[B, T, 14]``.
            target: Ground-truth actions of shape ``[B, T, 14]``.

        Returns:
            Dict with ``joints_loss`` and ``gripper_loss``.

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
        """
        if pred.shape != target.shape:
            msg = f"pred/target shapes must match, got {tuple(pred.shape)} vs {tuple(target.shape)}"
            raise ValueError(msg)
        action_dim = pred.shape[-1]
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        gripper_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = torch.stack(gripper_losses).mean() * self.GRIPPER_SCALE

        joints_idx = tuple(i for i in range(action_dim) if i not in set(self.gripper_idx))
        joints_loss = self.mse(pred[:, :, joints_idx], target[:, :, joints_idx]) * self.JOINTS_SCALE

        return {"joints_loss": joints_loss, "gripper_loss": gripper_loss}

    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero the gripper channels of proprioception and the noised action.

        Args:
            proprio: Proprioceptive state.
            action: Noised actions.

        Returns:
            Tuple of masked ``(proprio, action)``.
        """
        return _zero_grippers(proprio, action, self.gripper_idx)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Map the gripper logits through a sigmoid.

        Args:
            action: Predicted actions.

        Returns:
            Actions with gripper channels in ``[0, 1]``.
        """
        if action.size(-1) <= max(self.gripper_idx):
            return action
        action = action.clone()
        action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("agibot_ee6d")
class AGIBOTEE6DActionSpace(BaseActionSpace):
    """AGIBOT variant of :class:`EE6DActionSpace` that supervises the gripper with MSE.

    The AGIBOT datasets record a continuous gripper opening rather than a binary
    open/close, so every channel -- grippers included -- is a regression target and no
    channel is masked out of the transformer's input.
    """

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 10.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self) -> None:
        """Build the MSE criterion."""
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the position, rotation and gripper losses, all as scaled MSE.

        Args:
            pred: Predicted actions of shape ``[B, T, 20]``.
            target: Ground-truth actions of shape ``[B, T, 20]``.

        Returns:
            Dict with ``position_loss``, ``rotate6D_loss`` and ``gripper_loss``.

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
        """
        if pred.shape != target.shape:
            msg = f"pred/target shapes must match, got {tuple(pred.shape)} vs {tuple(target.shape)}"
            raise ValueError(msg)
        _ensure_indices_valid(pred.shape[-1], self.gripper_idx, "gripper_idx")

        gripper_loss = self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx]) * self.GRIPPER_SCALE
        position_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1])
            + self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE
        rotation_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1])
            + self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": position_loss,
            "rotate6D_loss": rotation_loss,
            "gripper_loss": gripper_loss,
        }


class _PaddedActionSpace(BaseActionSpace):
    """Base for spaces whose robot is narrower than the model's action vector.

    The transformer keeps its pretrained width so published checkpoints stay loadable,
    while the dataset's actions are zero-padded into it and predictions are trimmed back
    down for the robot.

    Attributes:
        real_dim: Width of the robot's control vector.
    """

    real_dim: int = 0

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-pad the last dimension from ``real_dim`` to ``dim_action``.

        Args:
            x: Tensor whose last dimension is ``real_dim`` or already ``dim_action``.

        Returns:
            Tensor whose last dimension is ``dim_action``.

        Raises:
            ValueError: If the last dimension matches neither width.
        """
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.real_dim:
            msg = f"Expected last dim {self.real_dim} or {self.dim_action}, got {x.size(-1)}"
            raise ValueError(msg)
        pad = x.new_zeros((*x.shape[:-1], self.dim_action - self.real_dim))
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Trim the last dimension down to ``real_dim``.

        Args:
            x: Tensor of width ``dim_action``.

        Returns:
            Tensor of width ``real_dim``.
        """
        return x[..., : self.real_dim]


@register_action("franka_joint7")
class FrankaJoint7ActionSpace(_PaddedActionSpace):
    """Franka Panda joint space: 7 joints padded into the model's 20 channels."""

    dim_action = 20
    real_dim = 7
    JOINTS_SCALE = 1.0

    def __init__(self) -> None:
        """Build the MSE criterion."""
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the joint MSE over the seven real channels.

        Args:
            pred: Predicted actions of shape ``[B, T, 20]``.
            target: Ground-truth actions of shape ``[B, T, 7]`` or ``[B, T, 20]``.

        Returns:
            Dict with ``joints_loss``.
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        joints_loss = self.mse(pred[:, :, : self.real_dim], target[:, :, : self.real_dim]) * self.JOINTS_SCALE
        return {"joints_loss": joints_loss}

    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad the noised action into the model's width.

        Args:
            proprio: Proprioceptive state, passed through unchanged.
            action: Noised actions.

        Returns:
            Tuple of ``(proprio, padded action)``.
        """
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Trim the prediction down to the seven Franka joints.

        Args:
            action: Predicted actions of shape ``[B, T, 20]``.

        Returns:
            Actions of shape ``[B, T, 7]``.
        """
        return self._trim_to_real_dim(action)


@register_action("auto")
class AutoActionSpace(_PaddedActionSpace):
    """Embodiment-agnostic space that adapts to the dataset's action width.

    The transformer keeps its pretrained ``max_dim`` width, the loss covers only the
    dataset's ``real_dim`` channels, and predictions are trimmed back to ``real_dim``. This
    is the default for Studio, where the real width is read from the training dataset's
    statistics.

    Args:
        real_dim: Action width of the dataset.
        max_dim: Action width the model predicts.
    """

    JOINTS_SCALE = 1.0

    def __init__(self, real_dim: int, max_dim: int) -> None:
        """Build the MSE criterion and record both widths.

        Raises:
            ValueError: If ``real_dim`` is not a positive width no wider than ``max_dim``.
        """
        super().__init__()
        if real_dim <= 0 or real_dim > max_dim:
            msg = f"real_dim must be in [1, max_dim={max_dim}], got {real_dim}"
            raise ValueError(msg)
        self.real_dim = real_dim
        self.dim_action = max_dim
        self.mse = nn.MSELoss()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the MSE over the dataset's real action channels.

        Args:
            pred: Predicted actions of shape ``[B, T, max_dim]``.
            target: Ground-truth actions of shape ``[B, T, real_dim]`` or ``[B, T, max_dim]``.

        Returns:
            Dict with ``joints_loss``.
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        joints_loss = self.mse(pred[:, :, : self.real_dim], target[:, :, : self.real_dim]) * self.JOINTS_SCALE
        return {"joints_loss": joints_loss}

    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad the noised action into the model's width.

        Args:
            proprio: Proprioceptive state, passed through unchanged.
            action: Noised actions.

        Returns:
            Tuple of ``(proprio, padded action)``.
        """
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Trim the prediction down to the dataset's action width.

        Args:
            action: Predicted actions of shape ``[B, T, max_dim]``.

        Returns:
            Actions of shape ``[B, T, real_dim]``.
        """
        return self._trim_to_real_dim(action)


@register_action("so101_bimanual")
class BimanualSO101ActionSpace(_PaddedActionSpace):
    """Bimanual SO-101: five joints plus a gripper per arm, padded into 20 channels.

    Layout is ``[left (5 joints + gripper), right (5 joints + gripper)]``; the grippers sit
    at channels 5 and 11 and are regressed rather than classified.
    """

    dim_action = 20
    real_dim = 12
    gripper_idx = (5, 11)
    GRIPPER_SCALE = 1.0
    JOINTS_SCALE = 1.0
    ARM_WIDTH = 6

    def __init__(self) -> None:
        """Build the MSE criterion."""
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute per-arm, joint and gripper losses over the twelve real channels.

        Args:
            pred: Predicted actions of shape ``[B, T, 20]``.
            target: Ground-truth actions of shape ``[B, T, 12]`` or ``[B, T, 20]``.

        Returns:
            Dict with ``joints_loss``, ``gripper_loss``, ``left_arm_loss`` and ``right_arm_loss``.
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)

        joints_loss = self.mse(pred[:, :, : self.real_dim], target[:, :, : self.real_dim]) * self.JOINTS_SCALE
        left_arm_loss = self.mse(pred[:, :, : self.ARM_WIDTH], target[:, :, : self.ARM_WIDTH])
        right_arm_loss = self.mse(
            pred[:, :, self.ARM_WIDTH : self.real_dim],
            target[:, :, self.ARM_WIDTH : self.real_dim],
        )
        gripper_loss = self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx]) * self.GRIPPER_SCALE

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
            "left_arm_loss": left_arm_loss,
            "right_arm_loss": right_arm_loss,
        }

    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad both tensors into the model's width and mask the gripper channels.

        Args:
            proprio: Proprioceptive state.
            action: Noised actions.

        Returns:
            Tuple of padded, gripper-masked ``(proprio, action)``.
        """
        return _zero_grippers(proprio, self._pad_to_model_dim(action), self.gripper_idx)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Map the gripper logits through a sigmoid and trim to the twelve real channels.

        Args:
            action: Predicted actions of shape ``[B, T, 20]``.

        Returns:
            Actions of shape ``[B, T, 12]``.
        """
        if action.size(-1) > max(self.gripper_idx):
            action = action.clone()
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return self._trim_to_real_dim(action)


__all__ = [
    "ACTION_REGISTRY",
    "AGIBOTEE6DActionSpace",
    "AutoActionSpace",
    "BaseActionSpace",
    "BimanualSO101ActionSpace",
    "EE6DActionSpace",
    "FrankaJoint7ActionSpace",
    "JointActionSpace",
    "build_action_space",
    "register_action",
]
