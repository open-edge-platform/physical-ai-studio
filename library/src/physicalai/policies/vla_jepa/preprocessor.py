# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""Pre- and postprocessing for VLA-JEPA.

Studio equivalent of LeRobot's ``lerobot.policies.vla_jepa.processor_vla_jepa`` pipeline.

The ordering is load-bearing and mirrors OpenPI:

    raw -> relative -> normalize -> model -> denormalize -> absolute -> gripper binarization

Handles:
- Image float cast, single-channel expansion and optional resize
- Optional absolute-to-relative action conversion (reversed on the way out)
- State / action normalization and action denormalization
- Optional action clipping and LIBERO gripper post-processing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F  # noqa: N812

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.data.observation import ACTION, IMAGES, STATE, Observation
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import VLAJEPAConfig

logger = logging.getLogger(__name__)


def _norm_type(name: str) -> NormalizationType:
    """Resolve a normalization type from its config string.

    Args:
        name: Normalization name, e.g. ``"MEAN_STD"``.

    Returns:
        The matching :class:`NormalizationType`.

    Raises:
        ValueError: If the name is not a known normalization type.
    """
    try:
        return NormalizationType(name.upper())
    except ValueError as e:
        supported = ", ".join(t.value for t in NormalizationType)
        msg = f"Unknown normalization '{name}'. Supported: {supported}."
        raise ValueError(msg) from e


class RelativeActionTransform:
    """Converts absolute actions to relative offsets and back.

    Mirrors OpenPI's ``DeltaActions``/``AbsoluteActions`` pair as ported in LeRobot's
    ``RelativeActionsProcessorStep``. A single instance is shared between the pre- and
    postprocessor so the reference state cached on the way in is available on the way out.

    Args:
        enabled: Whether the conversion is applied at all.
        exclude_joints: Joint names kept absolute. Empty means every dimension is converted.
        action_names: Per-dimension action names used to build the exclusion mask. When None,
            every dimension is converted.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        exclude_joints: Sequence[str] | None = None,
        action_names: Sequence[str] | None = None,
    ) -> None:
        """Initialize the transform.

        Args:
            enabled: Whether the conversion is applied at all.
            exclude_joints: Joint names kept absolute.
            action_names: Per-dimension action names used to build the exclusion mask.
        """
        self.enabled = enabled
        self.exclude_joints = list(exclude_joints or [])
        self.action_names = list(action_names) if action_names else None
        self._cached_state: torch.Tensor | None = None

    def build_mask(self, action_dim: int) -> list[bool]:
        """Build the per-dimension mask of dimensions converted to relative.

        Args:
            action_dim: Dimensionality of the action vector.

        Returns:
            A list of booleans, True where the dimension is converted.
        """
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))

        return mask

    def cache_state(self, state: torch.Tensor | None) -> None:
        """Cache the reference state used to reverse the conversion.

        Args:
            state: Current robot state of shape ``(B, state_dim)``, or None.
        """
        if state is not None:
            self._cached_state = state.detach()

    @property
    def cached_state(self) -> torch.Tensor | None:
        """The state cached by the preprocessor, or None when nothing ran yet."""
        return self._cached_state

    def _offset(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        mask = torch.tensor(
            self.build_mask(actions.shape[-1]),
            dtype=actions.dtype,
            device=actions.device,
        )
        state = state.to(device=actions.device, dtype=actions.dtype)
        offset = state[..., : mask.shape[0]] * mask
        chunked_action_dims = 3
        if actions.ndim == chunked_action_dims:
            offset = offset.unsqueeze(-2)
        return offset

    def to_relative(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Convert absolute actions to relative offsets.

        Args:
            actions: Actions of shape ``(B, T, action_dim)`` or ``(B, action_dim)``.
            state: Reference state of shape ``(B, state_dim)``.

        Returns:
            The actions with the masked dimensions shifted by the state.
        """
        offset = self._offset(actions, state)
        actions = actions.clone()
        actions[..., : offset.shape[-1]] -= offset
        return actions

    def to_absolute(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert relative actions back to absolute positions.

        Args:
            actions: Relative actions of shape ``(B, T, action_dim)`` or ``(B, action_dim)``.

        Returns:
            The actions shifted back by the cached reference state.

        Raises:
            RuntimeError: If the preprocessor has not cached a state yet.
        """
        if self._cached_state is None:
            msg = (
                "Relative actions are enabled but no state has been cached. The preprocessor must "
                "run before the postprocessor."
            )
            raise RuntimeError(msg)
        offset = self._offset(actions, self._cached_state)
        actions = actions.clone()
        actions[..., : offset.shape[-1]] += offset
        return actions


def prepare_images(
    image: torch.Tensor,
    resize_to: tuple[int, int] | None = None,
    *,
    expand_channels: bool = True,
) -> torch.Tensor:
    """Prepare an image tensor for the Qwen and V-JEPA fast processors.

    Casts to float, expands a single channel to three and optionally resizes. Idempotent, so the
    model can apply the same guard on batches that never went through this preprocessor. Works for
    any channels-first layout (the channel dim is -3): ``[C, H, W]``, ``[B, C, H, W]``,
    ``[B, T, C, H, W]``.

    Args:
        image: Image or video tensor, channels-first, values in [0, 1].
        resize_to: Target ``(height, width)``, or None to leave the resolution untouched.
        expand_channels: Whether a single channel is repeated to three.

    Returns:
        The prepared float tensor.
    """
    image = image.float()
    grayscale_channels = 1
    if expand_channels and image.shape[-3] == grayscale_channels:
        repeats = [1] * image.ndim
        repeats[-3] = 3
        image = image.repeat(*repeats)
    if resize_to is not None and tuple(image.shape[-2:]) != tuple(resize_to):
        device = image.device
        # NOTE: there is no "area" kernel on mps; resize on cpu and move back.
        if device.type == "mps":
            image = image.cpu()
        lead = image.shape[:-3]
        channels, height, width = image.shape[-3:]
        flat = F.interpolate(image.reshape(-1, channels, height, width), size=resize_to, mode="area")
        image = flat.reshape(*lead, channels, *resize_to).to(device)
    return image


class VLAJEPAPreprocessor(torch.nn.Module):
    """Preprocessor for VLA-JEPA model inputs.

    Args:
        features: Feature descriptors used to build the normalization buffers. When None, no
            normalization is applied.
        state_normalization: Normalization applied to the robot state.
        action_normalization: Normalization applied to actions.
        resize_images_to: Target ``(height, width)`` for input images, or None.
        relative_transform: Shared relative-action transform, or None to disable it.

    Example:
        >>> preprocessor = VLAJEPAPreprocessor(features=features)
        >>> batch = preprocessor(raw_batch)
    """

    def __init__(
        self,
        features: dict[str, Feature] | None = None,
        *,
        state_normalization: str = "MEAN_STD",
        action_normalization: str = "MIN_MAX",
        resize_images_to: tuple[int, int] | None = None,
        relative_transform: RelativeActionTransform | None = None,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            features: Feature descriptors used to build the normalization buffers.
            state_normalization: Normalization applied to the robot state.
            action_normalization: Normalization applied to actions.
            resize_images_to: Target ``(height, width)`` for input images, or None.
            relative_transform: Shared relative-action transform, or None to disable it.
        """
        super().__init__()

        self.resize_images_to: tuple[int, int] | None = (
            (resize_images_to[0], resize_images_to[1]) if resize_images_to else None
        )
        self.relative_transform = relative_transform

        if features is not None:
            norm_map = {
                FeatureType.STATE: _norm_type(state_normalization),
                FeatureType.ACTION: _norm_type(action_normalization),
            }
            self._state_action_normalizer: torch.nn.Module = FeatureNormalizeTransform(features, norm_map)
        else:
            self._state_action_normalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare images, apply the relative conversion and normalize state and actions.

        Args:
            batch: Flattened observation dict with ``images.*``, ``state``, ``action`` and ``task``.

        Returns:
            The processed batch, ready for :class:`~physicalai.policies.vla_jepa.VLAJEPAModel`.
        """
        batch = dict(batch)
        for key in Observation.get_flattened_keys(batch, IMAGES):
            if "is_pad" in key or batch.get(key) is None:
                continue
            batch[key] = prepare_images(batch[key], self.resize_images_to)

        batch = self._apply_relative_actions(batch)
        return self._state_action_normalizer(batch)

    def _apply_relative_actions(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Cache the current state and convert actions to relative offsets.

        Args:
            batch: Flattened observation dict.

        Returns:
            The batch, with actions converted when the transform is enabled.
        """
        if self.relative_transform is None:
            return batch

        state = batch.get(STATE)
        if state is not None and state.ndim > 2:  # noqa: PLR2004
            # Observation deltas are forward-looking, so index 0 is the current observation.
            state = state[:, 0, :]
        self.relative_transform.cache_state(state)

        action = batch.get(ACTION)
        if self.relative_transform.enabled and action is not None and state is not None:
            batch[ACTION] = self.relative_transform.to_relative(action, state)
        return batch


class VLAJEPAPostprocessor(torch.nn.Module):
    """Postprocessor for VLA-JEPA model outputs.

    Args:
        features: Feature descriptors used to build the denormalization buffers. When None, no
            denormalization is applied.
        action_normalization: Normalization that was applied to actions.
        clip_normalized_actions: Whether to clip normalized actions to [-1, 1]. Only honored under
            MIN_MAX normalization; a warning is logged otherwise.
        pre_snap_gripper_action: Whether to snap the gripper to {0, 1} before denormalization.
        binarize_gripper_action: Whether to binarize the gripper to {-1, 1} after denormalization.
        gripper_dim: Index of the gripper in the action vector.
        gripper_threshold: Threshold used by the gripper steps.
        relative_transform: Shared relative-action transform, or None to disable it.
    """

    def __init__(
        self,
        features: dict[str, Feature] | None = None,
        *,
        action_normalization: str = "MIN_MAX",
        clip_normalized_actions: bool = True,
        pre_snap_gripper_action: bool = False,
        binarize_gripper_action: bool = False,
        gripper_dim: int = 6,
        gripper_threshold: float = 0.5,
        relative_transform: RelativeActionTransform | None = None,
    ) -> None:
        """Initialize the postprocessor.

        Args:
            features: Feature descriptors used to build the denormalization buffers.
            action_normalization: Normalization that was applied to actions.
            clip_normalized_actions: Whether to clip normalized actions to [-1, 1].
            pre_snap_gripper_action: Whether to snap the gripper before denormalization.
            binarize_gripper_action: Whether to binarize the gripper after denormalization.
            gripper_dim: Index of the gripper in the action vector.
            gripper_threshold: Threshold used by the gripper steps.
            relative_transform: Shared relative-action transform, or None to disable it.
        """
        super().__init__()

        action_norm = _norm_type(action_normalization)
        self.pre_snap_gripper_action = pre_snap_gripper_action
        self.binarize_gripper_action = binarize_gripper_action
        self.gripper_dim = gripper_dim
        self.gripper_threshold = gripper_threshold
        self.relative_transform = relative_transform

        # Clipping to [-1, 1] is a range assertion under MIN_MAX, but under MEAN_STD the same clamp
        # truncates every action beyond 1 sigma. That shows up as a hesitant, low-amplitude policy
        # with no error anywhere, so refuse the flag instead of honoring it.
        self.clip_normalized_actions = clip_normalized_actions and action_norm is NormalizationType.MIN_MAX
        if clip_normalized_actions and not self.clip_normalized_actions:
            logger.warning(
                "`clip_normalized_actions=True` is ignored: it clips normalized actions to [-1, 1], "
                "which is only a no-op bound under MIN_MAX, but actions use %s. Under %s this would "
                "clamp every action to 1 sigma.",
                action_norm.value,
                action_norm.value,
            )

        if features is not None:
            action_features = {k: v for k, v in features.items() if v.ftype == FeatureType.ACTION}
            self._action_denormalizer: torch.nn.Module = FeatureNormalizeTransform(
                action_features,
                {FeatureType.ACTION: action_norm},
                inverse=True,
            )
        else:
            self._action_denormalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Clip, denormalize and restore actions to the robot's action space.

        Args:
            batch: Dict that may contain an ``action`` tensor produced by the model.

        Returns:
            A dict with the action converted back to the dataset's action space.
        """
        batch = dict(batch)
        action = batch.get(ACTION)
        if action is None:
            return batch

        if self.clip_normalized_actions:
            action = action.clamp(-1.0, 1.0)
        if self.pre_snap_gripper_action:
            action = self._snap_gripper(action)

        batch[ACTION] = action
        batch = self._action_denormalizer(batch)

        if self.relative_transform is not None and self.relative_transform.enabled:
            batch[ACTION] = self.relative_transform.to_absolute(batch[ACTION])
        if self.binarize_gripper_action:
            batch[ACTION] = self._binarize_gripper(batch[ACTION])
        return batch

    def _snap_gripper(self, action: torch.Tensor) -> torch.Tensor:
        """Snap the gripper dimension to {0, 1} in normalized space (starVLA LIBERO eval).

        Args:
            action: Normalized action tensor.

        Returns:
            The action with its gripper dimension snapped, or unchanged when out of range.
        """
        if action.shape[-1] <= self.gripper_dim:
            return action
        action = action.clone()
        action[..., self.gripper_dim] = (action[..., self.gripper_dim] >= self.gripper_threshold).float()
        return action

    def _binarize_gripper(self, action: torch.Tensor) -> torch.Tensor:
        """Map the gripper dimension to {-1, 1}: above the threshold -> -1, else 1.

        Args:
            action: Unnormalized action tensor.

        Returns:
            The action with its gripper dimension binarized, or unchanged when out of range.
        """
        if action.shape[-1] <= self.gripper_dim:
            return action
        action = action.clone()
        gripper = action[..., self.gripper_dim]
        action[..., self.gripper_dim] = 1.0 - 2.0 * (gripper > self.gripper_threshold).float()
        return action


def _warn_if_gripper_steps_are_misconfigured(
    config: VLAJEPAConfig,
    gripper_dim: int,
    stats: dict[str, dict[str, Any]] | None,
) -> None:
    """Warn when the gripper post-steps would pin the gripper to a constant.

    The binarization step thresholds the *unnormalized* gripper at `gripper_threshold`. When the
    dataset's physical range sits well above it, every value lands on the same side and the gripper
    never moves. That is detectable from the stats, so say so.

    Args:
        config: The policy config carrying the gripper flags and threshold.
        gripper_dim: Resolved gripper index.
        stats: Dataset statistics, or None.
    """
    if not (config.pre_snap_gripper_action or config.binarize_gripper_action):
        return
    action_stats = (stats or {}).get(ACTION)
    if not action_stats or "min" not in action_stats or "max" not in action_stats:
        return
    try:
        low = float(cast("list[float]", action_stats["min"])[gripper_dim])
        high = float(cast("list[float]", action_stats["max"])[gripper_dim])
    except (IndexError, TypeError, ValueError):
        return
    threshold = config.gripper_threshold
    # `pre_snap` writes {0, 1} in normalized space, which unnormalizes to the midpoint and the max.
    # Both landing on the same side of the threshold means a constant output.
    midpoint = (low + high) / 2.0
    if (midpoint > threshold) == (high > threshold):
        names = config.action_feature_names
        name = names[gripper_dim] if names and gripper_dim < len(names) else f"dim {gripper_dim}"
        logger.warning(
            "vla_jepa gripper post-processing looks misconfigured: action %s has a physical range "
            "of [%.3g, %.3g], and `gripper_threshold=%s` is compared against that unnormalized "
            "value. Both %.3g and %.3g fall on the same side of it, so the commanded gripper will "
            "be constant. Set `gripper_threshold` in the gripper's own units, or set "
            "`pre_snap_gripper_action=false` and `binarize_gripper_action=false` (the defaults) "
            "unless you are running LIBERO.",
            name,
            low,
            high,
            threshold,
            midpoint,
            high,
        )


def features_from_stats(stats: dict[str, dict[str, Any]] | None) -> dict[str, Feature]:
    """Build state/action feature descriptors from dataset statistics.

    Args:
        stats: Dataset statistics keyed by feature name (e.g. ``observation.state``, ``action``).

    Returns:
        Mapping from the raw feature name to its :class:`~physicalai.data.Feature`.
    """
    features: dict[str, Feature] = {}
    for key, stat in (stats or {}).items():
        if ACTION in key:
            feature_type = FeatureType.ACTION
        elif STATE in key:
            feature_type = FeatureType.STATE
        else:
            continue
        features[str(stat["name"])] = Feature(
            name=str(stat["name"]),
            ftype=feature_type,
            shape=cast("tuple[int, ...]", stat["shape"]),
            normalization_data=NormalizationParameters(
                mean=cast("list[float]", stat.get("mean")),
                std=cast("list[float]", stat.get("std")),
                min=cast("list[float]", stat.get("min")),
                max=cast("list[float]", stat.get("max")),
                q01=cast("list[float]", stat.get("q01")),
                q99=cast("list[float]", stat.get("q99")),
            ),
        )
    return features


_REQUIRED_STAT_FIELDS = {
    NormalizationType.MEAN_STD: ("mean", "std"),
    NormalizationType.MIN_MAX: ("min", "max"),
    NormalizationType.QUANTILES: ("q01", "q99"),
}


def _validate_stat_fields(features: dict[str, Feature], ftype: FeatureType, norm: NormalizationType) -> None:
    """Fail early when the statistics lack the fields the normalization needs.

    Args:
        features: Feature descriptors built from the dataset statistics.
        ftype: The feature type to check.
        norm: The normalization configured for that feature type.

    Raises:
        ValueError: If a feature of this type is missing a required statistic.
    """
    required = _REQUIRED_STAT_FIELDS.get(norm)
    if required is None:
        return
    for name, feature in features.items():
        if feature.ftype is not ftype or feature.normalization_data is None:
            continue
        missing = [field for field in required if getattr(feature.normalization_data, field) is None]
        if missing:
            msg = (
                f"Feature '{name}' has no {'/'.join(missing)} statistics, which {norm.value} "
                f"normalization requires. Provide dataset stats carrying them, or configure a "
                f"different normalization for this feature type."
            )
            raise ValueError(msg)


def make_vla_jepa_preprocessors(
    config: VLAJEPAConfig,
    stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[VLAJEPAPreprocessor, VLAJEPAPostprocessor]:
    """Create a matched preprocessor / postprocessor pair.

    The two share a single :class:`RelativeActionTransform`, so the state cached while preprocessing
    is the reference used to restore absolute actions on the way out.

    Args:
        config: Policy configuration.
        stats: Dataset statistics used to build the normalization buffers.

    Returns:
        Tuple of (preprocessor, postprocessor).
    """
    features = features_from_stats(stats)
    if features:
        _validate_stat_fields(features, FeatureType.STATE, _norm_type(config.state_normalization))
        _validate_stat_fields(features, FeatureType.ACTION, _norm_type(config.action_normalization))
    gripper_dim = config.resolved_gripper_dim
    _warn_if_gripper_steps_are_misconfigured(config, gripper_dim, stats)

    relative_transform = RelativeActionTransform(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )

    preprocessor = VLAJEPAPreprocessor(
        features=features or None,
        state_normalization=config.state_normalization,
        action_normalization=config.action_normalization,
        resize_images_to=config.resize_images_to,
        relative_transform=relative_transform,
    )
    postprocessor = VLAJEPAPostprocessor(
        features=features or None,
        action_normalization=config.action_normalization,
        clip_normalized_actions=config.clip_normalized_actions,
        pre_snap_gripper_action=config.pre_snap_gripper_action,
        binarize_gripper_action=config.binarize_gripper_action,
        gripper_dim=gripper_dim,
        gripper_threshold=config.gripper_threshold,
        relative_transform=relative_transform,
    )
    return preprocessor, postprocessor
