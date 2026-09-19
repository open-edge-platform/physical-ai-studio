# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-embodiment action-space mapping for the Cosmos 3 policy.

The exposed policy argument is ``embodiment`` (a string such as ``"pusht"`` or
``"droid_lerobot"``). Every embodiment maps internally to:

    * an **action space** (``identity`` or ``joint_pos``) that decides how raw dataset
      actions/state become the model action space, and
    * an **action-normalization** method (``none`` / ``minmax`` / ``quantile`` / ...).

Supported action spaces:

    * ``identity`` (pusht, aloha): raw state/action values, scaled into the model's
      ``[-1, 1]`` range with per-dataset min-max normalization.
    * ``joint_pos`` (droid_lerobot): raw 8D ``[joint(7), gripper(1)]`` action and state with a
      flipped gripper and no additional normalization -- the action space of the released DROID
      policy checkpoints (e.g. ``nvidia/cosmos3-edge-policy-droid``, whose model card documents
      an 8D DROID action), matching cosmos-framework's ``droid_lerobot_dataset``
      ``action_space="joint_pos"`` recipe.

Adding a new embodiment: register its numeric domain id in
``diffusers ... _EMBODIMENT_TO_DOMAIN_ID`` (and raw action width in
``_EMBODIMENT_TO_RAW_ACTION_DIM``), then map it here to a supported ``action_space`` (and, if
needed, an entry in the normalization / gripper-flip registries below). The datamodule is
responsible for delivering a single combined ``observation.state`` and ``action`` column
(e.g. DROID's 8D ``[joint(7), gripper(1)]``), matching the other policies in this repo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Action space per embodiment. Embodiments not listed default to ``identity``.
EMBODIMENT_ACTION_SPACE: dict[str, str] = {
    "droid_lerobot": "joint_pos",
}


def resolve_action_space(embodiment: str, override: str | None = None) -> str:
    """Return the action space for an embodiment; an explicit ``override`` wins when given."""
    if override is not None:
        return override
    return EMBODIMENT_ACTION_SPACE.get(embodiment, "identity")


# Embodiments whose raw gripper command is inverted relative to the model's convention
# (cosmos-framework ``_is_gripper_action_flipped``). DROID's LeRobot gripper is flipped.
EMBODIMENT_GRIPPER_FLIPPED: dict[str, bool] = {
    "droid_lerobot": True,
}


def embodiment_gripper_flipped(embodiment: str) -> bool:
    """Return whether the embodiment's raw gripper command must be inverted as ``1 - g``."""
    return EMBODIMENT_GRIPPER_FLIPPED.get(embodiment, False)


def flip_gripper_last_channel(action: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``action`` with its final (gripper) channel inverted as ``1 - g``."""
    flipped = action.clone()
    flipped[..., -1] = 1.0 - flipped[..., -1]
    return flipped


def uses_minmax_normalization(embodiment: str) -> bool:
    """Return whether an embodiment's state/actions are min-max normalized into ``[-1, 1]``.

    Only the ``identity`` action space relies on per-dataset min-max scaling.
    """
    return resolve_action_space(embodiment) == "identity"


# Action-normalization method per embodiment, mirroring the cosmos-framework dataset defaults.
# Embodiments not listed default to ``minmax`` (the identity per-dataset scaling); DROID trains
# raw (``none``). Values map to :mod:`normalization`.
EMBODIMENT_NORMALIZATION: dict[str, str] = {
    "droid_lerobot": "none",
}


def embodiment_normalization(embodiment: str) -> str:
    """Return the action-normalization method ("none"/"minmax"/"quantile"/...) for an embodiment."""
    return EMBODIMENT_NORMALIZATION.get(embodiment, "minmax")
