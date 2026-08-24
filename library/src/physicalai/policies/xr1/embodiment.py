# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XR1's fixed 60-slot state and action layout, and how to map a dataset onto it.

The released XR1 checkpoints were trained on a single, fixed vector layout: 60
state slots in joint space and 60 action slots expressed as end-effector-frame
deltas. Every slot has a meaning, and the slots a given embodiment does not use are
left at zero. The layout is reproduced here from the reference implementation's
``mibot/utils/io.py`` (``compose_state`` and ``ACTION_PARTS``).

Why this matters: a LeRobot dataset is simply a vector of some width, and padding
it with :func:`~physicalai.policies.xr1.io.pad_vector` puts dimension *i* of the
dataset into slot *i*. That is the right behaviour when training from scratch,
where the slot meanings are whatever the data says they are. It is the wrong
behaviour when starting from released weights, where slot 7 is a gripper and slot 6
is the seventh joint of a 7-DoF arm. Setting
:attr:`~physicalai.policies.xr1.config.XR1Config.state_slot_map` routes each
dataset dimension to the slot the checkpoint expects instead.

Only the state mapping can be given for a dataset such as ALOHA. Its actions are
absolute joint targets, while XR1's action slots are relative end-effector poses; the
two are related by forward kinematics and a finite difference, not by any permutation
of indices, so no ``action_slot_map`` can express it. The generic
:attr:`~physicalai.policies.xr1.config.XR1Config.action_slot_map` mechanism is
provided for embodiments where such a relation does exist - LIBERO, whose actions are
already end-effector deltas, is one.
"""

from __future__ import annotations

import torch

XR1_STATE_DIM = 60
XR1_ACTION_DIM = 60

#: State slots, in joint space. ``compose_state`` in the reference implementation.
STATE_SLOTS: dict[str, tuple[int, int]] = {
    "left_arm": (0, 7),
    "left_gripper": (7, 8),
    "right_arm": (8, 15),
    "right_gripper": (15, 16),
}

#: Action slots, as end-effector-frame deltas. ``ACTION_PARTS`` in the reference
#: implementation. ``aa`` is an axis-angle rotation delta.
ACTION_SLOTS: dict[str, tuple[int, int]] = {
    "left_ee_pos": (0, 3),
    "left_ee_aa": (3, 6),
    "left_gripper": (6, 7),
    "right_ee_pos": (8, 11),
    "right_ee_aa": (11, 14),
    "right_gripper": (14, 15),
    "waist": (16, 17),
    "base": (17, 20),
}

#: ALOHA's 14-dimensional joint state onto XR1's state slots. ALOHA is two 6-DoF
#: arms plus one gripper each; XR1 reserves seven joint slots per arm, so the
#: seventh slot of each arm stays zero and the grippers move to slots 7 and 15.
ALOHA_STATE_TO_XR1: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15)


def validate_slot_map(slot_map: tuple[int, ...], target_dim: int, name: str) -> None:
    """Check a slot map is a usable injection into ``target_dim`` slots.

    Args:
        slot_map: Destination slot for each source dimension.
        target_dim: Width of the destination vector.
        name: Field name, used in error messages.

    Raises:
        ValueError: If the map is empty, has duplicates, or names a slot outside
            the destination width.
    """
    if not slot_map:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if len(set(slot_map)) != len(slot_map):
        msg = f"{name} must not map two dimensions to the same slot, got {slot_map}"
        raise ValueError(msg)
    out_of_range = [slot for slot in slot_map if not 0 <= slot < target_dim]
    if out_of_range:
        msg = f"{name} slots {out_of_range} fall outside the {target_dim} available slots"
        raise ValueError(msg)


def scatter_slots(vector: torch.Tensor, slot_map: tuple[int, ...], target_dim: int) -> torch.Tensor:
    """Route each dimension of ``vector`` into its slot, zero-filling the rest.

    Args:
        vector: Tensor whose last dimension has ``len(slot_map)`` entries.
        slot_map: Destination slot for each source dimension.
        target_dim: Width of the returned vector.

    Returns:
        Tensor with the same leading shape and last dimension ``target_dim``.

    Raises:
        ValueError: If the vector width does not match the map length.
    """
    if vector.shape[-1] != len(slot_map):
        msg = f"Vector of width {vector.shape[-1]} does not match a slot map of length {len(slot_map)}"
        raise ValueError(msg)

    index = torch.as_tensor(slot_map, device=vector.device, dtype=torch.long)
    out = vector.new_zeros((*vector.shape[:-1], target_dim))
    return out.index_copy(-1, index, vector)


def gather_slots(vector: torch.Tensor, slot_map: tuple[int, ...]) -> torch.Tensor:
    """Read the mapped slots back out, inverting :func:`scatter_slots`.

    Args:
        vector: Tensor whose last dimension covers every slot in ``slot_map``.
        slot_map: Destination slot for each source dimension.

    Returns:
        Tensor with last dimension ``len(slot_map)``, in source order.
    """
    index = torch.as_tensor(slot_map, device=vector.device, dtype=torch.long)
    return vector.index_select(-1, index)


def slot_mask(slot_map: tuple[int, ...], target_dim: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Build the boolean mask of slots a map actually fills.

    Args:
        slot_map: Destination slot for each source dimension.
        target_dim: Width of the destination vector.
        device: Device for the returned mask.

    Returns:
        Boolean tensor of shape ``(target_dim,)``.
    """
    mask = torch.zeros(target_dim, dtype=torch.bool, device=device)
    mask[torch.as_tensor(slot_map, device=device, dtype=torch.long)] = True
    return mask
