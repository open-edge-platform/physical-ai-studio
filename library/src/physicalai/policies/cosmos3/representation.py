# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-domain action-space representation for Cosmos 3 policy.

Different embodiments express proprioceptive state and actions in different spaces,
and the Cosmos 3 checkpoint expects each domain in the space it was trained on:

    * ``identity`` (pusht, aloha, libero): raw state/action values, scaled into the
      model's ``[-1, 1]`` range with per-dataset min-max normalization.
    * ``joint_pos`` (droid_lerobot): raw 8D ``[joint(7), gripper(1)]`` action and state
      with a flipped gripper and no additional normalization. This is the action space of
      the released DROID policy checkpoints (e.g. ``nvidia/cosmos3-edge-policy-droid``,
      whose model card documents an 8D DROID action), matching cosmos-framework's
      ``droid_lerobot_dataset`` ``action_space="joint_pos"`` recipe.
    * ``droid_ee`` (robomind-franka): absolute end-effector pose token
      ``[pos(3), rot6d(6), gripper(1)]`` in the OpenCV camera frame with a flipped
      gripper. Framewise-relative pose chunks for actions. Already in the checkpoint's
      native space, so no additional min-max normalization is applied.
    * ``bridge_ee`` (bridge_orig_lerobot): the WidowX-specific kinematics/TCP/OpenCV
      frame corrections with backward-framewise relative rot6d actions and the raw
      (unflipped) gripper command. Also native, so no min-max normalization.

The pose math mirrors cosmos-framework's ``droid_lerobot_dataset`` /
``bridge_orig_lerobot_dataset`` (``pose_utils.convert_rotation`` +
``pose_abs_to_rel``) exactly, so a checkpoint trained here or with cosmos-framework
is swappable in either direction.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# 90-degree clockwise rotation about Z, mapping the DROID Franka panda_link8
# orientation into the OpenCV camera convention.
_DROID_TO_OPENCV = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

# Bridge (WidowX) frame corrections: raw state rotation -> kinematics frame, a fixed
# ee_gripper_link -> gripper_link re-reference, then kinematics -> OpenCV frame.
_BRIDGE_DEFAULT_ROTATION = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
_BRIDGE_TO_OPENCV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
_BRIDGE_TCP_TO_FLANGE = np.array(
    [[1.0, 0.0, 0.0, -0.093575], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)

# Representation per embodiment domain. Domains not listed use the identity representation.
# DROID uses ``joint_pos`` (8D raw joints + gripper) to match its released policy checkpoints;
# RoboMIND-Franka / Bridge stay on the end-effector pose representations.
DOMAIN_REPRESENTATION: dict[str, str] = {
    "droid_lerobot": "joint_pos",
    "robomind-franka": "droid_ee",
    "robomind_franka": "droid_ee",
    "bridge_orig_lerobot": "bridge_ee",
}

# Index of the gripper channel within a domain's raw ``observation.state`` row, used
# when a separate gripper value is not supplied to :func:`represent_state`.
DOMAIN_STATE_GRIPPER_INDEX: dict[str, int] = {
    "droid_lerobot": 6,
    "robomind-franka": 6,
    "robomind_franka": 6,
    "bridge_orig_lerobot": 7,
}

# Index of the gripper channel within a domain's raw ``action`` row. cosmos-framework builds the
# action-chunk gripper from the ACTION column (DROID/Bridge index 6), not the state, so the
# training target's gripper is the commanded gripper for each action step rather than the observed
# proprioceptive gripper. Domains not listed fall back to :data:`DOMAIN_STATE_GRIPPER_INDEX`.
DOMAIN_ACTION_GRIPPER_INDEX: dict[str, int] = {
    "droid_lerobot": 6,
    "robomind-franka": 6,
    "robomind_franka": 6,
    "bridge_orig_lerobot": 6,
}


def domain_representation(domain: str) -> str:
    """Return the action-space representation ("identity", "joint_pos", "droid_ee", "bridge_ee")."""
    return DOMAIN_REPRESENTATION.get(domain, "identity")


# Domains whose raw gripper command is inverted relative to the model's convention
# (cosmos-framework ``_is_gripper_action_flipped``). DROID's LeRobot gripper is flipped.
DOMAIN_GRIPPER_FLIPPED: dict[str, bool] = {
    "droid_lerobot": True,
}


def domain_gripper_flipped(domain: str) -> bool:
    """Whether the domain's raw gripper command must be inverted as ``1 - g``."""
    return DOMAIN_GRIPPER_FLIPPED.get(domain, False)


def flip_gripper_last_channel(action: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``action`` with its final (gripper) channel inverted as ``1 - g``."""
    flipped = action.clone()
    flipped[..., -1] = 1.0 - flipped[..., -1]
    return flipped


def domain_action_gripper_index(domain: str) -> int | None:
    """Return the gripper column index within a domain's raw ``action`` row, or ``None``.

    cosmos-framework reads the action-chunk gripper from the ACTION column; callers use this to
    pass the commanded gripper to :func:`represent_actions` instead of the observed state gripper.
    """
    return DOMAIN_ACTION_GRIPPER_INDEX.get(domain)


def uses_minmax_normalization(domain: str) -> bool:
    """Whether a domain's state/actions are min-max normalized into the model's ``[-1, 1]`` space.

    Pose-representation domains (``droid_ee``, ``bridge_ee``) are already in the checkpoint's
    native space and must not be re-scaled; identity domains rely on per-dataset min-max.
    """
    return domain_representation(domain) == "identity"


# Per-domain action-normalization method, mirroring the cosmos-framework dataset defaults
# (``resolve_action_normalization``). Domains not listed default to ``minmax`` (the identity
# per-dataset scaling). Pose domains keep their cosmos-framework method: DROID trains raw
# (``none``), Bridge / RoboMIND-Franka use ``quantile``. Values map to :mod:`normalization`.
DOMAIN_NORMALIZATION: dict[str, str] = {
    "droid_lerobot": "none",
    "robomind-franka": "quantile",
    "robomind_franka": "quantile",
    "bridge_orig_lerobot": "quantile",
}


def domain_normalization(domain: str) -> str:
    """Return the action-normalization method ("none"/"minmax"/"quantile"/...) for a domain."""
    return DOMAIN_NORMALIZATION.get(domain, "minmax")


# Split-column state layout: some datasets store the robot state as separate LeRobot
# sub-columns rather than a single combined ``observation.state`` column. These are the
# ``(pose, gripper)`` sub-column names, in canonical order, used to reassemble one
# ``[..., pose + gripper]`` state row. DROID's ``joint_pos`` state uses the joint-position
# sub-column; the ee-pose domains use ``cartesian_position``.
DOMAIN_SPLIT_STATE_COLUMNS: dict[str, tuple[str, str]] = {
    "droid_lerobot": ("joint_positions", "gripper_position"),
    "robomind-franka": ("cartesian_position", "gripper_position"),
    "robomind_franka": ("cartesian_position", "gripper_position"),
    "bridge_orig_lerobot": ("cartesian_position", "gripper_position"),
}


def assemble_state_sequence(domain: str, state: object) -> torch.Tensor:
    """Return the raw state as one tensor in canonical ``[pose..., gripper]`` column order.

    Datasets provide the robot state either as a single combined column (already canonical,
    returned unchanged) or as split sub-columns keyed by their LeRobot sub-column name
    (e.g. ``{"cartesian_position": [..., 6], "gripper_position": [...]}``, the original DROID
    layout). The split layout is concatenated back into one ``[..., pose + 1]`` tensor so that
    :func:`represent_state` / :func:`represent_actions` always receive a uniform raw state
    row/sequence regardless of how the dataset stored it.
    """
    if not isinstance(state, Mapping):
        return state  # already a single combined-column tensor
    cols = DOMAIN_SPLIT_STATE_COLUMNS.get(domain)
    if cols is None:
        msg = (
            f"Domain '{domain}' provides split state columns {sorted(state)} but no split-column "
            "layout is registered in DOMAIN_SPLIT_STATE_COLUMNS."
        )
        raise KeyError(msg)
    pose_key, grip_key = cols
    if pose_key not in state or grip_key not in state:
        msg = (
            f"Split state for domain '{domain}' expects sub-columns '{pose_key}' and '{grip_key}', got {sorted(state)}."
        )
        raise KeyError(msg)
    pose = state[pose_key]
    grip = state[grip_key]
    if grip.ndim < pose.ndim:
        grip = grip.unsqueeze(-1)
    return torch.cat([pose.to(grip.dtype), grip], dim=-1)


# ---------------------------------------------------------------------------
# Rotation helpers (mirror cosmos-framework pose_utils.convert_rotation)
# ---------------------------------------------------------------------------
def euler_xyz_to_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    """Euler xyz angles (radians, ``(...,3)``) -> rotation matrices ``(...,3,3)``."""
    flat = np.asarray(euler_xyz, dtype=np.float32).reshape(-1, 3)
    mats = R.from_euler("xyz", flat, degrees=False).as_matrix().astype(np.float32)
    return mats.reshape(*np.shape(euler_xyz)[:-1], 3, 3)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(...,3,3)`` -> 6D ``[R[:,0], R[:,1]]`` ``(...,6)``."""
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.swapaxes(matrix[..., :, :2], -1, -2).reshape(*matrix.shape[:-2], 6)


def _normalize_rotation_matrices(rot_matrices: np.ndarray) -> np.ndarray:
    """Project approximate matrices ``(...,3,3)`` onto ``SO(3)`` via SVD.

    Decoding rot6d from network outputs yields near- but not exactly orthonormal columns; this
    returns the closest proper rotation (determinant ``+1``), mirroring cosmos-framework's
    ``pose_utils._normalize_rotation_matrices`` used by ``pose_rel_to_abs(normalize_rotation=True)``.
    """
    matrices = np.asarray(rot_matrices, dtype=np.float32)
    original_shape = matrices.shape[:-2]
    flat = matrices.reshape(-1, 3, 3)
    u, _, vt = np.linalg.svd(flat)
    normalized = u @ vt
    reflection = np.linalg.det(normalized) < 0  # guard against improper rotations (det -1)
    if np.any(reflection):
        u_reflect = u.copy()
        u_reflect[reflection, :, -1] *= -1
        normalized[reflection] = u_reflect[reflection] @ vt[reflection]
    return normalized.astype(np.float32).reshape(*original_shape, 3, 3)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Inverse of :func:`matrix_to_rot6d`.

    Builds the third column via the cross product, then projects onto ``SO(3)`` with an SVD,
    matching cosmos-framework's ``convert_rotation(..., normalize_matrix=True)`` used when
    decoding network-emitted rot6d back to a valid rotation.
    """
    rot6d = np.asarray(rot6d, dtype=np.float32)
    col0, col1 = rot6d[..., :3], rot6d[..., 3:]
    col2 = np.cross(col0, col1)
    matrices = np.stack((col0, col1, col2), axis=-1).astype(np.float32)
    return _normalize_rotation_matrices(matrices)


def _rel_rot6d_backward(poses: np.ndarray) -> np.ndarray:
    """Absolute poses ``[N+1, 4, 4]`` -> framewise-relative ``[trans(3), rot6d(6)]`` ``[N, 9]``."""
    inv = np.linalg.inv(poses)
    return np.stack([
        np.concatenate([(inv[i] @ poses[i + 1])[:3, 3], matrix_to_rot6d((inv[i] @ poses[i + 1])[:3, :3])])
        for i in range(len(poses) - 1)
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# DROID end-effector representation
# ---------------------------------------------------------------------------
def droid_ee_state_token(cartesian_position: np.ndarray, gripper: float) -> torch.Tensor:
    """Build the DROID ee state token ``[pos(3), rot6d(6), gripper(1)]``.

    ``cartesian_position`` is the raw ``[x, y, z, euler_x, euler_y, euler_z]`` row in the
    robot frame. The rotation is mapped into the OpenCV camera frame and the gripper is
    flipped (``1 - g``), matching the checkpoint's ``use_state`` layout.
    """
    cartesian_position = np.asarray(cartesian_position, dtype=np.float32)
    xyz = cartesian_position[:3]
    rot = euler_xyz_to_matrix(cartesian_position[3:6]) @ _DROID_TO_OPENCV
    rot6d = matrix_to_rot6d(rot)
    grip = np.array([1.0 - float(gripper)], dtype=np.float32)  # DROID gripper is flipped
    return torch.from_numpy(np.concatenate([xyz, rot6d, grip])).float()  # [10]


def droid_ee_relative_actions(cartesian_seq: np.ndarray, gripper_seq: np.ndarray) -> torch.Tensor:
    """Absolute ee trajectory -> DROID relative action chunk ``[N, 10]``.

    ``cartesian_seq`` is ``[N+1, 6]`` absolute ``[xyz, euler_xyz]`` rows and ``gripper_seq`` is the
    matching gripper values. Framewise-relative pose (``delta_T = inv(T_i) @ T_{i+1}``) as
    ``[trans(3), rot6d(6)]`` in the OpenCV frame, with the flipped gripper appended. The last ``N``
    gripper values are used (cosmos-framework's ``gripper[-chunk_length:]``): when a per-action-step
    gripper (length ``N``, e.g. the raw ACTION column) is passed they are used directly; when the
    state gripper (length ``N+1``) is the fallback this selects each transition's destination frame.
    """
    cartesian_seq = np.asarray(cartesian_seq, dtype=np.float32)
    poses = np.tile(np.eye(4, dtype=np.float32), (len(cartesian_seq), 1, 1))
    poses[:, :3, :3] = euler_xyz_to_matrix(cartesian_seq[:, 3:6]) @ _DROID_TO_OPENCV
    poses[:, :3, 3] = cartesian_seq[:, :3]
    rel = _rel_rot6d_backward(poses)  # [N, 9]
    g = np.asarray(gripper_seq, dtype=np.float32).reshape(-1)[-rel.shape[0] :]
    grip = (1.0 - g)[:, None]  # DROID gripper is flipped
    return torch.from_numpy(np.concatenate([rel, grip], axis=-1)).float()  # [N, 10]


def _poses_from_ee_tokens(tokens: np.ndarray) -> np.ndarray:
    """Absolute ee pose tokens ``[N, >=9]`` (``[pos(3), rot6d(6), ...]``) -> poses ``[N, 4, 4]``."""
    tokens = np.asarray(tokens, dtype=np.float32).reshape(-1, tokens.shape[-1])
    poses = np.tile(np.eye(4, dtype=np.float32), (len(tokens), 1, 1))
    poses[:, :3, :3] = rot6d_to_matrix(tokens[:, 3:9])
    poses[:, :3, 3] = tokens[:, :3]
    return poses


def droid_ee_absolute_to_relative(
    current_state_token: np.ndarray | torch.Tensor,
    action_tokens: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Absolute DROID ee pose tokens -> framewise-relative action chunk ``[N, 10]``.

    The Cosmos 3 head emits absolute ee pose tokens ``[pos(3), rot6d(6), gripper(1)]`` in the
    same space as :func:`droid_ee_state_token`. This anchors them on the current-state token
    ``T_0`` and applies ``delta_T = inv(T_i) @ T_{i+1}`` over
    ``[current_state, pred_1, ..., pred_N]`` (the same conversion as :func:`_rel_rot6d_backward`
    used to build training targets), yielding the ``[trans(3), rot6d(6)]`` relative pose that
    matches :func:`droid_ee_relative_actions`. The raw gripper logit is thresholded to a hard
    ``0/1`` command, already in the flipped ``1 - g`` DROID convention of the tokens.
    """
    state = np.asarray(_as_numpy(current_state_token), dtype=np.float32).reshape(-1)
    acts = np.asarray(_as_numpy(action_tokens), dtype=np.float32)
    if acts.ndim == 1:
        acts = acts[None]
    poses = _poses_from_ee_tokens(np.concatenate([state[None, :9], acts[:, :9]], axis=0))
    rel = _rel_rot6d_backward(poses)  # [N, 9]
    grip = (acts[:, 9] > 0.0).astype(np.float32)[:, None]  # sigmoid>0.5 threshold; flipped 1-g convention
    return torch.from_numpy(np.concatenate([rel, grip], axis=-1)).float()  # [N, 10]


# ---------------------------------------------------------------------------
# Bridge (WidowX) end-effector representation
# ---------------------------------------------------------------------------
def _bridge_poses(state_seq: np.ndarray) -> np.ndarray:
    """Raw Bridge state rows ``[N, >=6]`` (``[xyz, euler_xyz, ...]``) -> corrected poses ``[N, 4, 4]``."""
    state_seq = np.asarray(state_seq, dtype=np.float32)
    poses = np.tile(np.eye(4, dtype=np.float32), (len(state_seq), 1, 1))
    poses[:, :3, :3] = euler_xyz_to_matrix(state_seq[:, 3:6])
    poses[:, :3, 3] = state_seq[:, :3]
    poses[:, :3, :3] = poses[:, :3, :3] @ _BRIDGE_DEFAULT_ROTATION
    poses = poses @ _BRIDGE_TCP_TO_FLANGE
    poses[:, :3, :3] = poses[:, :3, :3] @ _BRIDGE_TO_OPENCV
    return poses


def bridge_ee_state_token(state_row: np.ndarray, gripper: float) -> torch.Tensor:
    """Build the Bridge ee state token ``[pos(3), rot6d(6), gripper(1)]`` (raw, unflipped gripper)."""
    poses = _bridge_poses(np.asarray(state_row, dtype=np.float32)[None])
    xyz = poses[0, :3, 3]
    rot6d = matrix_to_rot6d(poses[0, :3, :3])
    grip = np.array([float(gripper)], dtype=np.float32)
    return torch.from_numpy(np.concatenate([xyz, rot6d, grip])).float()  # [10]


def bridge_ee_relative_actions(state_seq: np.ndarray, gripper_seq: np.ndarray) -> torch.Tensor:
    """Absolute Bridge state trajectory -> relative action chunk ``[N, 10]``.

    Uses the last ``N`` gripper values (cosmos-framework's ``gripper[-chunk_length:]``), raw and
    unflipped (the WidowX command). A per-action-step gripper (length ``N``, e.g. the raw ACTION
    column) is used directly; a fallback state gripper (length ``N+1``) contributes its last ``N``.
    """
    poses = _bridge_poses(np.asarray(state_seq, dtype=np.float32))
    rel = _rel_rot6d_backward(poses)  # [N, 9]
    g = np.asarray(gripper_seq, dtype=np.float32).reshape(-1)[-rel.shape[0] :]
    grip = g[:, None]
    return torch.from_numpy(np.concatenate([rel, grip], axis=-1)).float()  # [N, 10]


def bridge_ee_absolute_to_relative(
    current_state_token: np.ndarray | torch.Tensor,
    action_tokens: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Absolute Bridge ee pose tokens -> framewise-relative action chunk ``[N, 10]``.

    Mirror of :func:`droid_ee_absolute_to_relative` for the Bridge/WidowX convention: anchor the
    absolute pose tokens ``[pos(3), rot6d(6), gripper(1)]`` on the current-state token ``T_0`` and
    apply ``delta_T = inv(T_i) @ T_{i+1}`` to match :func:`bridge_ee_relative_actions`. Bridge uses
    the raw (unflipped) source-frame gripper, so the ``0/1`` command comes from frame ``i`` (the
    "from" frame) and no ``1 - g`` flip is applied.
    """
    state = np.asarray(_as_numpy(current_state_token), dtype=np.float32).reshape(-1)
    acts = np.asarray(_as_numpy(action_tokens), dtype=np.float32)
    if acts.ndim == 1:
        acts = acts[None]
    tokens = np.concatenate([state[None, :10], acts[:, :10]], axis=0)  # [N+1, 10]
    poses = _poses_from_ee_tokens(tokens[:, :9])
    rel = _rel_rot6d_backward(poses)  # [N, 9]
    grip = (tokens[: rel.shape[0], 9] > 0.0).astype(np.float32)[:, None]  # source-frame, unflipped 0/1
    return torch.from_numpy(np.concatenate([rel, grip], axis=-1)).float()  # [N, 10]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def _as_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        # numpy has no bfloat16; upcast to float32 before converting.
        return x.detach().to(torch.float32).cpu().numpy() if x.dtype == torch.bfloat16 else x.detach().cpu().numpy()
    return np.asarray(x)


def represent_state(
    domain: str,
    state_row: np.ndarray | torch.Tensor,
    gripper: float | None = None,
) -> torch.Tensor:
    """Raw proprioceptive state row -> the domain's action-space state token.

    ``identity`` returns the raw state unchanged; ``droid_ee`` / ``bridge_ee`` build the
    10-D ``[pos, rot6d, gripper]`` ee token. When ``gripper`` is not given it is read from
    the domain's configured gripper index within ``state_row``.
    """
    rep = domain_representation(domain)
    row = _as_numpy(state_row).astype(np.float32)
    if rep == "droid_ee":
        idx = DOMAIN_STATE_GRIPPER_INDEX.get(domain)
        g = (
            float(gripper)
            if gripper is not None
            else (float(row[idx]) if idx is not None and row.shape[-1] > idx else 0.0)
        )
        return droid_ee_state_token(row[:6], g)
    if rep == "bridge_ee":
        idx = DOMAIN_STATE_GRIPPER_INDEX.get(domain)
        g = (
            float(gripper)
            if gripper is not None
            else (float(row[idx]) if idx is not None and row.shape[-1] > idx else 0.0)
        )
        return bridge_ee_state_token(row, g)
    return torch.as_tensor(row).float()


def represent_actions(
    domain: str,
    state_seq: np.ndarray | torch.Tensor,
    action_gripper_seq: np.ndarray | torch.Tensor | None = None,
    raw_action_seq: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Raw state/action sequences -> the domain's action-space chunk.

    ``identity`` returns ``raw_action_seq`` unchanged; the ee representations build
    framewise-relative rot6d chunks from the absolute ``state_seq`` (``[N+1, >=6]``) and the
    matching gripper. The gripper is the commanded ``action_gripper_seq`` when provided (cosmos
    reads it from the raw ACTION column); otherwise it falls back to the observed state gripper at
    the domain's :data:`DOMAIN_STATE_GRIPPER_INDEX` within ``state_seq``.
    """
    rep = domain_representation(domain)
    if rep == "droid_ee":
        seq = _as_numpy(state_seq).astype(np.float32)
        idx = DOMAIN_STATE_GRIPPER_INDEX.get(domain)
        grip = (
            _as_numpy(action_gripper_seq)
            if action_gripper_seq is not None
            else (seq[:, idx] if idx is not None else np.zeros(len(seq)))
        )
        return droid_ee_relative_actions(seq[:, :6], grip)
    if rep == "bridge_ee":
        seq = _as_numpy(state_seq).astype(np.float32)
        idx = DOMAIN_STATE_GRIPPER_INDEX.get(domain)
        grip = (
            _as_numpy(action_gripper_seq)
            if action_gripper_seq is not None
            else (seq[:, idx] if idx is not None else np.zeros(len(seq)))
        )
        return bridge_ee_relative_actions(seq, grip)
    if raw_action_seq is None:
        msg = f"identity domain '{domain}' requires raw_action_seq for represent_actions."
        raise ValueError(msg)
    return torch.as_tensor(_as_numpy(raw_action_seq)).float()
