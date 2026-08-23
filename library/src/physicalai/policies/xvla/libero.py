# Copyright 2025 2toINF (https://github.com/2toINF) and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bridges a bimanual ``ee6d`` XVLA checkpoint to :class:`~physicalai.gyms.LiberoGym`.

``LiberoGym`` exposes a single-arm environment: an 8-dim proprioceptive state
(end-effector position + axis-angle orientation + a 2-dim gripper) and a 7-dim action
(end-effector delta + axis-angle delta + a 1-dim gripper). Published XVLA checkpoints
trained with ``action_mode="ee6d"`` instead predict a fixed 20-dim *bimanual*
end-effector layout -- two 10-dim arm slots of ``[xyz(3), 6D rotation(6), gripper(1)]``.
LIBERO's single arm occupies only the first slot; the second is left at zero, matching how
the upstream LeRobot LIBERO dataset populates this checkpoint's training data.

:class:`XVLALiberoPolicy` bridges the two: it maps ``LiberoGym``'s state into the first
arm's slot before calling the inherited XVLA forward pass, and maps the predicted bimanual
chunk back down to LIBERO's 7-dim control vector, converting the 6D rotation to axis-angle
and the sigmoid gripper probability to LIBERO's ``{-1, +1}`` signal -- the same conversion
upstream's ``XVLARotation6DToAxisAngleProcessorStep`` performs.

This bridging is inference-only (see :meth:`XVLALiberoPolicy.predict_action_chunk`); it is
not meant for training on LIBERO-shaped data, where the dataset's raw 7-dim actions would
not match the model's 20-dim training target. Use ``action_mode="auto"`` instead when
training or finetuning from scratch on a single-arm dataset -- see the "Cross-embodiment:
domains and action spaces" section of the XVLA docs for the tradeoff between the two.

Two details are specific to the published ``lerobot/xvla-libero`` checkpoint, confirmed
against the official evaluation recipe (``lerobot-eval --env.control_mode=absolute
--policy.path=lerobot/xvla-libero``, https://github.com/huggingface/lerobot, docs/source/xvla.mdx)
and LeRobot's ``LiberoProcessorStep``/``LiberoEnv`` source:

- Its action data records an **absolute** target end-effector pose per step, not a delta.
  Build the environment with ``control_mode="absolute"`` (``LiberoGym``'s default is
  ``"relative"``, i.e. deltas) so its controller interprets the bridged action the same way
  the checkpoint was trained to produce it; feeding an absolute-scale pose into a delta-mode
  controller saturates every joint on the first step and the episode never recovers --
  scoring zero without raising, so the mistake is silent. Pass ``control_mode="absolute"``
  to :class:`~physicalai.benchmark.gyms.LiberoBenchmark`, to
  :func:`~physicalai.gyms.create_libero_gyms`, or to :class:`~physicalai.gyms.LiberoGym`
  directly. Whether a different ``ee6d`` checkpoint uses absolute or delta actions is a
  property of how it was trained, not of the ``ee6d`` layout itself -- check before
  assuming either.
- Its domain id is **3**, published in the checkpoint's ``policy_preprocessor.json`` and
  auto-detected by :func:`~physicalai.policies.xvla.pretrained_utils.extract_domain_id`
  when loading via ``pretrained_name_or_path`` (see :class:`~physicalai.policies.xvla.XVLA`).

``LiberoGym`` flips every camera 180 degrees to match the LeRobot dataset convention, but
LeRobot's own ``LiberoEnv`` applies no flip at all and X-VLA's ``LiberoProcessorStep``
flips only the primary ("image"/agentview) view -- the wrist camera ("image2") is left as
rendered. :func:`_undo_libero_wrist_camera_flip` corrects for this so the wrist view
matches training; it runs automatically inside :meth:`XVLALiberoPolicy.predict_action_chunk`.

Three further mismatches were found by running one identical LIBERO observation through both
this bridging and LeRobot's own pipeline (``LiberoProcessorStep`` + the published
``policy_preprocessor.json`` + ``XVLAPolicy._build_model_inputs``) and diffing the resulting
model inputs tensor by tensor:

- The proprioceptive **rotation frame**. ``LiberoGym``'s state reports the ``right_hand``
  body orientation, while upstream builds its 6D rotation from the OSC controller's grip-site
  frame -- a fixed quarter turn apart. See :data:`GRIP_SITE_FROM_HAND_BODY`; this was the one
  mismatch large enough to matter, and correcting it is what moved LIBERO-10 off 0%.
- LeRobot's eval config renders LIBERO at **360x360**, where ``LiberoGym`` defaults to
  256x256. Both are resized to the model's 224x224 anyway, and the difference was measured
  as immaterial (12/15 vs 13/15 episodes, below), so ``LiberoGym``'s default is fine; pass
  ``observation_height=360, observation_width=360`` only if you want byte-faithful parity
  with LeRobot. The controller frequency needs nothing at all: LeRobot drives LIBERO at
  robosuite's 20 Hz default, which ``LiberoGym`` inherits from LIBERO's own
  ``OffScreenRenderEnv``. LeRobot's ``LiberoEnv`` config does carry an ``fps`` field set to
  30, but never feeds it to the simulator, so it is not a control rate -- do not read it as
  one.
- The checkpoint's own ``config.json`` already declares an ``empty_camera_0`` feature baked
  in from an earlier validation pass, so re-running upstream's ``validate_features()``
  formula on load (``max(published num_image_views, declared views + empty_cameras)``)
  resolves to **4** camera slots, not the 3 a naive reading of ``empty_cameras=1`` would
  suggest; ``pretrained_utils.load_config`` reproduces this exactly.

With these plus the domain id and tokenizer length above, ``XVLALiberoPolicy``'s model
inputs match LeRobot's to within float noise (images to 1e-6, proprioception to 1e-4 -- the
latter only because upstream reads a controller matrix that lags the returned observation by
the last few simulation substeps), and the bridged 7-dim actions agree to ~2e-3.

Which of those settings actually carry the success rate was then measured by ablation on
LIBERO-10 tasks 0-4, one episode each, changing one thing at a time from the recipe in the
example below:

===========================================  =========
Configuration                                Successes
===========================================  =========
recipe as below                              5/5
``control_mode="relative"``                  0/5
no wrist-camera flip correction              2/5
256x256 rendering                            4/5
``max_steps=520`` instead of 800             5/5
===========================================  =========

So ``control_mode="absolute"`` and the wrist-camera correction are load-bearing, while the
render resolution and a longer step budget are not: successful episodes finish in about 250
steps, well inside LIBERO-10's own 520 limit. Repeating the resolution arm with three
episodes per task put 360x360 at 13/15 and 256x256 at 12/15 -- a single episode apart, i.e.
indistinguishable. Those 15-episode numbers are also the more honest headline: the 5/5 and
10/10 figures come from a single episode per task, and flow-matching inference draws fresh
noise every call, so treat the ablation as evidence about which settings matter rather than
as a published success rate.

Example:
    >>> from physicalai.policies.xvla.libero import XVLALiberoPolicy
    >>> policy = XVLALiberoPolicy(pretrained_name_or_path="lerobot/xvla-libero")  # doctest: +SKIP
    >>> policy.eval()  # doctest: +SKIP
    >>> action = policy.select_action(observation)  # doctest: +SKIP
    >>> action.shape  # doctest: +SKIP
    torch.Size([1, 7])
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from .policy import XVLA

if TYPE_CHECKING:
    from physicalai.data import Observation

LIBERO_STATE_DIM = 8
"""Width of ``LiberoGym``'s proprioceptive state: eef position(3) + axis-angle(3) + gripper(2)."""

LIBERO_ACTION_DIM = 7
"""Width of ``LiberoGym``'s action: eef delta(3) + axis-angle delta(3) + gripper(1)."""

LIBERO_ARM_WIDTH = 10
"""Width of one arm's slot in XVLA's bimanual ee6d layout: xyz(3) + 6D rotation(6) + gripper(1)."""

_GRIPPER_CLOSE_THRESHOLD = 0.5
_WRIST_CAMERA_KEY = "image2"

GRIP_SITE_FROM_HAND_BODY = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
"""Rotation taking robosuite's ``right_hand`` body frame to the gripper's grip-site frame.

``LiberoGym``'s state reports orientation as ``quat2axisangle(robot0_eef_quat)``, and
robosuite's ``eef_quat`` sensor reads the **body** quaternion of ``right_hand`` -- while its
``eef_pos`` sensor, the OSC controller (``controller.ee_ori_mat``, the frame every absolute
orientation target is resolved against) and therefore X-VLA's own proprioception all use the
gripper's **grip site**. The two frames differ by a fixed -90 degree rotation about z, which
the Panda gripper's XML declares directly: ``<body name="right_gripper" ... quat="0.707107 0
0 -0.707107">``, with the ``eef`` body and ``grip_site`` under it both left at identity.

Without this correction the proprioceptive rotation handed to the model is a quarter turn off
about the tool axis at every step -- an orientation the checkpoint never saw in training,
which alone drove LIBERO-10 success to zero.
"""


def _undo_libero_wrist_camera_flip(batch: Observation) -> Observation:
    """Undo ``LiberoGym``'s 180-degree flip on the wrist camera only.

    ``LiberoGym`` flips every camera 180 degrees "to match the LeRobot convention". But
    LeRobot's own ``LiberoEnv`` applies **no** flip at all, and the flip X-VLA actually
    trains against comes entirely from its ``LiberoProcessorStep``, which flips only the
    primary ``"image"`` (agentview) view -- the wrist camera (``"image2"``) is left exactly
    as the simulator renders it. ``LiberoGym``'s primary camera therefore already matches
    what X-VLA expects, but its wrist camera is upside down relative to training unless
    this flip is undone.

    Args:
        batch: Observation using ``LiberoGym``'s flattened camera keys.

    Returns:
        A shallow copy of ``batch`` with the wrist camera flipped back; other fields
        (including the primary camera) pass through unchanged.
    """
    if not isinstance(batch.images, dict) or _WRIST_CAMERA_KEY not in batch.images:
        return batch
    images = dict(batch.images)
    images[_WRIST_CAMERA_KEY] = torch.flip(images[_WRIST_CAMERA_KEY], dims=[-2, -1])
    return replace(batch, images=images)


# --------------------------------------------------------------------------------------- #
# Rotation conversions (numpy, ported/adapted from the upstream xvla/utils.py)             #
# --------------------------------------------------------------------------------------- #
def _axis_angle_to_mat(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: an axis-angle rotation to a rotation matrix.

    Args:
        axis_angle: Rotation vector of shape ``(3,)``; its norm is the angle in radians.

    Returns:
        Rotation matrix of shape ``(3, 3)``.
    """
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:  # noqa: PLR2004
        return np.eye(3, dtype=np.float32)
    kx, ky, kz = (axis_angle / theta).astype(np.float32)
    skew = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def _mat_to_rotation_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """A rotation matrix's first two columns, XVLA's 6D rotation representation.

    Args:
        rotation_matrix: Rotation matrix of shape ``(3, 3)``.

    Returns:
        The 6D rotation representation, shape ``(6,)``.
    """
    return np.concatenate([rotation_matrix[:3, 0], rotation_matrix[:3, 1]])


def _mat2quat(rmat: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion.

    Ported unchanged from the upstream ``xvla/utils.py``.

    Args:
        rmat: Rotation matrix of shape ``(3, 3)`` or larger, only whose top-left 3x3 block
            is used.

    Returns:
        Quaternion ``(x, y, z, w)`` of shape ``(4,)``.
    """
    mat = np.asarray(rmat).astype(np.float32)[:3, :3]
    m00, m01, m02 = mat[0]
    m10, m11, m12 = mat[1]
    m20, m21, m22 = mat[2]
    k = np.array(
        [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ],
        dtype=np.float32,
    )
    k /= 3.0
    w, v = np.linalg.eigh(k)
    inds = np.array([3, 0, 1, 2])
    q1 = v[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to axis-angle: a unit direction scaled by its angle in radians.

    Ported unchanged from the upstream ``xvla/utils.py``.

    Args:
        quat: Quaternion ``(x, y, z, w)`` of shape ``(4,)``.

    Returns:
        Axis-angle rotation of shape ``(3,)``.
    """
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _rotation_6d_to_axis_angle(rotation_6d: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_mat_to_rotation_6d`: XVLA's 6D rotation to axis-angle.

    Adapted from the upstream ``rotate6d_to_axis_angle`` for a single (unbatched) vector.

    Args:
        rotation_6d: The 6D rotation representation, shape ``(6,)``.

    Returns:
        Axis-angle rotation of shape ``(3,)``.
    """
    a1, a2 = rotation_6d[0:3], rotation_6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-6)
    b2_orth = a2 - np.dot(b1, a2) * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth) + 1e-6)
    b3 = np.cross(b1, b2)
    rotation_matrix = np.stack([b1, b2, b3], axis=-1)
    return _quat2axisangle(_mat2quat(rotation_matrix))


# --------------------------------------------------------------------------------------- #
# Torch-facing batch conversions                                                          #
# --------------------------------------------------------------------------------------- #
def libero_state_to_ee6d_proprio(state: torch.Tensor) -> torch.Tensor:
    """Map ``LiberoGym``'s ``(..., 8)`` state to XVLA's ``(..., 20)`` bimanual ee6d proprio.

    LIBERO's single arm occupies the first :data:`LIBERO_ARM_WIDTH` (10) channels; the
    second arm's slot is zero. The gripper channel is zero regardless of LIBERO's own
    gripper reading, matching upstream's ``LiberoProcessorStep`` -- :class:`EE6DActionSpace`
    masks it out before the transformer sees it anyway (see
    :meth:`~physicalai.policies.xvla.action_hub.EE6DActionSpace.preprocess`), so any value
    placed here is discarded.

    The orientation is rotated by :data:`GRIP_SITE_FROM_HAND_BODY` on the way in: LIBERO's
    state carries the ``right_hand`` *body* orientation, while the 6D rotation upstream
    builds -- and the model was trained on -- is the gripper's *grip site*, a fixed quarter
    turn away.

    Args:
        state: LIBERO proprioceptive state, ``[eef position(3), axis-angle(3), gripper(2)]``.

    Returns:
        Proprioception of shape ``(..., 20)``.

    Raises:
        ValueError: If the last dimension is not 8.
    """
    if state.shape[-1] != LIBERO_STATE_DIM:
        msg = f"Expected LIBERO's {LIBERO_STATE_DIM}-dim state (pos + axis-angle + gripper), got {state.shape[-1]}"
        raise ValueError(msg)

    batch_shape = state.shape[:-1]
    device, dtype = state.device, state.dtype

    positions = state[..., :3]
    axis_angles = state[..., 3:6].reshape(-1, 3).detach().cpu().numpy()
    rotation_6d = torch.stack([
        torch.from_numpy(_mat_to_rotation_6d(_axis_angle_to_mat(aa) @ GRIP_SITE_FROM_HAND_BODY)) for aa in axis_angles
    ]).to(
        device=device,
        dtype=dtype,
    )
    rotation_6d = rotation_6d.reshape(*batch_shape, 6)

    gripper = torch.zeros(*batch_shape, 1, device=device, dtype=dtype)
    left_arm = torch.cat([positions, rotation_6d, gripper], dim=-1)
    return torch.cat([left_arm, torch.zeros_like(left_arm)], dim=-1)


def ee6d_action_to_libero(action: torch.Tensor) -> torch.Tensor:
    """Map XVLA's bimanual ee6d action to ``LiberoGym``'s ``(..., 7)`` control vector.

    Only the first arm's slot is used: its 6D rotation is converted to axis-angle and its
    sigmoid gripper probability to LIBERO's ``{-1, +1}`` signal (a value above 0.5 closes
    the gripper), matching upstream's ``XVLARotation6DToAxisAngleProcessorStep``.

    Args:
        action: Postprocessed ee6d action of shape ``(..., D)`` with ``D >= 10``.

    Returns:
        LIBERO action of shape ``(..., 7)``: ``[eef delta(3), axis-angle delta(3), gripper(1)]``.

    Raises:
        ValueError: If the last dimension is narrower than :data:`LIBERO_ARM_WIDTH`.
    """
    if action.shape[-1] < LIBERO_ARM_WIDTH:
        msg = f"Expected at least {LIBERO_ARM_WIDTH} action channels, got {action.shape[-1]}"
        raise ValueError(msg)

    batch_shape = action.shape[:-1]
    device, dtype = action.device, action.dtype

    positions = action[..., :3]
    rotation_6d = action[..., 3:9].reshape(-1, 6).detach().cpu().numpy()
    axis_angles = torch.stack([torch.from_numpy(_rotation_6d_to_axis_angle(r6d)) for r6d in rotation_6d]).to(
        device=device,
        dtype=dtype,
    )
    axis_angles = axis_angles.reshape(*batch_shape, 3)

    gripper = action[..., 9:10]
    gripper_signal = torch.where(gripper > _GRIPPER_CLOSE_THRESHOLD, 1.0, -1.0).to(dtype=dtype)
    return torch.cat([positions, axis_angles, gripper_signal], dim=-1)


class XVLALiberoPolicy(XVLA):
    """Adapts a bimanual ``ee6d`` XVLA checkpoint to :class:`~physicalai.gyms.LiberoGym`.

    See the module docstring for the conversion this performs. Behaves exactly like
    :class:`~physicalai.policies.xvla.XVLA` otherwise -- same constructor, same weights,
    same training path -- only the inference-time observation/action boundary differs.

    Example:
        >>> from physicalai.benchmark.gyms import LiberoBenchmark  # doctest: +SKIP
        >>> policy = XVLALiberoPolicy(pretrained_name_or_path="lerobot/xvla-libero")  # doctest: +SKIP
        >>> policy.eval()  # doctest: +SKIP
        >>> # control_mode="absolute" is required: this checkpoint predicts absolute target
        >>> # poses, not per-step deltas. Leaving the default scores zero silently.
        >>> benchmark = LiberoBenchmark(  # doctest: +SKIP
        ...     task_suite="libero_10", num_episodes=1, control_mode="absolute"
        ... )
        >>> results = benchmark.evaluate(policy)  # doctest: +SKIP
    """

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict one action chunk, bridged through LIBERO's single-arm interface.

        Args:
            batch: Observation carrying LIBERO's 8-dim proprioceptive state.

        Returns:
            Actions of shape ``[B, chunk_size, 7]`` in LIBERO's control convention.
        """
        batch = _undo_libero_wrist_camera_flip(batch)
        ee6d_actions = super().predict_action_chunk(self._to_ee6d_observation(batch))
        return ee6d_action_to_libero(ee6d_actions)

    @staticmethod
    def _to_ee6d_observation(batch: Observation) -> Observation:
        """Replace LIBERO's proprioceptive state with XVLA's bimanual ee6d proprio.

        Args:
            batch: Observation carrying LIBERO's 8-dim state (or none, e.g. for a
                ``use_proprio=False`` policy).

        Returns:
            A shallow copy of ``batch`` with ``state`` replaced; other fields (images, task)
            pass through unchanged.

        Raises:
            TypeError: If ``batch.state`` is present but not a plain tensor.
        """
        if batch.state is None:
            return batch
        if not isinstance(batch.state, torch.Tensor):
            msg = f"Expected a plain state tensor from LiberoGym, got {type(batch.state)}"
            raise TypeError(msg)
        return replace(batch, state=libero_state_to_ee6d_proprio(batch.state))


__all__ = [
    "GRIP_SITE_FROM_HAND_BODY",
    "LIBERO_ACTION_DIM",
    "LIBERO_ARM_WIDTH",
    "LIBERO_STATE_DIM",
    "XVLALiberoPolicy",
    "ee6d_action_to_libero",
    "libero_state_to_ee6d_proprio",
]
