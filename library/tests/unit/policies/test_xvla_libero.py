# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XVLA-to-LiberoGym bridging in ``physicalai.policies.xvla.libero``.

Fast and self-contained, matching ``test_xvla.py``: a miniature Florence-2 backbone and a
stubbed BART tokenizer let the whole pipeline run without any download.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from physicalai.data import Observation
from physicalai.policies.xvla.libero import (
    LIBERO_ACTION_DIM,
    LIBERO_ARM_WIDTH,
    LIBERO_STATE_DIM,
    XVLALiberoPolicy,
    ee6d_action_to_libero,
    libero_state_to_ee6d_proprio,
)

pytest.importorskip("transformers")

IMAGE_SIZE = 64
CHUNK_SIZE = 4
BATCH_SIZE = 2

TINY_FLORENCE: dict[str, Any] = {
    "vision_config": {
        "model_type": "florence_vision",
        "depths": [1, 1, 1, 1],
        "embed_dim": [8, 16, 24, 32],
        "num_heads": [1, 1, 1, 2],
        "num_groups": [1, 1, 1, 2],
        "window_size": 2,
        "projection_dim": 32,
        "max_temporal_embeddings": 4,
        "max_position_embeddings": 16,
    },
    "text_config": {
        "model_type": "bart",
        "d_model": 32,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": 32,
        "decoder_ffn_dim": 32,
        "vocab_size": 99,
        "max_position_embeddings": 64,
    },
}

VOCAB_SIZE = TINY_FLORENCE["text_config"]["vocab_size"]

DATASET_STATS: dict[str, dict[str, Any]] = {
    "observation.state": {
        "name": "observation.state",
        "type": "STATE",
        "shape": (LIBERO_STATE_DIM,),
    },
    "action": {
        "name": "action",
        "type": "ACTION",
        "shape": (20,),
    },
    "observation.images.image": {
        "name": "observation.images.image",
        "type": "VISUAL",
        "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
    },
    "observation.images.image2": {
        "name": "observation.images.image2",
        "type": "VISUAL",
        "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
    },
}


class FakeTokenizer:
    """Deterministic stand-in for Florence-2's BART tokenizer."""

    def __call__(self, prompts: list[str], max_length: int = 8, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tokenize by hashing each prompt into a fixed-length id sequence.

        Returns:
            Dict with ``input_ids`` of shape ``[len(prompts), max_length]``.
        """
        del kwargs
        ids = torch.tensor(
            [[(hash(prompt) + i) % VOCAB_SIZE for i in range(max_length)] for prompt in prompts],
            dtype=torch.long,
        )
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def build_policy(**overrides: Any) -> XVLALiberoPolicy:
    """Build a miniature :class:`XVLALiberoPolicy` matching a bimanual ``ee6d`` checkpoint.

    Args:
        **overrides: Config overrides.

    Returns:
        A ready-to-run policy.
    """
    kwargs: dict[str, Any] = {
        "florence_config": TINY_FLORENCE,
        "tokenizer_max_length": 8,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": CHUNK_SIZE,
        "hidden_size": 32,
        "depth": 1,
        "num_heads": 2,
        "num_domains": 3,
        "len_soft_prompts": 2,
        "dim_time": 8,
        "max_len_seq": 96,
        "max_state_dim": 20,
        "max_action_dim": 20,
        "action_mode": "ee6d",
        "num_denoising_steps": 2,
    }
    kwargs.update(overrides)
    policy = XVLALiberoPolicy(**kwargs)
    policy._initialize_model(DATASET_STATS)  # noqa: SLF001
    policy._preprocessor._tokenizer = FakeTokenizer()  # noqa: SLF001
    return policy


def make_libero_observation(batch_size: int = BATCH_SIZE) -> Observation:
    """Build a synthetic observation matching ``LiberoGym``'s output shapes.

    Args:
        batch_size: Number of samples.

    Returns:
        The observation batch.
    """
    return Observation(
        state=torch.randn(batch_size, LIBERO_STATE_DIM),
        task=["pick up the alphabet soup and place it in the basket"] * batch_size,
        images={
            "image": torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE),
            "image2": torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE),
        },
    )


# ============================================================================ #
# Rotation conversions                                                         #
# ============================================================================ #


class TestRotationConversions:
    """Correctness of the axis-angle <-> 6D-rotation conversions."""

    def test_round_trip_recovers_the_original_rotation(self) -> None:
        """Encoding then decoding a random rotation reconstructs the same matrix."""
        from physicalai.policies.xvla.libero import _axis_angle_to_mat, _mat_to_rotation_6d, _rotation_6d_to_axis_angle  # noqa: PLC0415

        rng = np.random.default_rng(0)
        max_error = 0.0
        for _ in range(200):
            axis_angle = rng.normal(size=3) * rng.uniform(0.01, 3.0)
            mat = _axis_angle_to_mat(axis_angle)
            recovered = _axis_angle_to_mat(_rotation_6d_to_axis_angle(_mat_to_rotation_6d(mat)))
            max_error = max(max_error, float(np.abs(mat - recovered).max()))
        assert max_error < 1e-4

    def test_matches_scipy_rodrigues(self) -> None:
        """The Rodrigues formula matches scipy's reference implementation."""
        scipy_spatial = pytest.importorskip("scipy.spatial.transform")
        from physicalai.policies.xvla.libero import _axis_angle_to_mat  # noqa: PLC0415

        rng = np.random.default_rng(1)
        for _ in range(100):
            axis_angle = rng.normal(size=3) * rng.uniform(0.01, 3.0)
            mine = _axis_angle_to_mat(axis_angle)
            reference = scipy_spatial.Rotation.from_rotvec(axis_angle).as_matrix()
            assert np.abs(mine - reference).max() < 1e-5

    def test_identity_rotation(self) -> None:
        """A near-zero axis-angle vector decodes back to near-zero."""
        from physicalai.policies.xvla.libero import _axis_angle_to_mat, _mat_to_rotation_6d, _rotation_6d_to_axis_angle  # noqa: PLC0415

        identity_6d = _mat_to_rotation_6d(_axis_angle_to_mat(np.zeros(3)))
        axis_angle = _rotation_6d_to_axis_angle(identity_6d)
        assert np.abs(axis_angle).max() < 1e-6


# ============================================================================ #
# Batch conversions                                                            #
# ============================================================================ #


class TestWristCameraFlip:
    """``LiberoGym`` flips every camera; only the wrist one needs correcting back."""

    def test_wrist_camera_is_flipped_back(self) -> None:
        """The wrist camera ("image2") is flipped 180 degrees relative to the input."""
        from physicalai.policies.xvla.libero import _undo_libero_wrist_camera_flip  # noqa: PLC0415

        primary = torch.rand(1, 3, 8, 8)
        wrist = torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
        batch = Observation(images={"image": primary, "image2": wrist})

        fixed = _undo_libero_wrist_camera_flip(batch)

        torch.testing.assert_close(fixed.images["image"], primary)
        torch.testing.assert_close(fixed.images["image2"], torch.flip(wrist, dims=[-2, -1]))
        # Flipping twice recovers the original: LiberoGym's flip and this undo cancel out.
        assert not torch.allclose(fixed.images["image2"], wrist)

    def test_no_op_without_a_wrist_camera(self) -> None:
        """A batch without an "image2" key passes through unchanged."""
        from physicalai.policies.xvla.libero import _undo_libero_wrist_camera_flip  # noqa: PLC0415

        batch = Observation(images={"image": torch.rand(1, 3, 8, 8)})
        assert _undo_libero_wrist_camera_flip(batch) is batch

    def test_no_op_without_images(self) -> None:
        """A batch without images (e.g. mid-pipeline) passes through unchanged."""
        from physicalai.policies.xvla.libero import _undo_libero_wrist_camera_flip  # noqa: PLC0415

        batch = Observation(state=torch.zeros(1, LIBERO_STATE_DIM))
        assert _undo_libero_wrist_camera_flip(batch) is batch


class TestLiberoConversions:
    """Shape and semantics of the proprio/action bridging functions."""

    def test_proprio_shape_and_arm_slots(self) -> None:
        """LIBERO's state fills only the first arm's slot; the second is zero."""
        state = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04]])
        proprio = libero_state_to_ee6d_proprio(state)
        assert proprio.shape == (1, 20)
        torch.testing.assert_close(proprio[0, :3], state[0, :3])
        assert torch.count_nonzero(proprio[0, LIBERO_ARM_WIDTH:]) == 0

    def test_proprio_gripper_channel_is_a_placeholder(self) -> None:
        """The proprio gripper channel is zeroed regardless of LIBERO's own reading.

        ``EE6DActionSpace.preprocess`` masks it out before the transformer sees it anyway.
        """
        state = torch.zeros(1, LIBERO_STATE_DIM)
        state[0, 6:8] = torch.tensor([0.04, -0.04])
        proprio = libero_state_to_ee6d_proprio(state)
        assert proprio[0, 9].item() == 0.0

    def test_proprio_rejects_wrong_width(self) -> None:
        """A non-LIBERO state width is reported rather than silently misinterpreted."""
        with pytest.raises(ValueError, match="Expected LIBERO's"):
            libero_state_to_ee6d_proprio(torch.zeros(1, 7))

    def test_zero_orientation_encodes_the_grip_site_frame(self) -> None:
        """A LIBERO state with zero orientation encodes the hand-to-grip-site rotation.

        Not the identity: LIBERO reports the ``right_hand`` body orientation while XVLA was
        trained on the grip site's, a quarter turn away (:data:`GRIP_SITE_FROM_HAND_BODY`).
        """
        state = torch.zeros(1, LIBERO_STATE_DIM)
        proprio = libero_state_to_ee6d_proprio(state)
        torch.testing.assert_close(proprio[0, 3:9], torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0]))

    def test_grip_site_offset_matches_the_panda_gripper_xml(self) -> None:
        """The frame correction is the -90-degree z rotation the Panda gripper XML declares.

        ``robosuite``'s ``panda_gripper.xml`` mounts ``right_gripper`` on the ``right_hand``
        body with ``quat="0.707107 0 0 -0.707107"`` (mujoco ``w, x, y, z``), and leaves the
        ``eef`` body and its ``grip_site`` at identity below it.
        """
        from physicalai.policies.xvla.libero import GRIP_SITE_FROM_HAND_BODY  # noqa: PLC0415

        angle = -np.pi / 2
        expected = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert np.abs(GRIP_SITE_FROM_HAND_BODY - expected).max() < 1e-6

    def test_proprio_rotation_is_the_grip_site_rotation(self) -> None:
        """The encoded 6D rotation is the state's rotation composed with the site offset."""
        from physicalai.policies.xvla.libero import (  # noqa: PLC0415
            GRIP_SITE_FROM_HAND_BODY,
            _axis_angle_to_mat,
            _mat_to_rotation_6d,
        )

        rng = np.random.default_rng(7)
        for _ in range(20):
            axis_angle = rng.normal(size=3) * rng.uniform(0.01, 3.0)
            state = torch.zeros(1, LIBERO_STATE_DIM)
            state[0, 3:6] = torch.from_numpy(axis_angle).float()

            proprio = libero_state_to_ee6d_proprio(state)

            expected = _mat_to_rotation_6d(_axis_angle_to_mat(axis_angle.astype(np.float32)) @ GRIP_SITE_FROM_HAND_BODY)
            torch.testing.assert_close(proprio[0, 3:9], torch.from_numpy(expected), atol=1e-5, rtol=1e-5)

    def test_action_shape_and_arm_slot(self) -> None:
        """Only the first arm's 10-channel slot of a bimanual action is used."""
        action = torch.zeros(1, CHUNK_SIZE, 20)
        action[..., 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # identity rotation
        libero_action = ee6d_action_to_libero(action)
        assert libero_action.shape == (1, CHUNK_SIZE, LIBERO_ACTION_DIM)
        torch.testing.assert_close(libero_action[..., 3:6], torch.zeros(1, CHUNK_SIZE, 3))

    def test_action_rejects_narrow_input(self) -> None:
        """An action narrower than one arm's slot is reported."""
        with pytest.raises(ValueError, match="at least"):
            ee6d_action_to_libero(torch.zeros(1, 5))

    @pytest.mark.parametrize(("gripper_prob", "expected"), [(0.9, 1.0), (0.1, -1.0), (0.5, -1.0)])
    def test_gripper_threshold(self, gripper_prob: float, expected: float) -> None:
        """A gripper probability above 0.5 closes the gripper; at or below opens it."""
        action = torch.zeros(1, 20)
        action[..., 9] = gripper_prob
        libero_action = ee6d_action_to_libero(action)
        assert libero_action[0, 6].item() == expected


# ============================================================================ #
# XVLALiberoPolicy                                                             #
# ============================================================================ #


class TestXVLALiberoPolicy:
    """End-to-end shape and behaviour tests for the bridged policy."""

    def test_is_an_xvla_policy(self) -> None:
        """The bridge is a real XVLA subclass, not a duck-typed wrapper."""
        from physicalai.policies.xvla import XVLA  # noqa: PLC0415

        assert isinstance(build_policy(), XVLA)

    def test_predict_action_chunk_shape(self) -> None:
        """A predicted chunk is LIBERO-shaped, not the model's native bimanual width."""
        policy = build_policy()
        policy.eval()
        actions = policy.predict_action_chunk(make_libero_observation())
        assert actions.shape == (BATCH_SIZE, CHUNK_SIZE, LIBERO_ACTION_DIM)
        assert bool(torch.isfinite(actions).all())

    def test_select_action_returns_a_single_libero_action(self) -> None:
        """``select_action`` (inherited from the base ``Policy``) also sees the bridge."""
        policy = build_policy()
        policy.eval()
        action = policy.select_action(make_libero_observation(batch_size=1))
        assert action.shape == (1, LIBERO_ACTION_DIM)

    def test_select_action_refills_the_queue(self) -> None:
        """Executing past a chunk's end transparently predicts and bridges the next one."""
        policy = build_policy()
        policy.eval()
        observation = make_libero_observation(batch_size=1)
        actions = [policy.select_action(observation) for _ in range(2 * CHUNK_SIZE + 1)]
        assert all(action.shape == (1, LIBERO_ACTION_DIM) for action in actions)

    def test_gripper_channel_is_plus_minus_one(self) -> None:
        """Every bridged action's gripper channel is exactly LIBERO's binary signal."""
        policy = build_policy()
        policy.eval()
        actions = policy.predict_action_chunk(make_libero_observation())
        gripper = actions[..., 6]
        assert bool(((gripper == 1.0) | (gripper == -1.0)).all())
