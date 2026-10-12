# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from physicalai.data import NormalizationParameters
from physicalai.policies.utils import JointFrameTransform


def test_joint_transform_round_trip_uses_supplied_frame() -> None:
    transform = JointFrameTransform(signs=(1.0, -1.0), offsets=(10.0, 20.0))
    robot_values = torch.tensor([[2.0, 3.0, 4.0]])

    checkpoint_values = transform.forward(robot_values)

    torch.testing.assert_close(checkpoint_values, torch.tensor([[12.0, 17.0, 4.0]]))
    torch.testing.assert_close(transform.inverse(checkpoint_values), robot_values)


def test_joint_transform_rejects_invalid_frame() -> None:
    with pytest.raises(ValueError, match="must match"):
        JointFrameTransform(signs=(1.0,), offsets=(0.0, 1.0))
    with pytest.raises(ValueError, match="either -1 or 1"):
        JointFrameTransform(signs=(2.0,), offsets=(0.0,))


def test_joint_transform_round_trip_applies_scales() -> None:
    transform = JointFrameTransform(signs=(1.0, -1.0), offsets=(10.0, 20.0), scales=(2.0, 0.5))
    robot_values = torch.tensor([[2.0, 4.0, 5.0]])

    checkpoint_values = transform.forward(robot_values)

    torch.testing.assert_close(checkpoint_values, torch.tensor([[14.0, 18.0, 5.0]]))
    torch.testing.assert_close(transform.inverse(checkpoint_values), robot_values)


def test_joint_transform_forward_normalization_applies_scales() -> None:
    transform = JointFrameTransform(signs=(1.0, -1.0), offsets=(10.0, 20.0), scales=(2.0, 0.5))
    normalization = NormalizationParameters(
        mean=[1.0, 2.0, 3.0],
        std=[1.0, 4.0, 2.0],
        q01=[-1.0, -4.0, 0.0],
        q99=[3.0, 8.0, 1.0],
    )

    transformed = transform.forward_normalization(normalization, dimension=3)

    assert transformed.mean == [12.0, 19.0, 3.0]
    assert transformed.std == [2.0, 2.0, 2.0]
    assert transformed.q01 == [8.0, 16.0, 0.0]
    assert transformed.q99 == [16.0, 22.0, 1.0]


def test_joint_transform_normalized_values_match_transformed_values() -> None:
    transform = JointFrameTransform(signs=(1.0, -1.0), offsets=(0.0, 90.0), scales=(1.5, 1.2))
    normalization = NormalizationParameters(q01=[-50.0, -80.0], q99=[40.0, 60.0])
    robot_values = torch.tensor([[-10.0, 25.0]])

    source = (robot_values - torch.tensor(normalization.q01)) / (
        torch.tensor(normalization.q99) - torch.tensor(normalization.q01)
    )
    target_normalization = transform.forward_normalization(normalization, dimension=2)
    target = (transform.forward(robot_values) - torch.tensor(target_normalization.q01)) / (
        torch.tensor(target_normalization.q99) - torch.tensor(target_normalization.q01)
    )

    # A sign flip mirrors the quantile window, so the flipped joint's position becomes 1 - position.
    torch.testing.assert_close(target, torch.stack([source[:, 0], 1.0 - source[:, 1]], dim=-1))


def test_joint_transform_rejects_invalid_scales() -> None:
    with pytest.raises(ValueError, match="must match"):
        JointFrameTransform(signs=(1.0, 1.0), offsets=(0.0, 0.0), scales=(1.0,))
    with pytest.raises(ValueError, match="must be positive"):
        JointFrameTransform(signs=(1.0,), offsets=(0.0,), scales=(0.0,))
    with pytest.raises(ValueError, match="must be positive"):
        JointFrameTransform(signs=(1.0,), offsets=(0.0,), scales=(-1.0,))
