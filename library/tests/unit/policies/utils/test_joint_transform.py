# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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
