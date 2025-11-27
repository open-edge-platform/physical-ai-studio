# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for data."""

from typing import Any

import torch

from .observation import Observation


def infer_batch_size(batch: dict[str, Any] | Observation) -> int:
    """Infer the batch size from the first tensor in the batch.

    This function scans the values of the input batch dictionary and returns
    the size of the first dimension of the first `torch.Tensor` it finds. It
    assumes that all tensors in the batch have the same batch dimension.

    Args:
        batch (dict[str, Any] | Observation): A dictionary where values may include tensors.

    Returns:
        int: The inferred batch size.

    Raises:
        ValueError: If no tensor is found in the batch.
    """
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            return v.shape[0]
    msg = "Could not infer batch size from batch."
    raise ValueError(msg)


__all__ = ["infer_batch_size"]
