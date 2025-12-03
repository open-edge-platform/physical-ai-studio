# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for data."""

from typing import Any

import torch

from .observation import Observation


def infer_batch_size(batch: dict[str, Any] | Observation) -> int:
    """Infer the batch size from the first tensor in the batch.

    This function scans the values of the input batch dictionary for important keys and returns
    the size of the first dimension of the first `torch.Tensor` it finds. It
    assumes that all tensors in the batch have the same batch dimension.

    Args:
        batch (dict[str, Any] | Observation): A dictionary where values may include tensors.

    Returns:
        int: The inferred batch size.

    Raises:
        ValueError: If no tensor is found in the batch.
    """
    data = batch.__dict__ if isinstance(batch, Observation) else batch

    priority_keys = ("action", "state", "images")

    # first scan observation keys we expect
    for key in priority_keys:
        if key in data:
            value = data[key]
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, torch.Tensor):
                        return item.shape[0]

    # fallback, scan all the values looking for a tensor
    for value in data.values():
        if isinstance(value, torch.Tensor):
            return value.shape[0]
        if isinstance(value, dict):
            for item in value.values():
                if isinstance(item, torch.Tensor):
                    return item.shape[0]

    msg = "Could not infer batch size — no tensors found."
    raise ValueError(msg)


__all__ = ["infer_batch_size"]
