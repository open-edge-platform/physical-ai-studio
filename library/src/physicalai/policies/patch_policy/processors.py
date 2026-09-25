# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from .config import PatchPolicyConfig


class PatchPolicyPreprocessor(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.nn.Identity()(batch)


class PatchPolicyPostprocessor(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.nn.Identity()(batch)


def make_policy_processors(config: PatchPolicyConfig) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Make Path Policies' preprocessor and postprocessor.

    Returns:
        A tuple of (preprocessor, postprocessor).
    """
    del config
    preprocessor = PatchPolicyPreprocessor()
    postprocessor = PatchPolicyPostprocessor()
    return preprocessor, postprocessor
