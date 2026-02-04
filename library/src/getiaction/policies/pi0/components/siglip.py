# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SigLIP vision encoder utilities for Pi0/Pi0.5."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import SiglipVisionModel


class SigLIPVisionEncoder(nn.Module):
    """Thin wrapper for SigLIP vision model."""

    def __init__(self, model_name: str, *, dtype: torch.dtype = torch.float32) -> None:
        """Initialize SigLIP vision encoder."""  # noqa: DOC501
        super().__init__()
        try:
            from transformers import SiglipVisionModel  # noqa: PLC0415
        except ImportError as e:
            msg = "SigLIP requires transformers. Install with: pip install transformers"
            raise ImportError(msg) from e

        self.model: SiglipVisionModel = SiglipVisionModel.from_pretrained(  # nosec B615
            model_name,
            torch_dtype=dtype,
            revision="main",
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to patch embeddings."""  # noqa: DOC201
        outputs = self.model(pixel_values)
        return outputs.last_hidden_state
