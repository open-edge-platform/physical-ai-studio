# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utils for input preprocessing."""

from enum import StrEnum


class PreprocessorBatchKeys(StrEnum):
    """Enum for preprocessor batch keys."""

    TOKENIZED_PROMPT = "tokenized_prompt"
    TOKENIZED_PROMPT_MASK = "tokenized_prompt_mask"
    IMAGE_MASKS = "image_masks"

TOKENIZED_PROMPT = PreprocessorBatchKeys.TOKENIZED_PROMPT.value
TOKENIZED_PROMPT_MASK = PreprocessorBatchKeys.TOKENIZED_PROMPT_MASK.value
IMAGE_MASKS = PreprocessorBatchKeys.IMAGE_MASKS.value


