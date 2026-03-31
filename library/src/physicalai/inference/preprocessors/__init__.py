# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference preprocessors.

Preprocessors transform observation dicts before the adapter bridge
flattens and filters them for the runtime adapter.
"""

from physicalai.inference.preprocessors.base import Preprocessor
from physicalai.inference.preprocessors.lambda_processor import LambdaPreprocessor

__all__ = [
    "LambdaPreprocessor",
    "Preprocessor",
]
