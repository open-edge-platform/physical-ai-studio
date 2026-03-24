# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for defining exportable PyTorch models."""


class ExportModel:
    """Mixin class for exportable PyTorch models.

    This mixin provides a common interface and utilities for PyTorch models that can be exported
    to various formats (e.g., ONNX, OpenVINO). It is designed to be used in conjunction with the
    base Model class to enable seamless integration of export functionality.
    """
