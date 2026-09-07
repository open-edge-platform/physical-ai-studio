# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Vendored RLDX-1 data-processing parity oracles (test-only).

These modules are copied from RLWRLD/RLDX-1 (itself derived from NVIDIA Isaac
GR00T N1.7, Apache-2.0). They are the golden reference the native
``Rldx1Preprocessor`` is validated against and have no production consumer, so
they live under the test tree rather than ``src``.

Modules:
    data_types              -> modality + action config enums / dataclasses
    pose                    -> Pose / EndEffectorPose / JointPose primitives
    action_chunking         -> ActionChunk and concrete chunk types
    data_utils              -> numpy normalization / modality helpers
    state_action_processor  -> numpy StateActionProcessor oracle
    processing_rldx         -> RLDXProcessor / RLDXDataCollator oracle (image
                               geometry uses the production torchvision-based
                               ``physicalai.policies.rldx1.augmentations``, no
                               albumentations dependency anywhere)
"""
