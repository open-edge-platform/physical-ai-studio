# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
#
# Vendored RLDX-1 data-processing pipeline.
#
# These modules are copied verbatim (with import paths rewritten to this
# subpackage) from RLWRLD/RLDX-1, which is itself modified from NVIDIA Isaac
# GR00T N1.7. The original code is licensed under Apache-2.0; the per-file
# SPDX headers and provenance notices are preserved.
#
# Upstream: https://github.com/rlwrld/RLDX-1
#   rldx/utils/qwen_vision_process.py         -> qwen_vision_process.py
# Original: https://github.com/NVIDIA/Isaac-GR00T
#
# Studio modifications (documented in-file): import paths rewritten to relative
# imports. The vendored parity oracles and their numpy-only dependencies --
# ``RLDXProcessor`` / ``RLDXDataCollator`` / ``build_processor`` (upstream
# ``rldx/model/core/processing_rldx.py``), the numpy ``StateActionProcessor``
# (upstream ``rldx/data/state_action/state_action_processor.py``), and the
# ``pose`` / ``action_chunking`` / ``data_utils`` / ``data_types`` helpers they
# rely on -- are no longer part of this runtime subpackage: Studio's native
# :class:`Rldx1Preprocessor` replaced those paths, so their only consumer is the
# parity test. They now live under ``tests/unit/policies/rldx1_vendored/``
# (imports rewritten to relative). Only ``qwen_vision_process`` remains, since
# the native preprocessor still calls into it at runtime.
"""Vendored RLDX-1 data-processing pipeline (Apache-2.0).

Re-exports the shared embodiment-projector constants. The vendored
``RLDXProcessor`` collator pipeline, the numpy ``StateActionProcessor``, and
their ``pose`` / ``action_chunking`` / ``data_utils`` / ``data_types``
dependencies were moved to ``tests/unit/policies/rldx1_vendored/`` -- they are
parity oracles with no production consumer, so keeping them out of ``src``
avoids dragging that machinery into the live import path. The only runtime
module left here is ``qwen_vision_process``.
"""

from physicalai.policies.rldx1.components.embodiments import (
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    GENERAL_EMBODIMENT_ID,
    NEW_EMBODIMENT_ID,
)

__all__ = [
    "EMBODIMENT_TAG_TO_PROJECTOR_INDEX",
    "GENERAL_EMBODIMENT_ID",
    "NEW_EMBODIMENT_ID",
]
