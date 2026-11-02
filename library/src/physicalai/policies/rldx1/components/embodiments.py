# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Embodiment-tag → projector-slot mapping for the RLDX-1 MSAT action head.

Extracted from the vendored ``processing/processing_rldx.py`` so that lightweight
consumers (e.g. :class:`Rldx1Config` validation) can resolve embodiment ids
without importing the full vendored data pipeline (``transformers``,
``albumentations``, ``torchvision``).
"""

from __future__ import annotations

# Projector slot (``embodiment_id``) per released RLDX-1 checkpoint.
#
# The id indexes the per-embodiment ``CategorySpecificLinear`` projectors in the
# MSAT action head (state/action encoders + decoder). A checkpoint must run on
# the same slot it was trained on; a wrong slot silently produces garbage
# actions. Values are taken from each model card's documented ``--embodiment-tag``
# and confirmed by weight diff vs ``RLWRLD/RLDX-1-PT`` (slot 35 is byte-identical
# to PT in every released FT, i.e. reserved/untrained).
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "general_embodiment": 0,  # FT-ROBOCASA, FT-RC365, FT-LIBERO, FT-GR1; default for new-robot FT
    "fractal20220817_data": 1,  # FT-SIMPLER-GOOGLE (OXE_FRACTAL)
    "bridge_orig": 3,  # FT-SIMPLER-WIDOWX (OXE_BRIDGE_ORIG)
    "new_embodiment": 35,  # legacy GR00T new-robot slot; superseded by general_embodiment
}

# Default slot for a fresh new-robot fine-tune from PT. RLDX-1 reserves and
# pre-conditions general_embodiment (slot 0, the highest-norm projector in PT)
# for downstream fine-tuning, and every released FT used it -- so train +
# inference both route through it self-consistently. The GR00T-inherited
# new_embodiment slot (35) is the lowest-norm reserved slot, superseded here and
# kept only for back-compat with GR00T-style checkpoints.
GENERAL_EMBODIMENT_ID = EMBODIMENT_TAG_TO_PROJECTOR_INDEX["general_embodiment"]
NEW_EMBODIMENT_ID = EMBODIMENT_TAG_TO_PROJECTOR_INDEX["new_embodiment"]
