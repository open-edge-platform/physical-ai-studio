# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Post-export hooks for model compression and optimization."""

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def compress_weights_openvino_int8_sym(model_path: str) -> None:
    """Compress an exported OpenVINO model to INT8 symmetric weights.

    Reads the model from ``model_path``, applies ``nncf.compress_weights``
    with ``INT8_SYM`` mode, and replaces the original model files.

    The compressed model is first saved to a temporary directory, then the
    original ``.xml`` and ``.bin`` files are replaced atomically to avoid
    issues with OpenVINO overwriting files it currently has open.

    Args:
        model_path: Path to the exported OpenVINO IR model (``.xml`` file).

    Raises:
        ImportError: If ``nncf`` is not installed.
    """
    try:
        import nncf  # noqa: PLC0415  # pyrefly: ignore[missing-import]
    except ImportError as e:
        msg = "nncf is required for weight compression hooks. Install with: pip install physicalai-train[nncf]"
        raise ImportError(msg) from e

    import openvino  # noqa: PLC0415

    logger.info("Compressing weights to INT8_SYM: %s", model_path)

    xml_path = Path(model_path)
    bin_path = xml_path.with_suffix(".bin")

    core = openvino.Core()
    ov_model = core.read_model(str(xml_path))

    compressed_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_SYM)

    # Stage the compressed model in a temp dir on the *same filesystem* as the export
    # (the parent directory) so the final swap can be an atomic ``Path.replace``. Using the
    # system temp dir risks a cross-filesystem move, which is non-atomic and can leave the
    # export without a model if it fails midway.
    with tempfile.TemporaryDirectory(dir=xml_path.parent) as tmp_dir:
        tmp_xml = Path(tmp_dir) / xml_path.name
        openvino.save_model(compressed_model, str(tmp_xml))
        tmp_bin = tmp_xml.with_suffix(".bin")

        # Release model references before touching original files —
        # OpenVINO memory-maps the .bin file, causing segfault if deleted while mapped.
        del compressed_model, ov_model, core

        # Atomically replace the originals. ``Path.replace`` overwrites the destination in a
        # single step on the same filesystem, so the originals are never removed unless the
        # compressed files are already in place.
        tmp_xml.replace(xml_path)
        tmp_bin.replace(bin_path)

    logger.info("INT8_SYM weight compression complete: %s", model_path)
