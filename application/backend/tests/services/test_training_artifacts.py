# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Keep the shared artifact import's security limits intact."""

from pathlib import Path
from zipfile import ZipFile

import pytest
from physicalai.data.archive_safety import InsufficientDiskSpaceError, ZipBombDetectedError

from services.training_backends.artifacts import extract_training_artifact


def test_extract_training_artifact(tmp_path: Path) -> None:
    archive = tmp_path / "model.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("nested/model.ckpt", "trained")
    destination = tmp_path / "model"
    extract_training_artifact(archive, destination, max_uncompressed_bytes=1024, min_free_bytes=0)
    assert (destination / "nested/model.ckpt").read_text() == "trained"


@pytest.mark.parametrize(
    "filename,max_bytes,min_free_bytes,error_type",
    [
        ("../escaped.ckpt", 1024, 0, ZipBombDetectedError),
        ("model.ckpt", 1, 0, ZipBombDetectedError),
        ("model.ckpt", 1024, 2**63, InsufficientDiskSpaceError),
    ],
)
def test_unsafe_or_oversized_artifacts_are_not_extracted(
    tmp_path: Path, filename: str, max_bytes: int, min_free_bytes: int, error_type: type[Exception]
) -> None:
    archive = tmp_path / "model.zip"
    with ZipFile(archive, "w") as output:
        output.writestr(filename, "trained")
    destination = tmp_path / "model"
    with pytest.raises(error_type):
        extract_training_artifact(archive, destination, max_bytes, min_free_bytes)
    assert not (tmp_path / "escaped.ckpt").exists()
    assert not (destination / "model.ckpt").exists()
