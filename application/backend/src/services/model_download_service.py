# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

SNAPSHOT_DIR_PATTERN = re.compile(r"^snapshot_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


class ModelDownloadService:
    @staticmethod
    def _is_snapshot_path(relative_path: Path) -> bool:
        """Check if a path is inside a snapshot directory."""
        top_level = relative_path.parts[0] if relative_path.parts else ""
        return bool(SNAPSHOT_DIR_PATTERN.match(top_level))

    def create_model_archive(self, model_path: Path, *, include_snapshot: bool = False) -> Path:
        """Create a zip archive of a model folder and return the archive path.

        :param model_path: Path to the model directory on disk.
        :param include_snapshot: When False (default), files inside snapshot_*
            directories are excluded from the archive.
        :return: Path to the temporary zip archive.
        """
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"Model path not found or not a directory: {model_path}")

        temporary_archive_path = Path(tempfile.gettempdir()) / f"model-{uuid4()}.zip"

        with zipfile.ZipFile(temporary_archive_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for path in model_path.rglob("*"):
                if not path.is_file():
                    continue

                relative = path.relative_to(model_path)

                if not include_snapshot and self._is_snapshot_path(relative):
                    continue

                archive.write(path, arcname=relative)

        return temporary_archive_path

    def create_backend_archive(self, export_dir: Path, backend: str) -> Path:
        """Create a zip archive of a single backend export directory.

        :param export_dir: Path to the backend export directory on disk.
        :param backend: Backend name used in the archive filename.
        :return: Path to the temporary zip archive.
        """
        if not export_dir.exists() or not export_dir.is_dir():
            raise FileNotFoundError(f"Backend export path not found or not a directory: {export_dir}")

        archive_path = Path(tempfile.gettempdir()) / f"model-export-{backend}-{uuid4()}.zip"

        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for path in export_dir.rglob("*"):
                if not path.is_file():
                    continue
                archive.write(path, arcname=path.relative_to(export_dir))

        return archive_path
