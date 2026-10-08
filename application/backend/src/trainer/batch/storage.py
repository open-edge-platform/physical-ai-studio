# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Where a batch job's inputs come from and its outputs go.

Each cloud provider supplies one :class:`JobStorage` implementation; the
batch run itself is provider-agnostic. :class:`LocalDirJobStorage` backs
tests and dry runs.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Protocol

from training import TrainingJobSpec

if TYPE_CHECKING:
    from pathlib import Path

    from trainer.batch.settings import BatchSettings
    from trainer.batch.status import BatchJobStatus


class JobStorage(Protocol):
    """Provider-specific access to one job's files."""

    def fetch_spec(self) -> TrainingJobSpec: ...

    def fetch_dataset(self, dest: Path) -> None: ...

    def put_status(self, status: BatchJobStatus) -> None: ...

    def put_artifact(self, archive: Path) -> None: ...


class LocalDirJobStorage:
    """Job storage rooted at a local directory."""

    def __init__(self, root: Path, settings: BatchSettings) -> None:
        self._root = root
        self._settings = settings

    def fetch_spec(self) -> TrainingJobSpec:
        return TrainingJobSpec.model_validate_json((self._root / self._settings.spec_key).read_text())

    def fetch_dataset(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._root / self._settings.dataset_key, dest)

    def put_status(self, status: BatchJobStatus) -> None:
        (self._root / self._settings.status_key).write_text(status.model_dump_json())

    def put_artifact(self, archive: Path) -> None:
        shutil.copyfile(archive, self._root / self._settings.artifact_key)
