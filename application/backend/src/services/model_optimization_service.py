# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import fcntl
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.schema import JobDB
from exceptions import DuplicateJobException, InsufficientDiskSpaceError, InvalidResourceError, ResourceNotFoundError
from repositories.mappers import JobMapper
from schemas.job import Job, OptimizeModelJob, OptimizeModelJobPayload
from schemas.model import Model
from schemas.model_variant import ExportPackage, VariantInfo
from services.model_service import ModelService
from settings import get_settings


class ModelOptimizationService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def submit(self, model_id: UUID) -> OptimizeModelJob:
        model = await ModelService(self.session).get_model_by_id(model_id)
        source = ExportPackage(ModelService.resolve_export(model, "openvino"), "openvino")
        if source.info.weight_compression is not None:
            raise InvalidResourceError("Model", "INT8 compression requires an uncompressed OpenVINO export")
        reservation = Path(model.path) / ".variant-reservations/openvino-int8"
        reservation.parent.mkdir(parents=True, exist_ok=True)
        if reservation.is_symlink() or reservation.parent.is_symlink():
            raise InvalidResourceError("Model", "Variant reservations must not contain symbolic links")
        # Keep the inode: unlinking a lock file can let two processes lock different files for the same target.
        with reservation.open("a+", encoding="utf-8") as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DuplicateJobException from exc
            active = await self.session.scalar(
                select(JobDB.id)
                .where(
                    JobDB.type == "optimize_model",
                    JobDB.status.in_(("pending", "running")),
                    func.json_extract(JobDB.payload, "$.model_id") == str(model.id),
                )
                .limit(1)
            )
            if active is not None or ExportPackage.for_model(Path(model.path), "openvino", "int8").directory.exists():
                raise DuplicateJobException("An INT8 export already exists or is being created")
            job = OptimizeModelJob(
                project_id=model.project_id,
                payload=OptimizeModelJobPayload(model_id=model.id),
                message="Creating INT8 export",
            )
            self.session.add(JobMapper.to_schema(job))
            try:
                await self.session.commit()
            except Exception:
                await self.session.rollback()
                raise
        return job

    @staticmethod
    def compress(model: Model, destination: Path, job_id: UUID) -> None:
        from physicalai.export.hooks import compress_weights_openvino_int8_sym

        source = ExportPackage(ModelService.resolve_export(model, "openvino"), "openvino")
        required_bytes = sum(path.stat().st_size for path in source.files)
        if shutil.disk_usage(destination.parent).free < required_bytes * 3:
            raise InsufficientDiskSpaceError("Not enough disk space to create the INT8 export")
        destination.mkdir()
        for path in source.files:
            target = destination / path.relative_to(source.directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        compress_weights_openvino_int8_sym(str(ExportPackage(destination, "openvino").artifact))
        manifest = ExportPackage(destination, "openvino").manifest
        manifest.setdefault("studio", {})["variant"] = VariantInfo(
            id="int8", weight_compression="int8_sym", created_at=datetime.now(UTC), job_id=job_id
        ).model_dump(mode="json")
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def staging(job_id: UUID) -> Path:
        return get_settings().models_dir / f".optimize-{job_id}"

    async def cleanup(self, job: Job) -> None:
        if not isinstance(job, OptimizeModelJob):
            raise TypeError("Variant cleanup requires an optimization job")
        shutil.rmtree(self.staging(job.id), ignore_errors=True)
        try:
            model = await ModelService(self.session).get_model_by_id(job.payload.model_id)
        except ResourceNotFoundError:
            return
        destination = ExportPackage.for_model(Path(model.path), "openvino", "int8").directory
        marker = destination / ".unpublished"
        if marker.is_file() and marker.read_text(encoding="utf-8") == str(job.id):
            shutil.rmtree(destination)
