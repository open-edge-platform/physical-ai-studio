# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.schema import JobDB, ModelDB
from exceptions import DuplicateJobException, InsufficientDiskSpaceError, InvalidResourceError
from repositories.mappers import JobMapper
from schemas.job import Job, OptimizeModelJob, OptimizeModelJobPayload
from schemas.model import Model
from schemas.model_variant import ExportPackage, VariantInfo
from services.model_service import ModelService
from settings import get_settings
from utils.filesystem import FileTree


class ModelOptimizationService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def submit(self, model_id: UUID) -> OptimizeModelJob:
        try:
            # Acquire the database write lock before checking for concurrent submissions.
            await self.session.execute(update(ModelDB).where(ModelDB.id == str(model_id)).values(id=ModelDB.id))
            model = await ModelService(self.session).get_model_by_id(model_id)
            source = ExportPackage(ModelService.resolve_export(model, "openvino"), "openvino")
            if source.info.weight_compression is not None:
                raise InvalidResourceError("Model", "INT8 compression requires an uncompressed OpenVINO export")
            jobs = await self.session.execute(
                select(JobDB.id, JobDB.status).where(
                    JobDB.type == "optimize_model",
                    JobDB.status.in_(("pending", "running", "completed")),
                    func.json_extract(JobDB.payload, "$.model_id") == str(model.id),
                )
            )
            pending = any(status != "completed" or self.staging(UUID(job_id)).exists() for job_id, status in jobs)
            if pending or ExportPackage.for_model(Path(model.path), "openvino", "int8").directory.exists():
                raise DuplicateJobException("An INT8 export already exists or is being created")
            job = OptimizeModelJob(
                project_id=model.project_id,
                payload=OptimizeModelJobPayload(model_id=model.id),
                message="Creating INT8 export",
            )
            self.session.add(JobMapper.to_schema(job))
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
        destination.parent.mkdir(parents=True, exist_ok=True)
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
        return FileTree(get_settings().models_dir).resolve(Path("optimization") / str(job_id))

    @staticmethod
    def publish(model: Model, job_id: UUID) -> None:
        """Publish an already-committed job's complete package with one rename."""
        staging = ModelOptimizationService.staging(job_id)
        destination = ExportPackage.for_model(Path(model.path), "openvino", "int8")
        if destination.directory.exists():
            if destination.info.job_id != job_id:
                raise FileExistsError("An INT8 export from another job already exists")
            shutil.rmtree(staging, ignore_errors=True)
            return
        if ExportPackage(staging, "openvino", "int8").info.job_id != job_id:
            raise ValueError("The staged export does not belong to this job")
        staging.rename(destination.directory)

    async def cleanup(self, job: Job) -> None:
        if not isinstance(job, OptimizeModelJob):
            raise TypeError("Variant cleanup requires an optimization job")
        shutil.rmtree(self.staging(job.id), ignore_errors=True)
