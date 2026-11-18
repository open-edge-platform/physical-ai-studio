# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.schema import JobDB, ModelDB
from exceptions import DuplicateJobException, InvalidResourceError
from repositories.mappers import JobMapper
from schemas.job import Job, ModelExportJob, ModelExportJobPayload
from schemas.model import Model, ModelExportAction
from schemas.model_export import ExportInfo, ExportPackage
from services.model_service import ModelService
from settings import get_settings
from utils.filesystem import FileTree


class ModelExportService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def submit(self, model_id: UUID, action: ModelExportAction) -> ModelExportJob:
        payload = ModelExportJobPayload(model_id=model_id, backend=action.backend, operation=action.operation)
        try:
            # Acquire the database write lock before checking for concurrent submissions.
            await self.session.execute(update(ModelDB).where(ModelDB.id == str(model_id)).values(id=ModelDB.id))
            model = await ModelService(self.session).get_model_by_id(model_id)
            if action not in ModelService.get_export_actions(model):
                raise InvalidResourceError("Model", "This export is unavailable, unsupported, or already exists")
            jobs = await self.session.execute(
                select(JobDB.id, JobDB.status).where(
                    JobDB.type == "model_export",
                    JobDB.status.in_(("pending", "running", "completed")),
                    func.json_extract(JobDB.payload, "$.model_id") == str(model.id),
                )
            )
            pending = any(status != "completed" or self.staging(UUID(job_id)).exists() for job_id, status in jobs)
            target = ExportPackage.for_model(Path(model.path), payload.backend)
            if pending or target.directory.exists():
                raise DuplicateJobException("An export already exists or is being created")
            job = ModelExportJob(
                project_id=model.project_id,
                payload=payload,
                message=f"Creating {payload.backend} export",
            )
            self.session.add(JobMapper.to_schema(job))
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return job

    @staticmethod
    def staging(job_id: UUID) -> Path:
        return FileTree(get_settings().models_dir).resolve(Path("export_jobs") / str(job_id))

    @staticmethod
    def export_checkpoint(model: Model, destination: Path, job: ModelExportJob) -> None:
        from physicalai.export import ExportablePolicyMixin
        from physicalai.policies import get_physicalai_policy_class

        from services.training_backends.local import resolve_hf_token
        from training.job import _hf_token_env

        checkpoint = FileTree(Path(model.path)).resolve(Path("model.ckpt"))
        destination.mkdir(parents=True)
        with _hf_token_env(resolve_hf_token()):
            policy = get_physicalai_policy_class(model.policy).load_from_checkpoint(
                str(checkpoint), map_location="cpu", compile_model=False
            )
            if not isinstance(policy, ExportablePolicyMixin):
                raise InvalidResourceError("Model", "Policy does not support export")
            policy.eval()
            # Export workers are isolated processes; keep ONNX sidecars inside their staging tree.
            with tempfile.TemporaryDirectory(prefix="export-", dir=destination) as temporary:
                previous = tempfile.tempdir
                try:
                    tempfile.tempdir = temporary
                    policy.export(destination, backend=job.payload.backend.value)
                finally:
                    tempfile.tempdir = previous
        manifest = ExportPackage(destination, job.payload.backend).manifest
        manifest.setdefault("studio", {})["export"] = ExportInfo(
            created_at=datetime.now(UTC), job_id=job.id
        ).model_dump(mode="json")
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def publish(model: Model, job_id: UUID, backend: str) -> None:
        """Publish an already-committed job's complete package with one rename."""
        staging = ModelExportService.staging(job_id)
        destination = ExportPackage.for_model(Path(model.path), backend)
        if destination.directory.exists():
            if destination.info.job_id != job_id:
                raise FileExistsError("An export from another job already exists")
            shutil.rmtree(staging, ignore_errors=True)
            return
        if ExportPackage(staging, backend).info.job_id != job_id:
            raise ValueError("The staged export does not belong to this job")
        staging.rename(destination.directory)

    async def cleanup(self, job: Job) -> None:
        if not isinstance(job, ModelExportJob):
            raise TypeError("Export cleanup requires a model export job")
        shutil.rmtree(self.staging(job.id), ignore_errors=True)
