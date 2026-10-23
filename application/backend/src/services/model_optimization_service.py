# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import DuplicateJobException, InsufficientDiskSpaceError, InvalidResourceError
from repositories.job_repo import JobRepository
from schemas.job import OptimizeModelJob, OptimizeModelJobPayload
from schemas.model import Model, ModelOptimizationInfo
from services.model_service import ModelService
from settings import get_settings


class ModelOptimizationService:
    """Submit optimization jobs and prepare independent OpenVINO exports."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def submit(self, model_id: UUID) -> OptimizeModelJob:
        """Queue symmetric INT8 weight compression for an existing model."""
        model = await ModelService(self.session).get_model_by_id(model_id)
        if model.optimization is not None:
            raise InvalidResourceError("Model", "This model is already optimized")
        self.openvino_artifact(Path(model.path) / "exports" / "openvino")
        payload = OptimizeModelJobPayload(model_id=model.id)
        repository = JobRepository(self.session)
        if await repository.is_job_duplicate(model.project_id, payload):
            raise DuplicateJobException
        job = OptimizeModelJob(project_id=model.project_id, payload=payload, message="Model optimization queued")
        await repository.save(job)
        return job

    @staticmethod
    def openvino_artifact(export_dir: Path) -> Path:
        """Resolve the manifest's policy IR without accepting paths outside its export."""
        try:
            manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
            artifact = manifest["model"]["artifacts"]["openvino"]
            if not isinstance(artifact, str) or not artifact:
                raise ValueError("Missing OpenVINO artifact")
            relative_path = Path(artifact)
            model_path = (export_dir / relative_path).resolve()
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("Unsafe OpenVINO artifact path")
            if not model_path.is_relative_to(export_dir.resolve()) or model_path.suffix != ".xml":
                raise ValueError("OpenVINO artifact must be an XML file inside the export")
            if not model_path.is_file() or not model_path.with_suffix(".bin").is_file():
                raise ValueError("OpenVINO XML or weights are missing")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise InvalidResourceError("Model", "A complete OpenVINO export is required for optimization") from exc
        return model_path

    @staticmethod
    def optimize_export(model: Model, destination: Path) -> ModelOptimizationInfo:
        """Copy and compress the policy IR, recording portable optimization metadata."""
        from physicalai.export.hooks import compress_weights_openvino_int8_sym

        source = Path(model.path) / "exports" / "openvino"
        artifact = ModelOptimizationService.openvino_artifact(source).relative_to(source.resolve())
        required_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
        # Compression stages another IR beside the copied export before replacing it.
        if shutil.disk_usage(destination.parent).free < required_bytes * 2:
            raise InsufficientDiskSpaceError("Not enough disk space to copy and optimize the OpenVINO export")
        export_dir = destination / "exports" / "openvino"
        shutil.copytree(source, export_dir)
        manifest_path = export_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        studio = manifest.setdefault("studio", {})
        if not isinstance(studio, dict):
            raise InvalidResourceError("Model", "The manifest's studio metadata must be an object")
        compress_weights_openvino_int8_sym(str(export_dir / artifact))
        optimization = ModelOptimizationInfo(source_model_id=model.id)
        studio["optimization"] = optimization.model_dump(mode="json")
        # Runtime preserves application-specific top-level manifest fields.
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return optimization

    @staticmethod
    def cleanup(job_id: UUID) -> None:
        """Remove unpublished output after cancellation, failure, or a killed worker."""
        models_dir = get_settings().models_dir
        for path in (models_dir / f".optimize-{job_id}", models_dir / str(job_id)):
            shutil.rmtree(path, ignore_errors=True)
