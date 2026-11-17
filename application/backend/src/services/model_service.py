import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceNotFoundError, ResourceType
from repositories import ModelRepository, SnapshotRepository
from schemas.job import TrainJob
from schemas.model import BackendExportDetail, BackendIOSpec, ExportVariantDetail, Model, TrainingSummary
from schemas.model_variant import ExportPackage, VariantId


class ModelService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = ModelRepository(session)

    async def get_model_list(self) -> list[Model]:
        return await self.repo.get_all()

    async def get_model_by_id(self, model_id: UUID) -> Model:
        model = await self.repo.get_by_id(model_id)
        if model is None:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        return model

    async def create_model(self, model: Model) -> Model:
        return await self.repo.save(model)

    async def update_model(self, model: Model, update: dict) -> Model:
        return await self.repo.update(model, update)

    async def delete_model(self, model: Model) -> None:
        snapshot_repo = SnapshotRepository(self.session)
        await self.repo.delete_by_id(model.id)

        if model.snapshot_id is not None:
            await snapshot_repo.delete_by_id(model.snapshot_id)

        model_path = Path(model.path).expanduser()
        shutil.rmtree(model_path)

    async def get_project_models(self, project_id: UUID) -> list[Model]:
        return await self.repo.get_project_models(project_id)

    @staticmethod
    def get_backend_details(model: Model) -> list[BackendExportDetail]:
        """Compute per-backend export details from the filesystem."""
        grouped: dict[str, list[ExportVariantDetail]] = {}
        for (backend, variant), directory in sorted(model.export_packages().items()):
            package = ExportPackage(directory, backend, variant)
            grouped.setdefault(backend, []).append(
                ExportVariantDetail(
                    **package.info.model_dump(),
                    size_bytes=sum(path.stat().st_size for path in package.files),
                    file_count=len(package.files),
                    exported_at=package.info.created_at
                    or datetime.fromtimestamp((directory / "manifest.json").stat().st_mtime, tz=UTC),
                    io_spec=BackendIOSpec.from_manifest(package.manifest),
                )
            )
        details = []
        for backend, variants in sorted(grouped.items()):
            variants.sort(key=lambda variant: variant.id != "original")
            original = next((variant for variant in variants if variant.id == "original"), None)
            details.append(
                BackendExportDetail(
                    type=backend,
                    size_bytes=original.size_bytes if original else 0,
                    file_count=original.file_count if original else 0,
                    exported_at=original.exported_at if original else None,
                    io_spec=original.io_spec if original else None,
                    variants=variants,
                )
            )
        return details

    @staticmethod
    def resolve_export(model: Model, backend: str, variant: VariantId = "original") -> Path:
        directory = model.export_packages().get((backend, variant))
        if directory is None:
            raise ResourceNotFoundError(ResourceType.MODEL, f"{model.id}/{backend}/{variant}")
        return directory

    @staticmethod
    def get_hparams(model: Model) -> dict | None:
        """Read training hyperparameters from the model directory.

        Looks for ``version_0/hparams.yaml`` (written by Lightning's CSVLogger).
        """
        hparams_path = Path(model.path) / "version_0" / "hparams.yaml"
        if not hparams_path.is_file():
            return None
        with hparams_path.open() as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_training_summary(training_job: TrainJob | None) -> TrainingSummary | None:
        """Extract a summary of training configuration from a training job.

        This merges fields from the job's payload (batch size, precision, etc.)
        with computed values like training duration.
        """
        if training_job is None:
            return None

        payload = training_job.payload
        device_type = str(payload.device.type) if payload.device is not None else None
        lora_enabled = payload.lora_enabled
        snapflow_distill_epochs = payload.snapflow_distill_epochs if payload.snapflow_enabled else None

        return TrainingSummary(
            max_epochs=payload.max_epochs,
            max_steps=payload.max_steps,
            batch_size=payload.batch_size,
            precision=str(payload.precision),
            compile_model=payload.compile_model,
            augment_images=payload.augment_images,
            val_split=payload.val_split,
            auto_scale_batch_size=payload.auto_scale_batch_size,
            num_workers=payload.num_workers,
            device_type=device_type,
            lora_enabled=lora_enabled,
            lora_rank=payload.lora_rank if lora_enabled else None,
            lora_alpha=payload.lora_alpha if lora_enabled else None,
            lora_dropout=payload.lora_dropout if lora_enabled else None,
            lora_use_dora=payload.lora_use_dora if lora_enabled else None,
            snapflow_enabled=payload.snapflow_enabled,
            snapflow_distill_epochs=snapflow_distill_epochs,
        )
