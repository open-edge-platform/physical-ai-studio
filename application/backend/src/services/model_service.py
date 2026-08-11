import shutil
from pathlib import Path
from uuid import UUID

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceNotFoundError, ResourceType
from repositories import ModelRepository, SnapshotRepository
from schemas.job import TrainJob
from schemas.model import BackendExportDetail, Model, TrainingSummary


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
        exports_dir = Path(model.path) / "exports"
        if not exports_dir.is_dir():
            return []

        details: list[BackendExportDetail] = []
        for backend_dir in sorted(exports_dir.iterdir()):
            detail = BackendExportDetail.from_backend_dir(backend_dir)
            if detail is not None:
                details.append(detail)

        return details

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

        return TrainingSummary(
            max_epochs=payload.max_epochs,
            max_steps=payload.max_steps,
            batch_size=payload.batch_size,
            precision=str(payload.precision),
            compile_model=payload.compile_model,
            val_split=payload.val_split,
            auto_scale_batch_size=payload.auto_scale_batch_size,
            num_workers=payload.num_workers,
            device_type=device_type,
        )
