from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from db.schema import DatasetRecordingEnvironmentDB
from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from schemas.dataset import RecordingEnvironment

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class RecordingEnvironmentService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _to_model(row: DatasetRecordingEnvironmentDB) -> RecordingEnvironment:
        mappings = row.mappings or {}
        return RecordingEnvironment(
            id=UUID(row.id),
            dataset_id=UUID(row.dataset_id),
            environment_id=UUID(row.environment_id),
            camera_mappings=mappings.get("camera_mappings", []),
        )

    async def list(self, dataset_id: UUID) -> list[RecordingEnvironment]:
        rows = (
            await self.session.scalars(
                select(DatasetRecordingEnvironmentDB)
                .where(DatasetRecordingEnvironmentDB.dataset_id == str(dataset_id))
                .order_by(DatasetRecordingEnvironmentDB.created_at)
            )
        ).all()
        return [self._to_model(row) for row in rows]

    async def get(self, dataset_id: UUID, recording_environment_id: UUID) -> RecordingEnvironment:
        row = await self.session.scalar(
            select(DatasetRecordingEnvironmentDB).where(
                DatasetRecordingEnvironmentDB.id == str(recording_environment_id),
                DatasetRecordingEnvironmentDB.dataset_id == str(dataset_id),
            )
        )
        if row is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, recording_environment_id)
        return self._to_model(row)

    async def create(self, dataset_id: UUID, environment_id: UUID, camera_mappings: list[dict]) -> RecordingEnvironment:
        row = DatasetRecordingEnvironmentDB(
            dataset_id=str(dataset_id),
            environment_id=str(environment_id),
            mappings={"camera_mappings": camera_mappings},
        )
        self.session.add(row)
        try:
            await self.session.commit()
        except IntegrityError as error:
            await self.session.rollback()
            raise ResourceAlreadyExistsError(
                "Recording environment", "This environment is already assigned to the dataset."
            ) from error
        return self._to_model(row)

    async def update(
        self, dataset_id: UUID, recording_environment_id: UUID, camera_mappings: list[dict]
    ) -> RecordingEnvironment:
        row = await self.session.scalar(
            select(DatasetRecordingEnvironmentDB).where(
                DatasetRecordingEnvironmentDB.id == str(recording_environment_id),
                DatasetRecordingEnvironmentDB.dataset_id == str(dataset_id),
            )
        )
        if row is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, recording_environment_id)
        row.mappings = {"camera_mappings": camera_mappings}
        await self.session.commit()
        return self._to_model(row)

    async def delete(self, dataset_id: UUID, recording_environment_id: UUID) -> None:
        row = await self.session.scalar(
            select(DatasetRecordingEnvironmentDB).where(
                DatasetRecordingEnvironmentDB.id == str(recording_environment_id),
                DatasetRecordingEnvironmentDB.dataset_id == str(dataset_id),
            )
        )
        if row is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, recording_environment_id)
        await self.session.delete(row)
        await self.session.commit()
