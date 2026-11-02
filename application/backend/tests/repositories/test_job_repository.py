import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.schema import Base, ProjectDB
from repositories.job_repo import JobRepository
from repositories.mappers.job_mapper import JobMapper
from schemas.aws_auth import AwsAuthenticationIssue
from schemas.base_job import JobStatus
from schemas.job import RemoteTrainJobPayload, TrainJob
from services.job_service import JobService


def test_duplicate_remote_job_payload_with_uuids_is_json_serializable() -> None:
    """Duplicate detection matches the persisted JSON representation of UUID fields."""

    async def check_duplicate() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        project_id = uuid4()
        payload = RemoteTrainJobPayload(
            project_id=project_id,
            dataset_id=uuid4(),
            policy="act",
            model_name="test-model",
            base_model_id=uuid4(),
            snapshot_id=uuid4(),
            remote_trainer_id=uuid4(),
            remote_trainer_url="https://trainer.example.test",
        )

        async with engine.begin() as connection:
            await connection.run_sync(lambda sync_connection: Base.metadata.create_all(sync_connection))

        async with session_factory() as session:
            session.add(ProjectDB(id=str(project_id), name="Test project"))
            await session.commit()
            job = TrainJob(project_id=project_id, payload=payload, created_at=datetime.now(tz=UTC))
            session.add(JobMapper.to_schema(job))
            await session.commit()

            repository = JobRepository(session)
            assert await repository.is_job_duplicate(project_id, payload)

            remote_job_id = uuid4()
            updated_payload = payload.model_copy(update={"remote_job_id": remote_job_id})
            updated_job = await repository.update(job, {"payload": updated_payload.model_dump(mode="json")})
            assert updated_job.payload.remote_job_id == remote_job_id

        await engine.dispose()

    asyncio.run(check_duplicate())


def test_authentication_updates_preserve_persisted_cancellation_and_remote_identity() -> None:
    async def check() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        project_id, remote_id, snapshot_id = uuid4(), uuid4(), uuid4()
        payload = RemoteTrainJobPayload(
            project_id=project_id,
            dataset_id=uuid4(),
            policy="act",
            model_name="test",
            remote_trainer_id=uuid4(),
            remote_job_id=remote_id,
            snapshot_id=snapshot_id,
        )
        async with factory() as session:
            session.add(ProjectDB(id=str(project_id), name="Test"))
            await session.commit()
            job = TrainJob(project_id=project_id, payload=payload, created_at=datetime.now(UTC))
            session.add(JobMapper.to_schema(job))
            await session.commit()
            service = JobService(session)
            await service.update_remote_job_state(job.id, cancel_requested=True)
            issue = AwsAuthenticationIssue(
                profile_name="studio", code="aws_credentials_expired", title="Expired", detail="Renew studio"
            )
            waiting = await service.update_remote_job_state(job.id, authentication_issue=issue)
            assert waiting.payload.cancel_requested is True
            assert waiting.payload.authentication_issue == issue
            progress = await service.update_job_status(job.id, status="running", progress=50)
            assert progress.payload.cancel_requested is True
            assert progress.payload.authentication_issue == issue
            new_snapshot_id = uuid4()
            updated = await service.update_remote_job_state(job.id, snapshot_id=new_snapshot_id)
            assert updated.payload.snapshot_id == new_snapshot_id
            assert updated.payload.authentication_issue == issue
            unchanged = await service.update_remote_job_state(job.id)
            assert unchanged.payload.authentication_issue == issue
            restored = await service.update_remote_job_state(job.id, authentication_issue=None)
            assert restored.payload.authentication_issue is None
            assert restored.payload.cancel_requested is True
            assert restored.payload.remote_job_id == remote_id
            assert restored.payload.snapshot_id == new_snapshot_id
        await engine.dispose()

    asyncio.run(check())


@pytest.mark.anyio
async def test_pending_training_transition_rejects_other_states() -> None:
    """Only claiming and cancellation are valid queued training transitions."""
    session = AsyncMock()
    with pytest.raises(ValueError, match="only be claimed or canceled"):
        await JobRepository(session).transition_pending_train_job(uuid4(), JobStatus.COMPLETED)
    session.execute.assert_not_awaited()
