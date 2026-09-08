import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.schema import Base, ProjectDB
from repositories.job_repo import JobRepository
from repositories.mappers.job_mapper import JobMapper
from schemas.job import RemoteTrainJobPayload, TrainJob


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
