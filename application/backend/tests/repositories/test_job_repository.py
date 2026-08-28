import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.schema import Base, ProjectDB
from repositories.job_repo import JobRepository
from repositories.mappers.job_mapper import JobMapper
from schemas.base_job import JobStatus, JobType
from schemas.job import LocalTrainJobPayload, RemoteTrainJobPayload, TrainJob


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


def test_pending_jobs_are_filtered_and_returned_in_submission_order() -> None:
    """Pending jobs of the requested type are returned oldest first."""

    async def get_pending_jobs() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        project_id = uuid4()
        payload = LocalTrainJobPayload(
            project_id=project_id,
            dataset_id=uuid4(),
            policy="act",
            model_name="test-model",
        )
        oldest_job = TrainJob(
            project_id=project_id,
            payload=payload,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        newest_job = TrainJob(
            project_id=project_id,
            payload=payload,
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        completed_job = TrainJob(
            project_id=project_id,
            payload=payload,
            status=JobStatus.COMPLETED,
            created_at=datetime(2026, 1, 3, tzinfo=UTC),
        )

        async with engine.begin() as connection:
            await connection.run_sync(lambda sync_connection: Base.metadata.create_all(sync_connection))

        async with session_factory() as session:
            session.add(ProjectDB(id=str(project_id), name="Test project"))
            await session.commit()
            session.add_all(JobMapper.to_schema(job) for job in (newest_job, completed_job, oldest_job))
            await session.commit()

            repository = JobRepository(session)
            pending_jobs = await repository.get_pending_jobs_by_type(JobType.TRAINING)

            assert [job.id for job in pending_jobs] == [oldest_job.id, newest_job.id]

        await engine.dispose()

    asyncio.run(get_pending_jobs())
