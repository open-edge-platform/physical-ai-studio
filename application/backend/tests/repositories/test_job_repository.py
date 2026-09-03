import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.schema import Base, JobDB, ProjectDB
from repositories.job_repo import JobRepository
from repositories.mappers.job_mapper import JobMapper
from schemas.base_job import JobStatus, JobType
from schemas.job import RemoteTrainJobPayload, TrainingTarget, TrainJob


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


def test_legacy_training_job_missing_training_target_is_backfilled() -> None:
    """Jobs persisted before `training_target` existed must still load.

    Regression test for `union_tag_not_found` on `training.payload`: a job
    submitted before remote/SSH training was added has no `training_target`
    key in its stored payload JSON at all.
    """

    async def check_legacy_jobs() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        project_id = uuid4()
        remote_trainer_id = uuid4()

        async with engine.begin() as connection:
            await connection.run_sync(lambda sync_connection: Base.metadata.create_all(sync_connection))

        async with session_factory() as session:
            session.add(ProjectDB(id=str(project_id), name="Test project"))
            await session.commit()
            session.add_all(
                [
                    JobDB(
                        id=str(uuid4()),
                        project_id=str(project_id),
                        type=JobType.TRAINING,
                        progress=0,
                        status=JobStatus.PENDING,
                        message="Job created",
                        payload={
                            "project_id": str(project_id),
                            "dataset_id": str(uuid4()),
                            "policy": "act",
                            "model_name": "legacy-local-model",
                            "batch_size": 8,
                            # No `training_target` key: predates the field entirely.
                        },
                    ),
                    JobDB(
                        id=str(uuid4()),
                        project_id=str(project_id),
                        type=JobType.TRAINING,
                        progress=0,
                        status=JobStatus.PENDING,
                        message="Job created",
                        payload={
                            "project_id": str(project_id),
                            "dataset_id": str(uuid4()),
                            "policy": "act",
                            "model_name": "legacy-remote-model",
                            "batch_size": 8,
                            "remote_trainer_id": str(remote_trainer_id),
                            # No `training_target` key either.
                        },
                    ),
                    JobDB(
                        id=str(uuid4()),
                        project_id=str(project_id),
                        type=JobType.TRAINING,
                        progress=0,
                        status=JobStatus.PENDING,
                        message="Job created",
                        payload={
                            "project_id": str(project_id),
                            "dataset_id": str(uuid4()),
                            "policy": "act",
                            "model_name": "legacy-local-with-stale-fields",
                            "batch_size": 8,
                            "training_target": "local",
                            # Stale fields from before per-target payloads forbade extras.
                            "remote_trainer_id": None,
                            "remote_trainer_url": None,
                            "remote_trainer_name": None,
                        },
                    ),
                ]
            )
            await session.commit()

            repository = JobRepository(session)
            jobs = await repository.get_jobs_by_type(project_id, JobType.TRAINING)

        targets = {job.payload.model_name: job.payload.training_target for job in jobs}
        assert targets["legacy-local-model"] is TrainingTarget.LOCAL
        assert targets["legacy-remote-model"] is TrainingTarget.REMOTE
        assert targets["legacy-local-with-stale-fields"] is TrainingTarget.LOCAL

        await engine.dispose()

    asyncio.run(check_legacy_jobs())
