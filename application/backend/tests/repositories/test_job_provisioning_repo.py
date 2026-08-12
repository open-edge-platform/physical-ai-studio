# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for `JobProvisioningRepository.list_active` / `list_stale` against a real DB.

Both drive startup recovery: `list_active` finds jobs whose container might
still be reclaimable, `list_stale` finds provisioning rows a crashed teardown
left behind for a job that already finished.
"""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.schema import Base, JobProvisioningDB, ProjectDB, RemoteServerDB
from repositories.job_provisioning_repo import JobProvisioningRepository
from repositories.mappers.job_mapper import JobMapper
from schemas.base_job import JobStatus
from schemas.job import TrainingTarget, TrainJob, TrainJobPayload


def _make_job(project_id, remote_server_id, status: JobStatus) -> TrainJob:
    payload = TrainJobPayload(
        project_id=project_id,
        dataset_id=uuid4(),
        policy="act",
        model_name="test-model",
        training_target=TrainingTarget.SSH,
        remote_server_id=remote_server_id,
    )
    return TrainJob(project_id=project_id, payload=payload, status=status, created_at=datetime.now(tz=UTC))


def test_list_active_and_list_stale_partition_by_job_status() -> None:
    """A PENDING/RUNNING job's row is active; a terminal job's row is stale."""

    async def run() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        project_id = uuid4()
        server_id = uuid4()

        async with engine.begin() as connection:
            await connection.run_sync(lambda sync_connection: Base.metadata.create_all(sync_connection))

        async with session_factory() as session:
            session.add(ProjectDB(id=str(project_id), name="Test project"))
            session.add(
                RemoteServerDB(id=str(server_id), name="Lab GPU box", ssh_host_alias="gpu-box", device_type="cuda")
            )
            await session.commit()

            pending_job = JobMapper.to_schema(_make_job(project_id, server_id, JobStatus.PENDING))
            running_job = JobMapper.to_schema(_make_job(project_id, server_id, JobStatus.RUNNING))
            failed_job = JobMapper.to_schema(_make_job(project_id, server_id, JobStatus.FAILED))
            session.add_all([pending_job, running_job, failed_job])
            await session.commit()

            session.add_all(
                [
                    JobProvisioningDB(
                        job_id=pending_job.id,
                        remote_server_id=str(server_id),
                        ssh_host_alias="gpu-box",
                        container_name="physicalai-trainer-pending",
                    ),
                    JobProvisioningDB(
                        job_id=running_job.id,
                        remote_server_id=str(server_id),
                        ssh_host_alias="gpu-box",
                        container_name="physicalai-trainer-running",
                    ),
                    JobProvisioningDB(
                        job_id=failed_job.id,
                        remote_server_id=str(server_id),
                        ssh_host_alias="gpu-box",
                        container_name="physicalai-trainer-failed",
                    ),
                ]
            )
            await session.commit()

            repository = JobProvisioningRepository(session)
            active = await repository.list_active()
            stale = await repository.list_stale()

            assert {str(row.job_id) for row in active} == {pending_job.id, running_job.id}
            assert {str(row.job_id) for row in stale} == {failed_job.id}

        await engine.dispose()

    asyncio.run(run())
