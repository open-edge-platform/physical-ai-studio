# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for `JobProvisioningRepository`'s active/stale queries against a real DB.

`list_active` finds jobs whose container might still be reclaimable, `list_stale`
finds provisioning rows a crashed teardown left behind for a job that already
finished, and `get_active_for_server` narrows `list_active` to a single server
(backing the remote-server status endpoint's `in_use_by_job_id`).
"""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.schema import Base, JobProvisioningDB, ProjectDB, RemoteServerDB
from repositories.job_provisioning_repo import JobProvisioningRepository
from repositories.mappers.job_mapper import JobMapper
from schemas.base_job import JobStatus
from schemas.job import SshTrainJobPayload, TrainingTarget, TrainJob


def _make_job(project_id, remote_server_id, status: JobStatus) -> TrainJob:
    payload = SshTrainJobPayload(
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


def test_get_active_for_server_returns_the_non_terminal_job_only() -> None:
    """`get_active_for_server` backs the status endpoint's `in_use_by_job_id`.

    A server with one terminal and one non-terminal provisioning row should
    report only the non-terminal one; a server with no rows at all reports
    `None`.
    """

    async def run() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        project_id = uuid4()
        busy_server_id = uuid4()
        idle_server_id = uuid4()

        async with engine.begin() as connection:
            await connection.run_sync(lambda sync_connection: Base.metadata.create_all(sync_connection))

        async with session_factory() as session:
            session.add(ProjectDB(id=str(project_id), name="Test project"))
            session.add_all(
                [
                    RemoteServerDB(
                        id=str(busy_server_id), name="Busy GPU box", ssh_host_alias="busy-box", device_type="cuda"
                    ),
                    RemoteServerDB(
                        id=str(idle_server_id), name="Idle GPU box", ssh_host_alias="idle-box", device_type="cuda"
                    ),
                ]
            )
            await session.commit()

            running_job = JobMapper.to_schema(_make_job(project_id, busy_server_id, JobStatus.RUNNING))
            failed_job = JobMapper.to_schema(_make_job(project_id, busy_server_id, JobStatus.FAILED))
            session.add_all([running_job, failed_job])
            await session.commit()

            session.add_all(
                [
                    JobProvisioningDB(
                        job_id=failed_job.id,
                        remote_server_id=str(busy_server_id),
                        ssh_host_alias="busy-box",
                        container_name="physicalai-trainer-failed",
                    ),
                    JobProvisioningDB(
                        job_id=running_job.id,
                        remote_server_id=str(busy_server_id),
                        ssh_host_alias="busy-box",
                        container_name="physicalai-trainer-running",
                    ),
                ]
            )
            await session.commit()

            repository = JobProvisioningRepository(session)
            active = await repository.get_active_for_server(busy_server_id)
            idle = await repository.get_active_for_server(idle_server_id)

            assert active is not None
            assert str(active.job_id) == running_job.id
            assert idle is None

        await engine.dispose()

    asyncio.run(run())
