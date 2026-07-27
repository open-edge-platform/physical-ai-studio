"""Integration coverage for the `remote_servers` / `job_provisioning` migration.

Proves the Alembic migration matches the ORM schema and that a record
persisted through the repository/mapper round-trips correctly, including its
encrypted-at-rest secret fields.
"""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.migration import MigrationManager
from db.schema import JobDB, ProjectDB
from repositories.job_provisioning_repo import JobProvisioningRepository
from repositories.remote_server_repo import RemoteServerRepository
from schemas.hardware import DeviceType
from schemas.job_provisioning import JobProvisioning
from schemas.remote_server import LastCheckSummary, RemoteServerInternal, SSHAuthType
from settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(DATA_DIR=str(tmp_path / "data"), STORAGE_DIR=str(tmp_path / "storage"))


def test_migration_creates_expected_tables_and_columns(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = MigrationManager(settings)

    assert manager.run_migrations() is True

    from sqlalchemy import create_engine

    engine = create_engine(settings.database_url_sync)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert "remote_servers" in tables
    assert "job_provisioning" in tables

    remote_server_columns = {c["name"] for c in inspector.get_columns("remote_servers")}
    assert {
        "id",
        "name",
        "host",
        "port",
        "username",
        "auth_type",
        "device_type",
        "ssh_secret_encrypted",
        "ssh_key_passphrase_encrypted",
        "host_key",
        "last_check_status",
        "last_check_at",
        "last_check_latency_ms",
        "last_check_reason_code",
        "created_at",
        "updated_at",
    } <= remote_server_columns
    # Confidential/internal columns are the mapper's job to withhold from the API,
    # not the DB's — but they must exist as ciphertext-only storage columns.
    assert "ssh_secret_encrypted" in remote_server_columns

    job_provisioning_columns = {c["name"] for c in inspector.get_columns("job_provisioning")}
    assert {
        "job_id",
        "remote_server_id",
        "image_ref",
        "image_fallback_reason",
        "image_digest",
        "container_id",
        "container_name",
        "remote_port",
        "local_tunnel_port",
        "trainer_build_version",
        "trainer_protocol_version",
    } <= job_provisioning_columns

    engine.dispose()


@pytest.mark.anyio
async def test_remote_server_and_job_provisioning_round_trip_through_repository(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert MigrationManager(settings).run_migrations() is True

    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        server_repo = RemoteServerRepository(session)
        record = RemoteServerInternal(
            id=uuid4(),
            name="gpu-box",
            host="10.0.0.5",
            port=22,
            username="trainer",
            auth_type=SSHAuthType.PASSWORD,
            device_type=DeviceType.CUDA,
            last_check=LastCheckSummary(),
            ssh_secret_encrypted="ciphertext-secret",
            ssh_key_passphrase_encrypted=None,
            host_key=None,
        )
        await server_repo.save(record)

        fetched = await server_repo.get_by_id(record.id)
        assert fetched is not None
        assert fetched.ssh_secret_encrypted == "ciphertext-secret"
        # Never expose secret/internal fields through the public schema.
        public = fetched.to_public()
        assert not hasattr(public, "ssh_secret_encrypted")

        job_id = uuid4()
        session.add(ProjectDB(id="project-1", name="Project"))
        await session.commit()
        session.add(
            JobDB(
                id=str(job_id),
                project_id="project-1",
                type="training",
                progress=0,
                status="pending",
                message="",
                payload={},
            )
        )
        await session.commit()

        provisioning_repo = JobProvisioningRepository(session)
        provisioning = JobProvisioning(
            job_id=job_id,
            remote_server_id=record.id,
            image_ref="ghcr.io/example/physicalai-trainer-cuda:abc123",
            container_id="container-abc",
            remote_port=54321,
            local_tunnel_port=12345,
        )
        await provisioning_repo.upsert(provisioning)

        fetched_provisioning = await provisioning_repo.get_by_job_id(job_id)
        assert fetched_provisioning is not None
        assert fetched_provisioning.container_id == "container-abc"
        assert fetched_provisioning.remote_port == 54321

        await provisioning_repo.delete_by_job_id(job_id)
        assert await provisioning_repo.get_by_job_id(job_id) is None

    await engine.dispose()
