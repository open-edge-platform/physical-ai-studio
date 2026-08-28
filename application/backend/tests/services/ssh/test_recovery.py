# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for SSH startup recovery orchestration.

`SshProvisioningService` itself is faked out entirely: these tests prove the
*orchestration* - reconciling stale rows, classifying each active row's
`verify_reattach` outcome, and excluding the right job ids from the per-server
sweep - independent of the real docker/SSH calls `verify_reattach` makes
(covered in `test_provisioning.py`).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from exceptions import ResourceNotFoundError, ResourceType
from schemas.hardware import DeviceType
from schemas.job_provisioning import JobProvisioning
from schemas.remote_server import RemoteServer
from services.ssh import recovery as recovery_module
from services.ssh.provisioning import ReattachFailureReason, ReattachVerification
from services.ssh.recovery import recover_ssh_jobs

# Every test in this module is async; mark the whole module rather than each
# test individually.
pytestmark = pytest.mark.anyio


def _server(name: str = "Lab GPU box") -> RemoteServer:
    return RemoteServer(id=uuid4(), name=name, ssh_host_alias="gpu-box", device_type=DeviceType.CUDA)


def _row(server: RemoteServer, *, container_name: str | None = "physicalai-trainer-job") -> JobProvisioning:
    return JobProvisioning(
        job_id=uuid4(), remote_server_id=server.id, ssh_host_alias=server.ssh_host_alias, container_name=container_name
    )


class FakeProvisioningRepository:
    """Stands in for `JobProvisioningRepository`: only the methods recovery calls."""

    def __init__(self, active: list[JobProvisioning], stale: list[JobProvisioning] | None = None) -> None:
        self._active = list(active)
        self._stale = list(stale or [])
        self.deleted: list = []

    async def list_active(self) -> list[JobProvisioning]:
        return list(self._active)

    async def list_stale(self) -> list[JobProvisioning]:
        return list(self._stale)

    async def delete_by_job_id(self, job_id) -> None:
        self.deleted.append(job_id)


class FakeRemoteServerService:
    """Stands in for `RemoteServerService`."""

    def __init__(self, servers: list[RemoteServer]) -> None:
        self._servers = {server.id: server for server in servers}

    async def get_remote_server(self, remote_server_id) -> RemoteServer:
        server = self._servers.get(remote_server_id)
        if server is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))
        return server

    async def list_remote_servers(self) -> list[RemoteServer]:
        return list(self._servers.values())


class FakeJobService:
    """Stands in for `JobService`: records every terminal-status write."""

    def __init__(self) -> None:
        self.status_updates: list[tuple] = []

    async def update_job_status(self, *, job_id, status, message=None, **kwargs) -> None:
        self.status_updates.append((job_id, status, message))


class FakeProvisioningService:
    """Stands in for `SshProvisioningService`, scripted per test.

    Recovery constructs this itself (`SshProvisioningService(provisioning_repo,
    settings)`), so the class - not an instance - is monkeypatched in; the
    scripted outcomes and call log live on class-level state reset per test via
    the `_fake_provisioning_service` fixture.
    """

    verify_outcomes: dict = {}
    sweep_results: dict = {}
    teardown_calls: list = []
    verify_calls: list = []
    sweep_calls: list = []

    def __init__(self, repository, settings=None) -> None:
        self._repository = repository

    async def teardown(self, job_id, server) -> None:
        FakeProvisioningService.teardown_calls.append(job_id)
        await self._repository.delete_by_job_id(job_id)

    async def verify_reattach(self, row: JobProvisioning, server: RemoteServer) -> ReattachVerification:
        FakeProvisioningService.verify_calls.append(row.job_id)
        return FakeProvisioningService.verify_outcomes[row.job_id]

    async def sweep_orphans(self, server: RemoteServer, active_job_ids: set) -> list[str]:
        FakeProvisioningService.sweep_calls.append((server.id, frozenset(active_job_ids)))
        return FakeProvisioningService.sweep_results.get(server.id, [])


@pytest.fixture(autouse=True)
def _fake_provisioning_service(monkeypatch):
    FakeProvisioningService.verify_outcomes = {}
    FakeProvisioningService.sweep_results = {}
    FakeProvisioningService.teardown_calls = []
    FakeProvisioningService.verify_calls = []
    FakeProvisioningService.sweep_calls = []
    monkeypatch.setattr(recovery_module, "SshProvisioningService", FakeProvisioningService)
    yield


async def test_confirmed_job_is_excluded_from_orphan_sweep() -> None:
    server = _server()
    row = _row(server)
    FakeProvisioningService.verify_outcomes[row.job_id] = ReattachVerification(ok=True)

    job_service = FakeJobService()
    report = await recover_ssh_jobs(job_service, FakeProvisioningRepository([row]), FakeRemoteServerService([server]))

    assert report.confirmed == 1
    assert report.failed == 0
    # Requeued to PENDING: `run_loop` only ever picks up jobs it finds
    # PENDING, and only that pickup opens the real tunnel and resumes SSE
    # streaming - a confirmed-but-still-RUNNING job would otherwise train to
    # completion on the remote server with nothing in Studio reattached to it.
    assert len(job_service.status_updates) == 1
    confirmed_job_id, status, message = job_service.status_updates[0]
    assert confirmed_job_id == row.job_id
    assert status.value == "pending"
    assert "restart" in message
    assert FakeProvisioningService.sweep_calls == [(server.id, frozenset({row.job_id}))]
    # Excluded from the generic orphan abort that follows, even though its
    # own job payload may not have persisted a remote_job_id yet.
    assert report.handled_job_ids == frozenset({row.job_id})


async def test_unrecoverable_outcome_fails_the_job_and_is_excluded_from_active_set() -> None:
    server = _server()
    row = _row(server)
    FakeProvisioningService.verify_outcomes[row.job_id] = ReattachVerification(
        ok=False, reason=ReattachFailureReason.CONTAINER_GONE
    )

    job_service = FakeJobService()
    report = await recover_ssh_jobs(job_service, FakeProvisioningRepository([row]), FakeRemoteServerService([server]))

    assert report.failed == 1
    assert report.confirmed == 0
    assert len(job_service.status_updates) == 1
    failed_job_id, status, message = job_service.status_updates[0]
    assert failed_job_id == row.job_id
    assert status.value == "failed"
    assert "no longer running" in message
    # Excluded from the active set the sweep is told about, so its container
    # (if this installation still owns one) is reclaimed by the sweep.
    assert FakeProvisioningService.sweep_calls == [(server.id, frozenset())]
    # Already explicitly failed here; still reported handled so the generic
    # pass's separate query for it (which would find it already FAILED, i.e.
    # not RUNNING) is a redundant, harmless no-op rather than a re-judgment.
    assert report.handled_job_ids == frozenset({row.job_id})


async def test_transient_outcome_leaves_job_pending_but_still_excludes_it_from_sweep() -> None:
    """Transient outcomes are neither failed nor torn down; they stay claimed by this job."""
    server = _server()
    row = _row(server)
    FakeProvisioningService.verify_outcomes[row.job_id] = ReattachVerification(
        ok=False, reason=ReattachFailureReason.HEALTH_NEVER_READY
    )

    job_service = FakeJobService()
    report = await recover_ssh_jobs(job_service, FakeProvisioningRepository([row]), FakeRemoteServerService([server]))

    assert report.transient == 1
    assert report.failed == 0
    # Requeued to PENDING, same as the confirmed case: `run_loop` only ever
    # picks up PENDING jobs, so this is what actually gets the reattach retried.
    assert len(job_service.status_updates) == 1
    assert job_service.status_updates[0][0] == row.job_id
    assert job_service.status_updates[0][1].value == "pending"
    # Still claimed - a transient outcome does not lose its exclusion from the sweep.
    assert FakeProvisioningService.sweep_calls == [(server.id, frozenset({row.job_id}))]
    # And is excluded from the generic orphan abort: it has no persisted
    # remote_job_id, so that pass alone would otherwise fail it outright.
    assert report.handled_job_ids == frozenset({row.job_id})


async def test_inspection_failed_outcome_is_also_treated_as_transient() -> None:
    """`INSPECTION_FAILED` (an operational `docker inspect` error) is retried,
    never conflated with a confirmed-gone container."""
    server = _server()
    row = _row(server)
    FakeProvisioningService.verify_outcomes[row.job_id] = ReattachVerification(
        ok=False, reason=ReattachFailureReason.INSPECTION_FAILED
    )

    job_service = FakeJobService()
    report = await recover_ssh_jobs(job_service, FakeProvisioningRepository([row]), FakeRemoteServerService([server]))

    assert report.transient == 1
    assert report.failed == 0
    assert len(job_service.status_updates) == 1
    assert job_service.status_updates[0][1].value == "pending"
    assert FakeProvisioningService.sweep_calls == [(server.id, frozenset({row.job_id}))]


async def test_row_with_no_container_yet_is_transient_and_never_calls_verify_reattach() -> None:
    server = _server()
    row = _row(server, container_name=None)

    job_service = FakeJobService()
    report = await recover_ssh_jobs(job_service, FakeProvisioningRepository([row]), FakeRemoteServerService([server]))

    assert report.transient == 1
    assert FakeProvisioningService.verify_calls == []
    # Requeued to PENDING so this job actually gets picked back up and
    # verified once a container eventually gets launched for it.
    assert len(job_service.status_updates) == 1
    assert job_service.status_updates[0][1].value == "pending"
    assert FakeProvisioningService.sweep_calls == [(server.id, frozenset({row.job_id}))]


async def test_reattach_verification_runs_before_the_orphan_sweep() -> None:
    server = _server()
    row = _row(server)
    FakeProvisioningService.verify_outcomes[row.job_id] = ReattachVerification(ok=True)

    await recover_ssh_jobs(FakeJobService(), FakeProvisioningRepository([row]), FakeRemoteServerService([server]))

    assert FakeProvisioningService.verify_calls == [row.job_id]
    assert len(FakeProvisioningService.sweep_calls) == 1


async def test_stale_rows_are_torn_down_and_deleted_before_active_rows_are_processed() -> None:
    server = _server()
    stale_row = _row(server)
    active_row = _row(server)
    FakeProvisioningService.verify_outcomes[active_row.job_id] = ReattachVerification(ok=True)

    repo = FakeProvisioningRepository([active_row], stale=[stale_row])
    report = await recover_ssh_jobs(FakeJobService(), repo, FakeRemoteServerService([server]))

    assert report.stale_rows_cleaned == 1
    assert FakeProvisioningService.teardown_calls == [stale_row.job_id]
    assert repo.deleted == [stale_row.job_id]
    # The stale row was never handed to verify_reattach.
    assert FakeProvisioningService.verify_calls == [active_row.job_id]


async def test_stale_row_for_a_deregistered_server_is_dropped_without_a_teardown_attempt() -> None:
    server = _server()
    stale_row = _row(server)
    repo = FakeProvisioningRepository([], stale=[stale_row])
    # No server registered at all: `get_remote_server` always raises.
    report = await recover_ssh_jobs(FakeJobService(), repo, FakeRemoteServerService([]))

    assert report.stale_rows_cleaned == 1
    assert FakeProvisioningService.teardown_calls == []
    assert repo.deleted == [stale_row.job_id]


async def test_missing_server_fails_the_active_job_without_touching_other_servers() -> None:
    known_server = _server("Known box")
    row_for_missing_server = JobProvisioning(
        job_id=uuid4(), remote_server_id=uuid4(), ssh_host_alias="ghost", container_name="physicalai-trainer-ghost"
    )
    row_for_known_server = _row(known_server)
    FakeProvisioningService.verify_outcomes[row_for_known_server.job_id] = ReattachVerification(ok=True)

    job_service = FakeJobService()
    repo = FakeProvisioningRepository([row_for_missing_server, row_for_known_server])
    report = await recover_ssh_jobs(job_service, repo, FakeRemoteServerService([known_server]))

    assert report.failed == 1
    assert report.confirmed == 1
    assert job_service.status_updates[0][0] == row_for_missing_server.job_id
    assert job_service.status_updates[0][1].value == "failed"
    # Only the known server was ever swept - there is no transport to dial for
    # a server that no longer exists.
    assert [server_id for server_id, _ in FakeProvisioningService.sweep_calls] == [known_server.id]
