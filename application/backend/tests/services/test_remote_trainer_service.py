import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from core.security.ssh_network_exposure import SshFeatureAvailability
from exceptions import (
    InvalidResourceError,
    ResourceAlreadyExistsError,
    ResourceInUseError,
    ResourceNotFoundError,
    SshAuthenticationError,
    SshConnectionError,
    SshFeatureDisabledError,
    SshHostKeyConfirmationRequiredError,
)
from schemas.remote_trainer import (
    AwsBatchConnection,
    DirectConnection,
    RemoteTrainer,
    RemoteTrainerCreate,
    RemoteTrainerUpdate,
    SshConnection,
)
from services import RemoteTrainerService
from services import remote_trainer_service as remote_trainer_service_module
from services.ssh import persistent_trainer

MODULE = "services.remote_trainer_service"


def _session() -> AsyncMock:
    return AsyncMock()


def _direct(url: str = "https://trainer.test") -> DirectConnection:
    return DirectConnection(url=url)


def _ssh(alias: str = "gpu-box", remote_port: int = 8001, local_port: int = 8001) -> SshConnection:
    return SshConnection(ssh_host_alias=alias, ssh_remote_port=remote_port, ssh_local_port=local_port)


def _aws() -> AwsBatchConnection:
    return AwsBatchConnection(
        region="eu-west-1",
        studio_role_arn="arn:aws:iam::123456789012:role/studio",
        bucket="jobs-bucket",
        targets={"g4dn.xlarge": {"queue": "arn:q", "job_definition": "arn:jd"}},
    )


def _remote_trainer(connection=None, name: str = "trainer") -> RemoteTrainer:
    return RemoteTrainer(id=uuid4(), name=name, connection=connection or _direct())


def test_remote_trainer_name_is_trimmed() -> None:
    config = RemoteTrainerCreate(name="  trainer  ", connection=_direct())

    assert config.name == "trainer"


def test_remote_trainer_rejects_whitespace_only_name() -> None:
    with pytest.raises(ValidationError):
        RemoteTrainerCreate(name="   ", connection=_direct())


@pytest.mark.anyio
async def test_list_remote_trainers_uses_stable_repository_order() -> None:
    session = _session()
    repository = MagicMock()
    repository.list_ordered = AsyncMock(return_value=[_remote_trainer()])

    with patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository):
        result = await RemoteTrainerService(session).list_remote_trainers()

    assert result == [repository.list_ordered.return_value[0]]
    repository.list_ordered.assert_awaited_once_with()


@pytest.mark.anyio
async def test_create_duplicate_url_returns_conflict() -> None:
    session = _session()
    repository = MagicMock()
    repository.list_ordered = AsyncMock(return_value=[_remote_trainer()])

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        pytest.raises(ResourceAlreadyExistsError) as error,
    ):
        await RemoteTrainerService(session).create_remote_trainer(RemoteTrainerCreate(name="dup", connection=_direct()))

    assert error.value.http_status == 409
    repository.save.assert_not_called()


@pytest.mark.anyio
async def test_create_aws_batch_trainer_skips_ssh_and_tunnel() -> None:
    session = _session()
    remote_trainer = _remote_trainer(_aws())
    repository = MagicMock()
    repository.save = AsyncMock(return_value=remote_trainer)
    repository.list_ordered = AsyncMock(return_value=[])

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=SshFeatureAvailability(network_exposed=True)),
        patch(f"{MODULE}.remote_trainer_tunnel_manager.sync_tunnel", new_callable=AsyncMock) as sync_tunnel,
        patch(f"{MODULE}.persistent_trainer.start", new_callable=AsyncMock) as start,
    ):
        saved = await RemoteTrainerService(session).create_remote_trainer(
            RemoteTrainerCreate(name="batch", connection=_aws())
        )

    assert saved is remote_trainer
    sync_tunnel.assert_awaited_once_with(remote_trainer, None)
    start.assert_not_called()


@pytest.mark.anyio
async def test_update_ignores_explicit_null_fields() -> None:
    session = _session()
    remote_trainer = _remote_trainer()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_trainer)
    repository.update = AsyncMock(return_value=remote_trainer)

    with patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository):
        await RemoteTrainerService(session).update_remote_trainer(remote_trainer.id, RemoteTrainerUpdate(name=None))

    repository.update.assert_awaited_once_with(remote_trainer, {})


@pytest.mark.anyio
async def test_update_restores_trainer_when_host_key_confirmation_is_required() -> None:
    session = _session()
    remote_trainer = _remote_trainer()
    updated_trainer = remote_trainer.model_copy(update={"name": "updated"})
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_trainer)
    repository.update = AsyncMock(side_effect=[updated_trainer, remote_trainer])

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.remote_trainer_tunnel_manager") as tunnel_manager,
        pytest.raises(SshHostKeyConfirmationRequiredError),
    ):
        tunnel_manager.sync_tunnel = AsyncMock(
            side_effect=SshHostKeyConfirmationRequiredError("trainer.test", "SHA256:fingerprint")
        )
        await RemoteTrainerService(session).update_remote_trainer(
            remote_trainer.id,
            RemoteTrainerUpdate(name="updated"),
        )

    assert repository.update.await_count == 2
    repository.update.assert_any_await(updated_trainer, remote_trainer.model_dump(include={"name", "connection"}))


@pytest.mark.anyio
async def test_delete_missing_remote_trainer_raises_not_found() -> None:
    session = _session()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=None)

    with patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository), pytest.raises(ResourceNotFoundError):
        await RemoteTrainerService(session).delete_remote_trainer(uuid4())

    repository.delete_by_id.assert_not_called()


@pytest.mark.anyio
async def test_create_rejects_ssh_tunnel_config_when_feature_inactive() -> None:
    session = _session()
    repository = MagicMock()

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(
            f"{MODULE}.get_ssh_feature_availability",
            return_value=SshFeatureAvailability(network_exposed=True),
        ),
        pytest.raises(SshFeatureDisabledError),
    ):
        await RemoteTrainerService(session).create_remote_trainer(
            RemoteTrainerCreate(name="trainer", connection=_ssh("training-box"))
        )

    repository.save.assert_not_called()


@pytest.mark.anyio
async def test_create_with_install_waits_for_prerequisites_before_starting_container() -> None:
    config = RemoteTrainerCreate(name="gpu", connection=_ssh())
    repository = MagicMock()
    saved: list[RemoteTrainer] = []

    async def save(trainer: RemoteTrainer) -> RemoteTrainer:
        saved.append(trainer)
        return trainer

    repository.save = AsyncMock(side_effect=save)
    repository.get_by_id = AsyncMock(side_effect=lambda _trainer_id: saved[0])
    repository.list_ordered = AsyncMock(return_value=[])
    transport = MagicMock()
    transport.__aenter__ = AsyncMock(return_value=transport)
    transport.__aexit__ = AsyncMock(return_value=False)
    install_started = asyncio.Event()
    finish_install = asyncio.Event()

    async def install(_transport: object) -> str:
        install_started.set()
        await finish_install.wait()
        return "ready"

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.remote_trainer_tunnel_manager.sync_tunnel", new_callable=AsyncMock) as sync_tunnel,
        patch(f"{MODULE}.SshTransport", return_value=transport),
        patch(f"{MODULE}.host_installer.install", side_effect=install),
        patch(f"{MODULE}.persistent_trainer.start", new_callable=AsyncMock) as start,
        patch.object(RemoteTrainerService, "_start_persistent_trainer_in_background") as launch,
        patch.object(RemoteTrainerService, "_require_no_active_jobs", new_callable=AsyncMock),
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=SshFeatureAvailability(network_exposed=False)),
    ):
        trainer = await RemoteTrainerService(_session()).create_remote_trainer(config, install_prerequisites=True)
        sync_tunnel.assert_awaited_once()
        await install_started.wait()
        launch.assert_not_called()
        start.assert_not_awaited()
        finish_install.set()
        await remote_trainer_service_module._background_installs[trainer.id]
        start.assert_awaited_once_with(trainer)


@pytest.mark.anyio
async def test_install_reconnects_after_docker_group_change() -> None:
    trainer = _remote_trainer(_ssh(), name="gpu")
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)
    transport = MagicMock()
    transport.__aenter__ = AsyncMock(return_value=transport)
    transport.__aexit__ = AsyncMock(return_value=False)
    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.SshTransport", return_value=transport) as connect,
        patch(f"{MODULE}.host_installer.install", new=AsyncMock(side_effect=["relogin_required", "ready"])) as install,
        patch(f"{MODULE}.persistent_trainer.start", new_callable=AsyncMock) as start,
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=SshFeatureAvailability(network_exposed=False)),
        patch.object(RemoteTrainerService, "_require_no_active_jobs", new_callable=AsyncMock),
    ):
        await RemoteTrainerService(_session()).install_remote_trainer(trainer.id)
        await remote_trainer_service_module._background_installs[trainer.id]
        assert connect.call_count == 2
        assert install.await_args_list[1].kwargs == {"check_only": True}
        start.assert_awaited_once_with(trainer)


@pytest.mark.anyio
async def test_create_with_install_rejects_direct_url() -> None:
    with pytest.raises(InvalidResourceError):
        await RemoteTrainerService(_session()).create_remote_trainer(
            RemoteTrainerCreate(name="direct", connection=_direct()), install_prerequisites=True
        )


@pytest.mark.anyio
async def test_install_reports_failure_without_starting_container() -> None:
    trainer = _remote_trainer(_ssh(), name="gpu")
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)
    transport = MagicMock()
    transport.__aenter__ = AsyncMock(return_value=transport)
    transport.__aexit__ = AsyncMock(return_value=False)
    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.SshTransport", return_value=transport),
        patch(f"{MODULE}.host_installer.install", new=AsyncMock(return_value="nvidia_driver_install_failed")),
        patch(f"{MODULE}.persistent_trainer.start", new_callable=AsyncMock) as start,
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=SshFeatureAvailability(network_exposed=False)),
        patch.object(RemoteTrainerService, "_require_no_active_jobs", new_callable=AsyncMock),
    ):
        service = RemoteTrainerService(_session())
        await service.install_remote_trainer(trainer.id)
        assert persistent_trainer.get_launch_phase(trainer.id) == "Installing host prerequisites"
        await remote_trainer_service_module._background_installs[trainer.id]
        assert persistent_trainer.get_launch_failure(trainer.id) == "nvidia_driver_install_failed"
        start.assert_not_awaited()


async def test_install_reserves_trainer_before_active_job_query_completes() -> None:
    trainer = _remote_trainer(_ssh(), name="gpu")
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)
    checking = asyncio.Event()
    release = asyncio.Event()
    finish_install = asyncio.Event()

    async def check_jobs(_trainer_id):
        checking.set()
        await release.wait()

    async def install_host(_transport):
        await finish_install.wait()
        return "installation_failed"

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=SshFeatureAvailability(network_exposed=False)),
        patch.object(RemoteTrainerService, "_require_no_active_jobs", side_effect=check_jobs),
        patch(f"{MODULE}.host_installer.install", new=AsyncMock(side_effect=install_host)),
        patch.object(remote_trainer_service_module, "_setup_lock", asyncio.Lock()),
        patch(f"{MODULE}.SshTransport") as transport,
    ):
        transport.return_value.__aenter__ = AsyncMock(return_value=transport.return_value)
        transport.return_value.__aexit__ = AsyncMock(return_value=False)
        service = RemoteTrainerService(_session())
        first = asyncio.create_task(service.install_remote_trainer(trainer.id))
        await checking.wait()
        second = asyncio.create_task(service.install_remote_trainer(trainer.id))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        await first
        with pytest.raises(ResourceInUseError):
            await second
        finish_install.set()
        await remote_trainer_service_module._background_installs[trainer.id]
        transport.assert_called_once()


async def test_install_reports_ssh_connection_failure_in_health() -> None:
    trainer = _remote_trainer(_ssh(), name="gpu")
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)
    transport = MagicMock()
    transport.__aenter__ = AsyncMock(side_effect=SshConnectionError("gpu-box"))
    transport.__aexit__ = AsyncMock(return_value=False)
    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.SshTransport", return_value=transport),
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=SshFeatureAvailability(network_exposed=False)),
        patch.object(RemoteTrainerService, "_require_no_active_jobs", new_callable=AsyncMock),
    ):
        await RemoteTrainerService(_session()).install_remote_trainer(trainer.id)
        await remote_trainer_service_module._background_installs[trainer.id]
    assert persistent_trainer.get_launch_failure(trainer.id) == "ssh_install_connection_failed"


@pytest.mark.parametrize("reason", ["reboot_required", "nvidia_driver_unavailable"])
async def test_reboot_refuses_host_with_running_containers(reason: str) -> None:
    trainer = _remote_trainer(_ssh(), name="gpu")
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)
    transport = MagicMock()
    transport.__aenter__ = AsyncMock(return_value=transport)
    transport.__aexit__ = AsyncMock(return_value=False)
    transport.run_command = AsyncMock(return_value=MagicMock(ok=True, stdout="running-container\n"))
    persistent_trainer.set_install_failure(trainer.id, reason)
    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.SshTransport", return_value=transport),
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=SshFeatureAvailability(network_exposed=False)),
        patch.object(RemoteTrainerService, "_require_no_active_jobs", new_callable=AsyncMock),
    ):
        await RemoteTrainerService(_session()).reboot_installed_host(trainer.id)
        await remote_trainer_service_module._background_installs[trainer.id]
    transport.run_command.assert_awaited_once_with(
        ["sh", "-c", 'if [ "$(id -u)" -eq 0 ]; then docker ps -q; else sudo -n docker ps -q; fi']
    )
    assert persistent_trainer.get_launch_failure(trainer.id) == "reboot_blocked_active_containers"


async def test_confirmed_reboot_waits_for_new_boot_before_launch() -> None:
    trainer = _remote_trainer(_ssh(), name="gpu")
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)
    before = MagicMock()
    before.__aenter__ = AsyncMock(return_value=before)
    before.__aexit__ = AsyncMock(return_value=False)
    before.run_command = AsyncMock(
        side_effect=[
            MagicMock(ok=True, stdout=""),
            MagicMock(ok=True, first_line=lambda: "old-boot"),
            MagicMock(ok=True),
        ]
    )
    after = MagicMock()
    after.__aenter__ = AsyncMock(return_value=after)
    after.__aexit__ = AsyncMock(return_value=False)
    after.run_command = AsyncMock(return_value=MagicMock(ok=True, first_line=lambda: "new-boot"))
    persistent_trainer.set_install_failure(trainer.id, "reboot_required")
    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.SshTransport", side_effect=[before, after]),
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=SshFeatureAvailability(network_exposed=False)),
        patch.object(RemoteTrainerService, "_require_no_active_jobs", new_callable=AsyncMock),
        patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock),
        patch(f"{MODULE}.host_installer.install", new=AsyncMock(return_value="ready")),
        patch(f"{MODULE}.persistent_trainer.start", new_callable=AsyncMock) as start,
    ):
        await RemoteTrainerService(_session()).reboot_installed_host(trainer.id)
        await remote_trainer_service_module._background_installs[trainer.id]
    before.run_command.assert_any_await(
        ["sh", "-c", 'if [ "$(id -u)" -eq 0 ]; then docker ps -q; else sudo -n docker ps -q; fi']
    )
    before.run_command.assert_any_await(["sudo", "-n", "reboot"])
    start.assert_awaited_once_with(trainer)


async def test_create_syncs_the_tunnel_manager_on_success() -> None:
    session = _session()
    remote_trainer = _remote_trainer()
    repository = MagicMock()
    repository.save = AsyncMock(return_value=remote_trainer)
    repository.list_ordered = AsyncMock(return_value=[])

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.remote_trainer_tunnel_manager") as tunnel_manager,
    ):
        tunnel_manager.sync_tunnel = AsyncMock()
        await RemoteTrainerService(session).create_remote_trainer(
            RemoteTrainerCreate(name="trainer", connection=_direct())
        )

    tunnel_manager.sync_tunnel.assert_awaited_once_with(remote_trainer, None)


@pytest.mark.anyio
async def test_create_deletes_trainer_when_tunnel_sync_fails() -> None:
    session = _session()
    remote_trainer = _remote_trainer()
    repository = MagicMock()
    repository.save = AsyncMock(return_value=remote_trainer)
    repository.list_ordered = AsyncMock(return_value=[])
    repository.delete_by_id = AsyncMock()

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.remote_trainer_tunnel_manager") as tunnel_manager,
        pytest.raises(SshAuthenticationError),
    ):
        tunnel_manager.sync_tunnel = AsyncMock(side_effect=SshAuthenticationError("trainer.test"))
        await RemoteTrainerService(session).create_remote_trainer(
            RemoteTrainerCreate(name="trainer", connection=_direct())
        )

    repository.delete_by_id.assert_awaited_once_with(remote_trainer.id)


@pytest.mark.anyio
async def test_create_rejects_an_ssh_host_port_already_used_by_another_trainer() -> None:
    session = _session()
    existing = _remote_trainer(_ssh("gpu-box", 8001, 9001), name="existing")
    repository = MagicMock()
    repository.list_ordered = AsyncMock(return_value=[existing])

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        pytest.raises(ResourceAlreadyExistsError, match="remote port 8001"),
    ):
        await RemoteTrainerService(session).create_remote_trainer(
            RemoteTrainerCreate(name="conflict", connection=_ssh("gpu-box", 8001, 9002))
        )

    repository.save.assert_not_called()


@pytest.mark.anyio
async def test_create_does_not_block_on_persistent_trainer_launch_for_ssh_trainer() -> None:
    """Regression guard: launching the container (image pull, etc.) can take
    minutes and must never block the save request - only the fast tunnel sync
    is awaited before the response returns.
    """
    session = _session()
    remote_trainer = _remote_trainer(_ssh(), name="ssh-trainer")
    repository = MagicMock()
    repository.save = AsyncMock(return_value=remote_trainer)
    repository.list_ordered = AsyncMock(return_value=[])

    started = asyncio.Event()

    async def _blocks_forever(*_args: object, **_kwargs: object) -> None:
        started.set()
        await asyncio.sleep(3600)

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.remote_trainer_tunnel_manager") as tunnel_manager,
        patch(f"{MODULE}.persistent_trainer") as persistent_trainer_module,
    ):
        tunnel_manager.sync_tunnel = AsyncMock()
        persistent_trainer_module.start = AsyncMock(side_effect=_blocks_forever)

        result = await asyncio.wait_for(
            RemoteTrainerService(session).create_remote_trainer(
                RemoteTrainerCreate(name="ssh-trainer", connection=_ssh())
            ),
            timeout=1,
        )
        await asyncio.wait_for(started.wait(), timeout=1)

    assert result == remote_trainer
    persistent_trainer_module.start.assert_awaited_once()


@pytest.mark.anyio
async def test_active_job_guard_matches_only_live_jobs_on_the_selected_trainer() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    trainer_id = uuid4()
    async with engine.begin() as connection:
        await connection.execute(text("CREATE TABLE jobs (id TEXT, type TEXT, status TEXT, payload JSON)"))
        for job_id, target, status in (("old", trainer_id, "completed"), ("other", uuid4(), "running")):
            await connection.execute(
                text("INSERT INTO jobs VALUES (:id, 'training', :status, :payload)"),
                {"id": job_id, "status": status, "payload": f'{{"remote_trainer_id":"{target}"}}'},
            )
    async with AsyncSession(engine) as session:
        service = RemoteTrainerService(session)
        await service._require_no_active_jobs(trainer_id)
        await session.execute(
            text("INSERT INTO jobs VALUES ('queued', 'training', 'pending', :payload)"),
            {"payload": f'{{"remote_trainer_id":"{trainer_id}"}}'},
        )
        with pytest.raises(ResourceInUseError):
            await service._require_no_active_jobs(trainer_id)
    await engine.dispose()


@pytest.mark.anyio
async def test_delete_rejects_ssh_trainer_with_running_or_queued_job() -> None:
    session = _session()
    session.execute.return_value = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = "active-job"
    trainer = _remote_trainer(_ssh(), name="gpu")
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.persistent_trainer.stop", new_callable=AsyncMock) as stop,
        pytest.raises(ResourceInUseError),
    ):
        await RemoteTrainerService(session).delete_remote_trainer(trainer.id)

    stop.assert_not_awaited()
    repository.delete_by_id.assert_not_called()


@pytest.mark.anyio
async def test_update_ssh_port_restarts_container_but_keeps_its_volume() -> None:
    session = _session()
    session.execute.return_value = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = None
    trainer = _remote_trainer(_ssh(), name="gpu")
    new_connection = _ssh(remote_port=8002)
    updated = trainer.model_copy(update={"connection": new_connection})
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)
    repository.list_ordered = AsyncMock(return_value=[trainer])
    repository.update = AsyncMock(return_value=updated)

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.remote_trainer_tunnel_manager.sync_tunnel", new_callable=AsyncMock),
        patch(f"{MODULE}.persistent_trainer.stop", new_callable=AsyncMock) as stop,
        patch.object(RemoteTrainerService, "_start_persistent_trainer_in_background") as start,
    ):
        await RemoteTrainerService(session).update_remote_trainer(
            trainer.id, RemoteTrainerUpdate(connection=new_connection)
        )

    stop.assert_awaited_once_with(trainer, remove_volume=False)
    start.assert_called_once_with(updated, None)


@pytest.mark.anyio
async def test_switching_ssh_trainer_to_direct_removes_old_container_and_volume() -> None:
    session = _session()
    session.execute.return_value = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = None
    trainer = _remote_trainer(_ssh(), name="gpu")
    updated = RemoteTrainer(id=trainer.id, name="gpu", connection=_direct())
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)
    repository.list_ordered = AsyncMock(return_value=[trainer])
    repository.update = AsyncMock(return_value=updated)

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.remote_trainer_tunnel_manager.sync_tunnel", new_callable=AsyncMock),
        patch(f"{MODULE}.persistent_trainer.stop", new_callable=AsyncMock) as stop,
    ):
        await RemoteTrainerService(session).update_remote_trainer(trainer.id, RemoteTrainerUpdate(connection=_direct()))

    stop.assert_awaited_once_with(trainer, remove_volume=True)


@pytest.mark.anyio
async def test_update_ssh_connection_rejects_active_job() -> None:
    session = _session()
    session.execute.return_value = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = "active-job"
    trainer = _remote_trainer(_ssh(), name="gpu")
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        pytest.raises(ResourceInUseError),
    ):
        await RemoteTrainerService(session).update_remote_trainer(
            trainer.id, RemoteTrainerUpdate(connection=_ssh(remote_port=8002))
        )

    repository.update.assert_not_called()


@pytest.mark.anyio
async def test_delete_cancels_an_inflight_launch_before_stopping_container() -> None:
    session = _session()
    session.execute.return_value = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = None
    trainer = _remote_trainer(_ssh(), name="gpu")
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=trainer)
    repository.delete_by_id = AsyncMock()
    started = asyncio.Event()
    canceled = asyncio.Event()

    async def _launch(*_args: object) -> None:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            canceled.set()
            raise

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.persistent_trainer.start", new_callable=AsyncMock, side_effect=_launch),
        patch(f"{MODULE}.persistent_trainer.stop", new_callable=AsyncMock) as stop,
        patch(f"{MODULE}.remote_trainer_tunnel_manager.stop_tunnel", new_callable=AsyncMock),
    ):
        RemoteTrainerService._start_persistent_trainer_in_background(trainer, None)
        await asyncio.wait_for(started.wait(), timeout=1)
        await RemoteTrainerService(session).delete_remote_trainer(trainer.id)

    assert canceled.is_set()
    stop.assert_awaited_once_with(trainer)
    repository.delete_by_id.assert_awaited_once_with(trainer.id)


@pytest.mark.anyio
async def test_delete_stops_the_tunnel_manager() -> None:
    session = _session()
    remote_trainer = _remote_trainer()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_trainer)
    repository.delete_by_id = AsyncMock()

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.remote_trainer_tunnel_manager") as tunnel_manager,
    ):
        tunnel_manager.stop_tunnel = AsyncMock()
        await RemoteTrainerService(session).delete_remote_trainer(remote_trainer.id)

    tunnel_manager.stop_tunnel.assert_awaited_once_with(remote_trainer.id)
