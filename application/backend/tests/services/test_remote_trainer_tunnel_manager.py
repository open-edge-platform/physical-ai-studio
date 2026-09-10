# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standing per-trainer SSH tunnel manager."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import services.remote_trainer_tunnel_manager as tunnel_manager
from core.security.ssh_network_exposure import SshFeatureAvailability
from schemas.remote_trainer import RemoteTrainer

MODULE = "services.remote_trainer_tunnel_manager"


def _trainer(*, ssh_host_alias: str | None = "training-box", local_port: int | None = 8001) -> RemoteTrainer:
    return RemoteTrainer(
        id=uuid4(),
        name="trainer",
        url="http://127.0.0.1:8001",
        ssh_host_alias=ssh_host_alias,
        ssh_remote_port=8001 if ssh_host_alias else None,
        ssh_local_port=local_port if ssh_host_alias else None,
    )


def _active_availability() -> SshFeatureAvailability:
    return SshFeatureAvailability(network_exposed=False)


def _inactive_availability() -> SshFeatureAvailability:
    return SshFeatureAvailability(network_exposed=True)


@pytest.fixture(autouse=True)
def _clear_tunnels():
    """Every test gets a clean module-level tunnel table."""
    tunnel_manager._tunnels.clear()
    yield
    tunnel_manager._tunnels.clear()


@pytest.mark.anyio
async def test_sync_tunnel_does_nothing_for_a_trainer_without_ssh_config() -> None:
    with patch(f"{MODULE}.SshTunnel") as tunnel_cls:
        await tunnel_manager.sync_tunnel(_trainer(ssh_host_alias=None, local_port=None))

    tunnel_cls.assert_not_called()


@pytest.mark.anyio
async def test_sync_tunnel_opens_a_tunnel_when_configured_and_feature_active() -> None:
    trainer = _trainer()
    tunnel = AsyncMock()
    tunnel.local_port = 8001

    with (
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=_active_availability()),
        patch(f"{MODULE}.SshTunnel", return_value=tunnel) as tunnel_cls,
    ):
        await tunnel_manager.sync_tunnel(trainer)

    tunnel_cls.assert_called_once()
    tunnel.open.assert_awaited_once()
    assert tunnel_manager._tunnels[trainer.id] is tunnel


@pytest.mark.anyio
async def test_sync_tunnel_is_a_noop_when_ssh_feature_is_inactive() -> None:
    trainer = _trainer()

    with (
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=_inactive_availability()),
        patch(f"{MODULE}.SshTunnel") as tunnel_cls,
    ):
        await tunnel_manager.sync_tunnel(trainer)

    tunnel_cls.assert_not_called()
    assert trainer.id not in tunnel_manager._tunnels


@pytest.mark.anyio
async def test_sync_tunnel_replaces_an_existing_tunnel() -> None:
    trainer = _trainer()
    old_tunnel = AsyncMock()
    tunnel_manager._tunnels[trainer.id] = old_tunnel
    new_tunnel = AsyncMock()
    new_tunnel.local_port = 8001

    with (
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=_active_availability()),
        patch(f"{MODULE}.SshTunnel", return_value=new_tunnel),
    ):
        await tunnel_manager.sync_tunnel(trainer)

    old_tunnel.close.assert_awaited_once()
    assert tunnel_manager._tunnels[trainer.id] is new_tunnel


@pytest.mark.anyio
async def test_stop_tunnel_closes_and_drops_it() -> None:
    trainer_id = uuid4()
    tunnel = AsyncMock()
    tunnel_manager._tunnels[trainer_id] = tunnel

    await tunnel_manager.stop_tunnel(trainer_id)

    tunnel.close.assert_awaited_once()
    assert trainer_id not in tunnel_manager._tunnels


@pytest.mark.anyio
async def test_stop_all_closes_every_tunnel() -> None:
    first, second = AsyncMock(), AsyncMock()
    tunnel_manager._tunnels[uuid4()] = first
    tunnel_manager._tunnels[uuid4()] = second

    await tunnel_manager.stop_all()

    first.close.assert_awaited_once()
    second.close.assert_awaited_once()
    assert tunnel_manager._tunnels == {}


@pytest.mark.anyio
async def test_start_all_skips_every_trainer_when_feature_inactive() -> None:
    with (
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=_inactive_availability()),
        patch(f"{MODULE}.SshTunnel") as tunnel_cls,
    ):
        await tunnel_manager.start_all([_trainer()])

    tunnel_cls.assert_not_called()


@pytest.mark.anyio
async def test_start_all_skips_trainers_without_ssh_config() -> None:
    with patch(f"{MODULE}.sync_tunnel", new=AsyncMock()) as sync:
        await tunnel_manager.start_all([_trainer(ssh_host_alias=None, local_port=None)])

    sync.assert_not_called()


@pytest.mark.anyio
async def test_open_tunnel_failure_is_swallowed_and_leaves_no_tunnel_registered() -> None:
    trainer = _trainer()
    tunnel = AsyncMock()
    tunnel.open.side_effect = RuntimeError("host unreachable")

    with (
        patch(f"{MODULE}.get_ssh_feature_availability", return_value=_active_availability()),
        patch(f"{MODULE}.SshTunnel", return_value=tunnel),
    ):
        await tunnel_manager.sync_tunnel(trainer)

    assert trainer.id not in tunnel_manager._tunnels
