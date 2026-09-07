# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SSH remote-trainer feature's fail-closed network-exposure policy."""

from __future__ import annotations

import pytest

from core.security.ssh_network_exposure import (
    evaluate_ssh_feature_availability,
    get_ssh_feature_availability,
    is_loopback_host,
)


class _Settings:
    """Minimal settings stand-in: only the two fields this module reads."""

    def __init__(self, *, ssh_remote_trainer_enabled: bool, host: str) -> None:
        self.ssh_remote_trainer_enabled = ssh_remote_trainer_enabled
        self.host = host


# --------------------------------------------------------------------------- #
# is_loopback_host                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.5", "::1", "localhost"])
def test_is_loopback_host_true_for_loopback_addresses(host: str) -> None:
    assert is_loopback_host(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "10.0.0.5"])
def test_is_loopback_host_false_for_bind_all_or_routable_addresses(host: str) -> None:
    assert is_loopback_host(host) is False


def test_is_loopback_host_fails_closed_for_an_unresolvable_hostname() -> None:
    assert is_loopback_host("this-host-does-not-resolve.invalid") is False


# --------------------------------------------------------------------------- #
# evaluate_ssh_feature_availability                                           #
# --------------------------------------------------------------------------- #


def test_disabled_feature_is_never_reported_as_network_exposed() -> None:
    """An off feature has nothing to expose, regardless of the bind address."""
    settings = _Settings(ssh_remote_trainer_enabled=False, host="0.0.0.0")

    availability = evaluate_ssh_feature_availability(settings)

    assert availability.enabled is False
    assert availability.network_exposed is False
    assert availability.active is False
    assert availability.reason is None


def test_enabled_feature_on_loopback_is_active() -> None:
    settings = _Settings(ssh_remote_trainer_enabled=True, host="127.0.0.1")

    availability = evaluate_ssh_feature_availability(settings)

    assert availability.active is True
    assert availability.network_exposed is False
    assert availability.reason is None


def test_enabled_feature_on_ipv6_loopback_is_active() -> None:
    settings = _Settings(ssh_remote_trainer_enabled=True, host="::1")

    availability = evaluate_ssh_feature_availability(settings)

    assert availability.active is True


def test_enabled_feature_on_bind_all_fails_closed() -> None:
    settings = _Settings(ssh_remote_trainer_enabled=True, host="0.0.0.0")

    availability = evaluate_ssh_feature_availability(settings)

    assert availability.enabled is True
    assert availability.network_exposed is True
    assert availability.active is False
    assert availability.reason is not None
    assert "loopback" in availability.reason


def test_bind_host_argument_overrides_settings_host() -> None:
    """A caller that knows the real bind address (e.g. a CLI flag) can override `settings.host`."""
    settings = _Settings(ssh_remote_trainer_enabled=True, host="127.0.0.1")

    availability = evaluate_ssh_feature_availability(settings, bind_host="0.0.0.0")

    assert availability.active is False
    assert availability.network_exposed is True


def test_reason_never_names_a_host_alias_or_container() -> None:
    """The reason can reach a pre-auth status endpoint or a startup log line."""
    settings = _Settings(ssh_remote_trainer_enabled=True, host="0.0.0.0")

    availability = evaluate_ssh_feature_availability(settings)

    assert availability.reason is not None
    for leaked_term in ("ssh_host_alias", "container", "gpu-box", "physicalai-trainer-"):
        assert leaked_term not in availability.reason


# --------------------------------------------------------------------------- #
# get_ssh_feature_availability caching                                        #
# --------------------------------------------------------------------------- #


def test_get_ssh_feature_availability_is_memoized_until_cache_cleared(monkeypatch) -> None:
    get_ssh_feature_availability.cache_clear()
    monkeypatch.setenv("SSH_REMOTE_TRAINER_ENABLED", "false")
    monkeypatch.setenv("HOST", "127.0.0.1")

    first = get_ssh_feature_availability()

    monkeypatch.setenv("SSH_REMOTE_TRAINER_ENABLED", "true")
    monkeypatch.setenv("HOST", "0.0.0.0")
    second = get_ssh_feature_availability()

    assert first == second  # still cached: settings changed, but the cache was not cleared

    get_ssh_feature_availability.cache_clear()
    third = get_ssh_feature_availability()
    assert third.enabled is True
    assert third.network_exposed is True

    get_ssh_feature_availability.cache_clear()
