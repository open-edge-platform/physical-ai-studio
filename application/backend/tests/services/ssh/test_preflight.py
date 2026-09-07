# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the two-tier SSH preflight.

The fake transport matches remote commands by prefix and returns canned results,
so every check can be driven independently without an SSH server. Three
properties are load-bearing:

* ``test_tier1_never_*`` prove Tier 1 does no image work. It gates a save
  request, so a multi-gigabyte pull inside it would be a timeout waiting to
  happen - and the router's tests depend on it.
* ``test_tier1_gpu_busy_*`` prove a busy GPU is a non-blocking ``WARNING``. A
  user must be able to register or edit a target while a job runs on it.
* ``test_tier1_*_host_key`` prove an unknown key and a changed key produce
  different reason codes and never mark authentication as failed - verification
  happens before authentication is attempted.
"""

import re
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from exceptions import (
    SshAgentRequiredError,
    SshAuthenticationError,
    SshConnectionError,
    SshHostAliasNotFoundError,
    SshHostKeyMismatchError,
    SshHostKeyUnknownError,
)
from schemas.hardware import DeviceType
from schemas.remote_server import RemoteServer
from schemas.ssh_preflight import CheckKey, CheckOutcome, PreflightTier
from services.ssh import preflight as preflight_module
from services.ssh.preflight import (
    METHOD_NVIDIA_SMI,
    METHOD_RENDER_NODE,
    METHOD_XPU_SMI,
    PROTOCOL_LABEL,
    REASON_AGENT_REQUIRED,
    REASON_ALIAS_NOT_FOUND,
    REASON_AUTH_FAILED,
    REASON_COMMAND_FAILED,
    REASON_DEVICE_UNAVAILABLE,
    REASON_DOCKER_UNAVAILABLE,
    REASON_DRIVER_MISSING,
    REASON_GPU_BUSY,
    REASON_HOST_KEY_MISMATCH,
    REASON_HOST_KEY_UNKNOWN,
    REASON_IMAGE_PULLING,
    REASON_IMAGE_UNRESOLVED,
    REASON_INSUFFICIENT_DISK,
    REASON_NO_SIGNAL,
    REASON_PROTOCOL_MISMATCH,
    REASON_PROTOCOL_TAG_UNRESOLVED,
    REASON_PROTOCOL_UNKNOWN,
    REASON_REGISTRY_UNREACHABLE,
    REASON_TOOL_MISSING,
    REASON_UNPARSEABLE_OUTPUT,
    REASON_UNREACHABLE,
    registry_probe_url,
    reset_transport_factory,
    run_tier1_preflight,
    run_tier2_preflight,
    trainer_image_ref,
)
from services.ssh.transport import CommandFailure, CommandResult

ALIAS = "gpu-box"
_REGISTRY = "ghcr.io/open-edge-platform"
_CUDA_IMAGE = f"{_REGISTRY}/physicalai-trainer-cuda:protocol-1"
_CUDA_LATEST = f"{_REGISTRY}/physicalai-trainer-cuda:latest"

_DF_HEADER = "Filesystem 1B-blocks Used Available Capacity Mounted on\n"
# 80 GiB available, comfortably above the 60 GiB default requirement.
_PLENTY_OF_DISK = f"{_DF_HEADER}/dev/sda1 900000000000 100000000000 85899345920 12% /var/lib/docker\n"
# 1 GiB available.
_ALMOST_NO_DISK = f"{_DF_HEADER}/dev/sda1 900000000000 890000000000 1073741824 99% /var/lib/docker\n"

# Representative `xpu-smi discovery` output: an ASCII box-drawing table, not one
# plain line. `first_line()` on this would return the top border, not a device
# name - the regression `_xpu_discovery_detail` guards against.
_XPU_DISCOVERY_TABLE = (
    "+-----------+--------------------------------------------------+\n"
    "| Device ID | Device Information                                |\n"
    "+-----------+--------------------------------------------------+\n"
    "| 0         | Device Name: Intel(R) Data Center GPU Max 1100    |\n"
    "|           | Vendor Name: Intel(R) Corporation                 |\n"
    "|           | PCI BDF Address: 0000:29:00.0                     |\n"
    "+-----------+--------------------------------------------------+\n"
)


def _server(device_type: DeviceType = DeviceType.CUDA, alias: str = ALIAS) -> RemoteServer:
    return RemoteServer(
        id=uuid4(),
        name="Lab GPU box",
        ssh_host_alias=alias,
        device_type=device_type,
    )


class FakeTransport:
    """Records every command and answers from a prefix-matched script.

    Matching is by command prefix rather than exact equality, so a test only has
    to name the part of a command it cares about.
    """

    def __init__(
        self,
        script: dict[str, CommandResult] | None = None,
        connect_error: Exception | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.script = script or {}
        self.connect_error = connect_error
        self.fail_after = fail_after
        self.commands: list[tuple[str, ...]] = []
        self.connect_count = 0
        self.closed = False

    async def connect(self) -> None:
        self.connect_count += 1
        if self.connect_error is not None:
            raise self.connect_error

    async def close(self) -> None:
        self.closed = True

    async def run_command(self, argv, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
        self.commands.append(tuple(argv))
        if self.fail_after is not None and len(self.commands) > self.fail_after:
            raise SshConnectionError(ALIAS, reason="connection_lost")

        joined = " ".join(argv)
        for prefix, result in self.script.items():
            if joined.startswith(prefix):
                return result
        return _fail(f"unscripted command: {joined}")

    def ran(self, fragment: str) -> bool:
        return any(fragment in " ".join(argv) for argv in self.commands)


def _ok(stdout: str = "") -> CommandResult:
    return CommandResult(argv=(), command="", exit_status=0, stdout=stdout)


def _fail(stderr: str = "command failed", exit_status: int = 1) -> CommandResult:
    return CommandResult(argv=(), command="", exit_status=exit_status, stderr=stderr)


def _timed_out() -> CommandResult:
    return CommandResult(argv=(), command="", exit_status=124, failure=CommandFailure.TIMEOUT)


def _healthy_cuda_script() -> dict[str, CommandResult]:
    """A script where every Tier 1 check passes on an idle CUDA host."""
    return {
        "docker version": _ok("27.3.1\n"),
        "df -B1 -P /var/lib/docker": _ok(_PLENTY_OF_DISK),
        "nvidia-smi --query-gpu": _ok("NVIDIA RTX 6000 Ada Generation, 550.90.07\n"),
        "nvidia-smi --query-compute-apps": _ok("\n"),
        "curl --head": _ok(""),
    }


def _healthy_tier2_script() -> dict[str, CommandResult]:
    return {
        f"docker manifest inspect {_CUDA_IMAGE}": _ok('{"schemaVersion": 2}'),
        "cosign version": _ok("v2.4.1"),
        "cosign verify": _ok("Verified OK"),
        "docker run": _ok("True\n"),
        "docker image inspect": _ok("1\n"),
    }


@pytest.fixture
def install_transport(monkeypatch):
    """Install a fake transport and return it, restoring the real one after."""

    def install(transport: FakeTransport) -> FakeTransport:
        monkeypatch.setattr(preflight_module, "transport_factory", lambda _alias: transport)
        return transport

    yield install
    reset_transport_factory()


@pytest.fixture(autouse=True)
def resolvable_alias(monkeypatch, tmp_path):
    """Make the alias resolve, without touching the developer's real SSH config."""
    from schemas.remote_server import ResolvedSshHost

    def fake_resolve(_config_path, alias: str) -> ResolvedSshHost:
        return ResolvedSshHost(alias=alias, hostname=f"{alias}.example.com", port=22, user="tester", found=True)

    monkeypatch.setattr(preflight_module, "resolve_alias", fake_resolve)


def _outcome(result, key: CheckKey) -> CheckOutcome:
    check = result.check(key)
    assert check is not None, f"missing check: {key}"
    return check.outcome


def _reason(result, key: CheckKey) -> str | None:
    check = result.check(key)
    assert check is not None, f"missing check: {key}"
    return check.reason_code


def _detail_for(result, key: CheckKey) -> str | None:
    check = result.check(key)
    assert check is not None, f"missing check: {key}"
    return check.detail


# --------------------------------------------------------------------------- #
# Tier 1: result shape                                                        #
# --------------------------------------------------------------------------- #

_TIER1_KEYS = (
    CheckKey.ALIAS_RESOLVED,
    CheckKey.REACHABLE,
    CheckKey.AUTHENTICATED,
    CheckKey.HOST_KEY_VERIFIED,
    CheckKey.DOCKER_USABLE,
    CheckKey.DISK_SPACE,
    CheckKey.DRIVER_PRESENT,
    CheckKey.REGISTRY_REACHABLE,
    CheckKey.GPU_FREE,
)


async def test_tier1_healthy_cuda_server_passes_every_check(install_transport) -> None:
    install_transport(FakeTransport(_healthy_cuda_script()))

    result = await run_tier1_preflight(_server())

    assert result.passed is True
    assert [check.outcome for check in result.checks] == [CheckOutcome.PASSED] * len(_TIER1_KEYS)


async def test_tier1_always_reports_every_check(install_transport) -> None:
    # The UI renders a fixed list of checks, so a tier that ended early must still
    # account for every key rather than silently omitting some.
    install_transport(FakeTransport(connect_error=SshConnectionError(ALIAS, reason="timeout")))

    result = await run_tier1_preflight(_server())

    assert {check.key for check in result.checks} == set(_TIER1_KEYS)


async def test_tier1_result_is_tagged_and_timed(install_transport) -> None:
    install_transport(FakeTransport(_healthy_cuda_script()))
    server = _server()

    result = await run_tier1_preflight(server)

    assert result.tiers_run == [PreflightTier.TIER_1]
    assert result.remote_server_id == server.id
    assert result.latency_ms is not None
    assert result.checked_at.tzinfo is not None
    assert result.checked_at <= datetime.now(UTC)
    assert all(check.tier is PreflightTier.TIER_1 for check in result.checks)
    assert all(check.duration_ms is not None for check in result.checks)


async def test_tier1_closes_the_transport(install_transport) -> None:
    transport = install_transport(FakeTransport(_healthy_cuda_script()))

    await run_tier1_preflight(_server())

    assert transport.closed is True


async def test_tier1_opens_exactly_one_connection(install_transport) -> None:
    # One connection for all the probes: reconnecting per check would multiply the
    # handshake cost inside a request that has to stay bounded.
    transport = install_transport(FakeTransport(_healthy_cuda_script()))

    await run_tier1_preflight(_server())

    assert transport.connect_count == 1


# --------------------------------------------------------------------------- #
# Tier 1: the no-image-work guarantee                                         #
# --------------------------------------------------------------------------- #


async def test_tier1_never_pulls_an_image(install_transport) -> None:
    transport = install_transport(FakeTransport(_healthy_cuda_script()))

    await run_tier1_preflight(_server())

    assert not transport.ran("docker pull")


async def test_tier1_never_inspects_a_manifest(install_transport) -> None:
    # Not even a manifest inspect: that is Tier 2's IMAGE_RESOLVED check, and it
    # authenticates against the registry.
    transport = install_transport(FakeTransport(_healthy_cuda_script()))

    await run_tier1_preflight(_server())

    assert not transport.ran("docker manifest")
    assert not transport.ran("docker image inspect")


async def test_tier1_never_starts_a_container(install_transport) -> None:
    transport = install_transport(FakeTransport(_healthy_cuda_script()))

    await run_tier1_preflight(_server())

    assert not transport.ran("docker run")


async def test_tier1_registry_check_is_a_head_request(install_transport) -> None:
    transport = install_transport(FakeTransport(_healthy_cuda_script()))

    await run_tier1_preflight(_server())

    curl = next(argv for argv in transport.commands if argv[0] == "curl")
    assert "--head" in curl
    assert curl[-1] == "https://ghcr.io/v2/"


async def test_tier1_runs_no_tier2_check(install_transport) -> None:
    install_transport(FakeTransport(_healthy_cuda_script()))

    result = await run_tier1_preflight(_server())

    tier2_keys = {
        CheckKey.IMAGE_RESOLVED,
        CheckKey.IMAGE_SIGNATURE,
        CheckKey.CONTAINER_DEVICE_PROBE,
        CheckKey.PROTOCOL_COMPATIBLE,
    }
    assert tier2_keys.isdisjoint({check.key for check in result.checks})
    assert PreflightTier.TIER_2 not in result.tiers_run


# --------------------------------------------------------------------------- #
# Tier 1: alias and device gating                                             #
# --------------------------------------------------------------------------- #


async def test_tier1_unresolvable_alias_fails_without_connecting(install_transport, monkeypatch) -> None:
    from schemas.remote_server import ResolvedSshHost

    monkeypatch.setattr(
        preflight_module,
        "resolve_alias",
        lambda _path, alias: ResolvedSshHost(alias=alias, found=False),
    )
    transport = install_transport(FakeTransport(_healthy_cuda_script()))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.ALIAS_RESOLVED) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.ALIAS_RESOLVED) == REASON_ALIAS_NOT_FOUND
    assert transport.connect_count == 0
    assert result.passed is False


async def test_tier1_unresolvable_alias_skips_the_remaining_checks(install_transport, monkeypatch) -> None:
    from schemas.remote_server import ResolvedSshHost

    monkeypatch.setattr(
        preflight_module,
        "resolve_alias",
        lambda _path, alias: ResolvedSshHost(alias=alias, found=False),
    )
    install_transport(FakeTransport())

    result = await run_tier1_preflight(_server())

    skipped = [check.key for check in result.checks if check.outcome is CheckOutcome.SKIPPED]
    assert set(skipped) == set(_TIER1_KEYS) - {CheckKey.ALIAS_RESOLVED}


async def test_tier1_rejects_an_unsupported_device_type(install_transport) -> None:
    # No trainer image exists for CPU, so the driver check cannot be performed and
    # must not silently pass. Constructed via model_construct because the schema
    # rejects this on save - only a pre-validation record can reach here.
    transport = install_transport(FakeTransport(_healthy_cuda_script()))
    server = RemoteServer.model_construct(
        id=uuid4(),
        name="Old record",
        ssh_host_alias=ALIAS,
        device_type=DeviceType.CPU,
    )

    result = await run_tier1_preflight(server)

    assert _outcome(result, CheckKey.DRIVER_PRESENT) is CheckOutcome.FAILED
    assert result.passed is False
    assert transport.connect_count == 0


# --------------------------------------------------------------------------- #
# Tier 1: connection-stage failures                                           #
# --------------------------------------------------------------------------- #


async def test_tier1_unknown_host_key_is_reported_as_unknown(install_transport) -> None:
    install_transport(FakeTransport(connect_error=SshHostKeyUnknownError(ALIAS)))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.HOST_KEY_VERIFIED) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.HOST_KEY_VERIFIED) == REASON_HOST_KEY_UNKNOWN
    assert result.passed is False


async def test_tier1_changed_host_key_is_reported_as_a_mismatch(install_transport) -> None:
    # A different reason code from the unknown case, because the remedies differ:
    # accepting a fingerprint is right for one and dangerous for the other.
    install_transport(FakeTransport(connect_error=SshHostKeyMismatchError(ALIAS)))

    result = await run_tier1_preflight(_server())

    assert _reason(result, CheckKey.HOST_KEY_VERIFIED) == REASON_HOST_KEY_MISMATCH


@pytest.mark.parametrize("error", [SshHostKeyUnknownError(ALIAS), SshHostKeyMismatchError(ALIAS)])
async def test_tier1_host_key_failure_does_not_blame_authentication(install_transport, error: Exception) -> None:
    # Verification happens before authentication is attempted, so reporting
    # AUTHENTICATED as failed would send the user chasing the wrong problem.
    install_transport(FakeTransport(connect_error=error))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.AUTHENTICATED) is CheckOutcome.SKIPPED
    assert _outcome(result, CheckKey.REACHABLE) is CheckOutcome.PASSED


async def test_tier1_authentication_failure_reports_a_reachable_host(install_transport) -> None:
    # The server answered and rejected the identity, so reachability is proven.
    install_transport(FakeTransport(connect_error=SshAuthenticationError(ALIAS)))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.REACHABLE) is CheckOutcome.PASSED
    assert _outcome(result, CheckKey.HOST_KEY_VERIFIED) is CheckOutcome.PASSED
    assert _outcome(result, CheckKey.AUTHENTICATED) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.AUTHENTICATED) == REASON_AUTH_FAILED


async def test_tier1_passphrase_protected_key_without_an_agent(install_transport) -> None:
    install_transport(FakeTransport(connect_error=SshAgentRequiredError(ALIAS)))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.AUTHENTICATED) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.AUTHENTICATED) == REASON_AGENT_REQUIRED
    assert result.passed is False


async def test_tier1_unreachable_host_skips_rather_than_fails_dependent_checks(install_transport) -> None:
    # Studio never reached the host, so it must not claim Docker is broken there.
    install_transport(FakeTransport(connect_error=SshConnectionError(ALIAS, reason="timeout")))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.REACHABLE) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.REACHABLE) == REASON_UNREACHABLE
    for key in (CheckKey.DOCKER_USABLE, CheckKey.DISK_SPACE, CheckKey.DRIVER_PRESENT, CheckKey.REGISTRY_REACHABLE):
        assert _outcome(result, key) is CheckOutcome.SKIPPED


async def test_tier1_alias_not_found_at_connect_time_is_handled(install_transport) -> None:
    # The config can change between resolution and the dial.
    install_transport(FakeTransport(connect_error=SshHostAliasNotFoundError(ALIAS)))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.REACHABLE) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.REACHABLE) == REASON_ALIAS_NOT_FOUND


async def test_tier1_connection_lost_midway_skips_the_rest(install_transport) -> None:
    # Docker answered, then the link dropped: the remaining checks are unknown,
    # not failed.
    transport = install_transport(FakeTransport(_healthy_cuda_script(), fail_after=1))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.DOCKER_USABLE) is CheckOutcome.PASSED
    assert _outcome(result, CheckKey.DISK_SPACE) is CheckOutcome.SKIPPED
    assert _reason(result, CheckKey.DISK_SPACE) == REASON_UNREACHABLE
    assert _outcome(result, CheckKey.GPU_FREE) is CheckOutcome.SKIPPED
    assert transport.closed is True


async def test_tier1_a_dropped_connection_never_blocks_on_gpu_free(install_transport) -> None:
    install_transport(FakeTransport(_healthy_cuda_script(), fail_after=1))

    result = await run_tier1_preflight(_server())

    gpu_free = result.check(CheckKey.GPU_FREE)
    assert gpu_free is not None
    assert gpu_free.blocking is False


# --------------------------------------------------------------------------- #
# Tier 1: Docker                                                              #
# --------------------------------------------------------------------------- #


async def test_tier1_docker_socket_denied_fails_the_docker_check(install_transport) -> None:
    script = _healthy_cuda_script()
    script["docker version"] = _fail("permission denied while trying to connect to the Docker daemon socket")
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.DOCKER_USABLE) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.DOCKER_USABLE) == REASON_DOCKER_UNAVAILABLE
    assert result.passed is False


async def test_tier1_docker_check_queries_the_server_version(install_transport) -> None:
    # The client binary existing says nothing about the socket being reachable by
    # this user, which is the failure this check is for.
    transport = install_transport(FakeTransport(_healthy_cuda_script()))

    await run_tier1_preflight(_server())

    assert transport.ran("{{.Server.Version}}")


async def test_tier1_docker_failure_detail_is_sanitized(install_transport) -> None:
    script = _healthy_cuda_script()
    script["docker version"] = _fail("\x1b[31mdenied\x1b[0m\x00")
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    check = result.check(CheckKey.DOCKER_USABLE)
    assert check is not None
    assert check.detail == "denied"


async def test_tier1_docker_timeout_fails_the_docker_check(install_transport) -> None:
    script = _healthy_cuda_script()
    script["docker version"] = _timed_out()
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.DOCKER_USABLE) is CheckOutcome.FAILED


# --------------------------------------------------------------------------- #
# Tier 1: disk space                                                          #
# --------------------------------------------------------------------------- #


async def test_tier1_insufficient_disk_fails(install_transport) -> None:
    script = _healthy_cuda_script()
    script["df -B1 -P /var/lib/docker"] = _ok(_ALMOST_NO_DISK)
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.DISK_SPACE) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.DISK_SPACE) == REASON_INSUFFICIENT_DISK
    assert result.passed is False


async def test_tier1_disk_check_falls_back_to_the_root_filesystem(install_transport) -> None:
    # A rootless or relocated Docker data root means /var/lib/docker need not
    # exist; that is not a failure.
    script = _healthy_cuda_script()
    script["df -B1 -P /var/lib/docker"] = _fail("No such file or directory")
    script["df -B1 -P /"] = _ok(_PLENTY_OF_DISK)
    transport = install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.DISK_SPACE) is CheckOutcome.PASSED
    assert transport.ran("df -B1 -P /")


async def test_tier1_unparseable_df_output_fails_rather_than_guesses(install_transport) -> None:
    script = _healthy_cuda_script()
    script["df -B1 -P /var/lib/docker"] = _ok("some totally unexpected output\n")
    script["df -B1 -P /"] = _ok("some totally unexpected output\n")
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.DISK_SPACE) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.DISK_SPACE) == REASON_UNPARSEABLE_OUTPUT


async def test_tier1_disk_detail_reports_free_and_required_space(install_transport) -> None:
    script = _healthy_cuda_script()
    script["df -B1 -P /var/lib/docker"] = _ok(_ALMOST_NO_DISK)
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    check = result.check(CheckKey.DISK_SPACE)
    assert check is not None
    assert check.detail is not None
    assert "1.0 GiB free" in check.detail
    assert "required" in check.detail


async def test_tier1_both_df_probes_failing_fails_the_check(install_transport) -> None:
    script = _healthy_cuda_script()
    script["df -B1 -P /var/lib/docker"] = _fail("df: not found")
    script["df -B1 -P /"] = _fail("df: not found")
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.DISK_SPACE) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.DISK_SPACE) == REASON_COMMAND_FAILED


# --------------------------------------------------------------------------- #
# Tier 1: driver detection                                                    #
# --------------------------------------------------------------------------- #


async def test_tier1_cuda_driver_detected_via_nvidia_smi(install_transport) -> None:
    install_transport(FakeTransport(_healthy_cuda_script()))

    result = await run_tier1_preflight(_server())

    check = result.check(CheckKey.DRIVER_PRESENT)
    assert check is not None
    assert check.outcome is CheckOutcome.PASSED
    assert check.method == METHOD_NVIDIA_SMI
    assert check.detail == "NVIDIA RTX 6000 Ada Generation, 550.90.07"


async def test_tier1_missing_nvidia_smi_fails_the_driver_check(install_transport) -> None:
    script = _healthy_cuda_script()
    script["nvidia-smi --query-gpu"] = _fail("nvidia-smi: command not found", exit_status=127)
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.DRIVER_PRESENT) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.DRIVER_PRESENT) == REASON_DRIVER_MISSING
    assert result.passed is False


async def test_tier1_cuda_host_is_never_probed_with_xpu_tools(install_transport) -> None:
    transport = install_transport(FakeTransport(_healthy_cuda_script()))

    await run_tier1_preflight(_server())

    assert not transport.ran("xpu-smi")


async def test_tier1_xpu_driver_detected_via_xpu_smi(install_transport) -> None:
    # Real `xpu-smi discovery` output is an ASCII box-drawing table, not one
    # plain line - `first_line()` would return the top border instead of a
    # device name (see `_xpu_discovery_detail`).
    script = {
        "docker version": _ok("27.3.1"),
        "df -B1 -P /var/lib/docker": _ok(_PLENTY_OF_DISK),
        "xpu-smi discovery": _ok(_XPU_DISCOVERY_TABLE),
        "xpu-smi stats": _ok("GPU Memory Used (MiB), 1024, GPU Memory Total (MiB), 49152"),
        "curl --head": _ok(""),
    }
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server(DeviceType.XPU))

    check = result.check(CheckKey.DRIVER_PRESENT)
    assert check is not None
    assert check.outcome is CheckOutcome.PASSED
    assert check.method == METHOD_XPU_SMI
    assert check.detail == "Intel(R) Data Center GPU Max 1100"
    assert not check.detail.startswith("+")


async def test_tier1_xpu_driver_falls_back_to_the_render_node(install_transport) -> None:
    # xpu-smi is frequently absent on hosts whose XPUs work, so requiring it would
    # reject valid targets.
    script = {
        "docker version": _ok("27.3.1"),
        "df -B1 -P /var/lib/docker": _ok(_PLENTY_OF_DISK),
        "xpu-smi discovery": _fail("xpu-smi: command not found", exit_status=127),
        "sh -c": _ok(""),
        "curl --head": _ok(""),
    }
    transport = install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server(DeviceType.XPU))

    check = result.check(CheckKey.DRIVER_PRESENT)
    assert check is not None
    assert check.outcome is CheckOutcome.PASSED
    assert check.method == METHOD_RENDER_NODE
    assert transport.ran("/dev/dri/renderD")


async def test_tier1_render_node_probe_requires_the_intel_vendor_id(install_transport) -> None:
    # A render node alone is not enough: any DRM device creates one.
    script = {
        "docker version": _ok("27.3.1"),
        "df -B1 -P /var/lib/docker": _ok(_PLENTY_OF_DISK),
        "xpu-smi discovery": _fail("not found", exit_status=127),
        "sh -c": _ok(""),
        "curl --head": _ok(""),
    }
    transport = install_transport(FakeTransport(script))

    await run_tier1_preflight(_server(DeviceType.XPU))

    probe = next(argv for argv in transport.commands if argv[0] == "sh")
    assert "0x8086" in probe
    assert "/sys/class/drm/*/device/vendor" in " ".join(probe)


async def test_tier1_xpu_with_neither_signal_fails(install_transport) -> None:
    script = {
        "docker version": _ok("27.3.1"),
        "df -B1 -P /var/lib/docker": _ok(_PLENTY_OF_DISK),
        "xpu-smi discovery": _fail("not found", exit_status=127),
        "sh -c": _fail("", exit_status=1),
        "curl --head": _ok(""),
    }
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server(DeviceType.XPU))

    assert _outcome(result, CheckKey.DRIVER_PRESENT) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.DRIVER_PRESENT) == REASON_DRIVER_MISSING


# --------------------------------------------------------------------------- #
# Tier 1: registry reachability                                               #
# --------------------------------------------------------------------------- #


async def test_tier1_unreachable_registry_fails(install_transport) -> None:
    script = _healthy_cuda_script()
    script["curl --head"] = _fail("curl: (6) Could not resolve host: ghcr.io", exit_status=6)
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    assert _outcome(result, CheckKey.REGISTRY_REACHABLE) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.REGISTRY_REACHABLE) == REASON_REGISTRY_UNREACHABLE
    assert result.passed is False


async def test_tier1_registry_is_probed_from_the_remote_host(install_transport) -> None:
    # A registry reachable from the Studio machine but firewalled on the GPU box is
    # exactly the misconfiguration this catches, so the probe has to run there.
    transport = install_transport(FakeTransport(_healthy_cuda_script()))

    await run_tier1_preflight(_server())

    assert transport.ran("curl --head")


def test_registry_probe_url_uses_only_the_registry_host() -> None:
    assert registry_probe_url("ghcr.io/open-edge-platform") == "https://ghcr.io/v2/"
    assert registry_probe_url("registry.example.com:5000/team/sub") == "https://registry.example.com:5000/v2/"
    assert registry_probe_url("https://ghcr.io/open-edge-platform") == "https://ghcr.io/v2/"


# --------------------------------------------------------------------------- #
# Tier 1: GPU occupancy is reported, never blocking                           #
# --------------------------------------------------------------------------- #


async def test_tier1_gpu_busy_is_a_non_blocking_warning(install_transport) -> None:
    # The named requirement: a user must be able to register or edit a server while
    # a job is running on it, so a busy GPU is reported and never blocks a save.
    script = _healthy_cuda_script()
    script["nvidia-smi --query-compute-apps"] = _ok("12345\n12346\n")
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    gpu_free = result.check(CheckKey.GPU_FREE)
    assert gpu_free is not None
    assert gpu_free.outcome is CheckOutcome.WARNING
    assert gpu_free.outcome is not CheckOutcome.FAILED
    assert gpu_free.blocking is False
    assert gpu_free.reason_code == REASON_GPU_BUSY
    assert result.passed is True
    assert result.blocking_failures == []


async def test_tier1_gpu_free_check_is_never_blocking(install_transport) -> None:
    install_transport(FakeTransport(_healthy_cuda_script()))

    result = await run_tier1_preflight(_server())

    gpu_free = result.check(CheckKey.GPU_FREE)
    assert gpu_free is not None
    assert gpu_free.blocking is False


async def test_tier1_gpu_busy_detail_reports_the_process_count(install_transport) -> None:
    script = _healthy_cuda_script()
    script["nvidia-smi --query-compute-apps"] = _ok("111\n222\n333\n")
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    check = result.check(CheckKey.GPU_FREE)
    assert check is not None
    assert check.detail is not None
    assert "3" in check.detail


async def test_tier1_unqueryable_gpu_occupancy_is_skipped_not_failed(install_transport) -> None:
    script = _healthy_cuda_script()
    script["nvidia-smi --query-compute-apps"] = _fail("Unable to determine the device handle")
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    gpu_free = result.check(CheckKey.GPU_FREE)
    assert gpu_free is not None
    assert gpu_free.outcome is CheckOutcome.SKIPPED
    assert gpu_free.blocking is False
    assert result.passed is True


async def test_tier1_xpu_occupancy_warns_when_memory_is_mostly_used(install_transport) -> None:
    script = {
        "docker version": _ok("27.3.1"),
        "df -B1 -P /var/lib/docker": _ok(_PLENTY_OF_DISK),
        "xpu-smi discovery": _ok("Device 0"),
        "xpu-smi stats": _ok("40960, 49152"),
        "curl --head": _ok(""),
    }
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server(DeviceType.XPU))

    gpu_free = result.check(CheckKey.GPU_FREE)
    assert gpu_free is not None
    assert gpu_free.outcome is CheckOutcome.WARNING
    assert gpu_free.blocking is False
    assert result.passed is True


async def test_tier1_xpu_occupancy_without_a_signal_is_skipped(install_transport) -> None:
    # Reporting a busy accelerator as free would send a job straight into an
    # out-of-memory failure, so an unanswerable probe is SKIPPED, never PASSED.
    script = {
        "docker version": _ok("27.3.1"),
        "df -B1 -P /var/lib/docker": _ok(_PLENTY_OF_DISK),
        "xpu-smi discovery": _fail("not found", exit_status=127),
        "sh -c": _ok(""),
        "curl --head": _ok(""),
    }
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server(DeviceType.XPU))

    gpu_free = result.check(CheckKey.GPU_FREE)
    assert gpu_free is not None
    assert gpu_free.outcome is CheckOutcome.SKIPPED
    assert gpu_free.reason_code == REASON_NO_SIGNAL
    assert gpu_free.blocking is False


async def test_tier1_xpu_unparseable_stats_are_skipped(install_transport) -> None:
    script = {
        "docker version": _ok("27.3.1"),
        "df -B1 -P /var/lib/docker": _ok(_PLENTY_OF_DISK),
        "xpu-smi discovery": _ok("Device 0"),
        "xpu-smi stats": _ok("no numbers here"),
        "curl --head": _ok(""),
    }
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server(DeviceType.XPU))

    assert _outcome(result, CheckKey.GPU_FREE) is CheckOutcome.SKIPPED
    assert _reason(result, CheckKey.GPU_FREE) == REASON_NO_SIGNAL


async def test_tier1_multiple_failures_are_all_reported(install_transport) -> None:
    # The user should see every cause at once, not just the first one.
    script = _healthy_cuda_script()
    script["docker version"] = _fail("denied")
    script["nvidia-smi --query-gpu"] = _fail("not found", exit_status=127)
    script["curl --head"] = _fail("could not resolve host", exit_status=6)
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    failed = {check.key for check in result.blocking_failures}
    assert failed == {CheckKey.DOCKER_USABLE, CheckKey.DRIVER_PRESENT, CheckKey.REGISTRY_REACHABLE}


async def test_tier1_never_raises_for_a_failing_server(install_transport) -> None:
    install_transport(FakeTransport({}))

    result = await run_tier1_preflight(_server())

    assert result.passed is False


async def test_tier1_detail_never_carries_escape_sequences(install_transport) -> None:
    script = _healthy_cuda_script()
    script["docker version"] = _fail("\x1b]8;;https://evil.example.com\x07click me\x1b]8;;\x07")
    script["curl --head"] = _fail("\x1b[2Jcleared\x00")
    install_transport(FakeTransport(script))

    result = await run_tier1_preflight(_server())

    for check in result.checks:
        assert check.detail is None or not re.search(r"[\x00-\x08\x0b-\x1f]", check.detail)


# --------------------------------------------------------------------------- #
# Tier 2                                                                      #
# --------------------------------------------------------------------------- #

_TIER2_KEYS = (
    CheckKey.IMAGE_RESOLVED,
    CheckKey.IMAGE_SIGNATURE,
    CheckKey.CONTAINER_DEVICE_PROBE,
    CheckKey.PROTOCOL_COMPATIBLE,
)


async def test_tier2_healthy_server_passes_every_check(install_transport) -> None:
    install_transport(FakeTransport(_healthy_tier2_script()))

    result = await run_tier2_preflight(_server())

    assert result.passed is True
    assert {check.key for check in result.checks} == set(_TIER2_KEYS)
    assert result.tiers_run == [PreflightTier.TIER_2]
    assert all(check.tier is PreflightTier.TIER_2 for check in result.checks)


async def test_tier2_resolves_the_protocol_pinned_tag(install_transport) -> None:
    transport = install_transport(FakeTransport(_healthy_tier2_script()))

    result = await run_tier2_preflight(_server())

    assert transport.ran(f"docker manifest inspect {_CUDA_IMAGE}")
    check = result.check(CheckKey.IMAGE_RESOLVED)
    assert check is not None
    assert check.detail == _CUDA_IMAGE


async def test_tier2_honours_an_image_ref_hint(install_transport) -> None:
    hint = "ghcr.io/example/custom-trainer:dev"
    transport = install_transport(FakeTransport({f"docker manifest inspect {hint}": _ok("{}")}))

    result = await run_tier2_preflight(_server(), image_ref_hint=hint)

    assert transport.ran(f"docker manifest inspect {hint}")
    assert _outcome(result, CheckKey.IMAGE_RESOLVED) is CheckOutcome.PASSED


async def test_tier2_falls_back_to_latest_with_a_warning(install_transport) -> None:
    # A protocol bump can land in Studio before CI has published a matching image;
    # blocking every verification until then would be worse than saying which tag
    # was used.
    script = _healthy_tier2_script()
    script.pop(f"docker manifest inspect {_CUDA_IMAGE}")
    script[f"docker manifest inspect {_CUDA_LATEST}"] = _ok("{}")
    install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    check = result.check(CheckKey.IMAGE_RESOLVED)
    assert check is not None
    assert check.outcome is CheckOutcome.WARNING
    assert check.reason_code == REASON_PROTOCOL_TAG_UNRESOLVED
    assert check.detail is not None
    assert _CUDA_LATEST in check.detail
    assert result.passed is True


async def test_tier2_unresolvable_image_fails_and_skips_the_rest(install_transport) -> None:
    transport = install_transport(FakeTransport({}))

    result = await run_tier2_preflight(_server())

    assert _outcome(result, CheckKey.IMAGE_RESOLVED) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.IMAGE_RESOLVED) == REASON_IMAGE_UNRESOLVED
    for key in (CheckKey.IMAGE_SIGNATURE, CheckKey.CONTAINER_DEVICE_PROBE, CheckKey.PROTOCOL_COMPATIBLE):
        assert _outcome(result, key) is CheckOutcome.SKIPPED
    assert not transport.ran("docker run")


async def test_tier2_missing_cosign_is_skipped_not_failed(install_transport) -> None:
    # Signature verification here is defense in depth; the publish-time signature
    # is the primary control.
    script = _healthy_tier2_script()
    script["cosign version"] = _fail("cosign: command not found", exit_status=127)
    install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    check = result.check(CheckKey.IMAGE_SIGNATURE)
    assert check is not None
    assert check.outcome is CheckOutcome.SKIPPED
    assert check.reason_code == REASON_TOOL_MISSING
    assert check.blocking is False
    assert result.passed is True


async def test_tier2_failed_signature_verification_warns_without_blocking(install_transport) -> None:
    script = _healthy_tier2_script()
    script["cosign verify"] = _fail("no matching signatures")
    install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    check = result.check(CheckKey.IMAGE_SIGNATURE)
    assert check is not None
    assert check.outcome is CheckOutcome.WARNING
    assert check.blocking is False


async def test_tier2_device_probe_runs_a_one_shot_container(install_transport) -> None:
    # The authoritative check, and the reason Tier 2 exists: a driver on the host
    # says nothing about the runtime passing the device into a container.
    transport = install_transport(FakeTransport(_healthy_tier2_script()))

    result = await run_tier2_preflight(_server())

    run = next(argv for argv in transport.commands if argv[:2] == ("docker", "run"))
    assert "--rm" in run
    assert "--gpus" in run
    assert "torch.cuda.is_available()" in " ".join(run)
    assert _outcome(result, CheckKey.CONTAINER_DEVICE_PROBE) is CheckOutcome.PASSED


async def test_tier2_xpu_device_probe_passes_the_dri_device(install_transport) -> None:
    xpu_image = f"{_REGISTRY}/physicalai-trainer-xpu:protocol-1"
    script = {
        f"docker manifest inspect {xpu_image}": _ok("{}"),
        "cosign version": _fail("not found", exit_status=127),
        "sh -c stat -c %g": _ok("44\n"),
        "docker run": _ok("True"),
        "docker image inspect": _ok("1"),
    }
    transport = install_transport(FakeTransport(script))

    await run_tier2_preflight(_server(DeviceType.XPU))

    run = next(argv for argv in transport.commands if argv[:2] == ("docker", "run"))
    assert "/dev/dri" in run
    assert "torch.xpu.is_available()" in " ".join(run)
    # Without --group-add <render gid> the fixed non-root container user cannot
    # open the render node even though it is passed through by --device.
    assert "--group-add" in run
    assert "44" in run


async def test_resolve_render_group_gid_returns_none_when_unavailable(install_transport) -> None:
    from services.ssh.preflight import resolve_render_group_gid

    transport = install_transport(FakeTransport({}))

    gid = await resolve_render_group_gid(transport)

    assert gid is None


async def test_resolve_render_group_gid_parses_stat_output(install_transport) -> None:
    from services.ssh.preflight import resolve_render_group_gid

    transport = install_transport(FakeTransport({"sh -c stat -c %g": _ok("44\n")}))

    gid = await resolve_render_group_gid(transport)

    assert gid == "44"


async def test_tier2_device_unavailable_in_the_container_fails(install_transport) -> None:
    script = _healthy_tier2_script()
    script["docker run"] = _ok("False\n")
    install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    assert _outcome(result, CheckKey.CONTAINER_DEVICE_PROBE) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.CONTAINER_DEVICE_PROBE) == REASON_DEVICE_UNAVAILABLE
    assert result.passed is False


async def test_tier2_device_probe_failure_fails_the_check(install_transport) -> None:
    script = _healthy_tier2_script()
    script["docker run"] = _fail("could not select device driver with capabilities: [[gpu]]", exit_status=125)
    install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    assert _outcome(result, CheckKey.CONTAINER_DEVICE_PROBE) is CheckOutcome.FAILED


async def test_tier2_matching_protocol_version_passes(install_transport) -> None:
    transport = install_transport(FakeTransport(_healthy_tier2_script()))

    result = await run_tier2_preflight(_server(), protocol_version=1)

    assert _outcome(result, CheckKey.PROTOCOL_COMPATIBLE) is CheckOutcome.PASSED
    assert transport.ran(PROTOCOL_LABEL)


async def test_tier2_protocol_mismatch_fails(install_transport) -> None:
    script = _healthy_tier2_script()
    script["docker image inspect"] = _ok("7\n")
    install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server(), protocol_version=1)

    assert _outcome(result, CheckKey.PROTOCOL_COMPATIBLE) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.PROTOCOL_COMPATIBLE) == REASON_PROTOCOL_MISMATCH
    assert result.passed is False


async def test_tier2_an_unlabelled_image_fails_the_protocol_check(install_transport) -> None:
    script = _healthy_tier2_script()
    script["docker image inspect"] = _ok("<no value>\n")
    install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    assert _outcome(result, CheckKey.PROTOCOL_COMPATIBLE) is CheckOutcome.FAILED
    assert _reason(result, CheckKey.PROTOCOL_COMPATIBLE) == REASON_PROTOCOL_UNKNOWN


async def test_tier2_an_unparseable_protocol_label_fails(install_transport) -> None:
    script = _healthy_tier2_script()
    script["docker image inspect"] = _ok("v2-beta\n")
    install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    assert _reason(result, CheckKey.PROTOCOL_COMPATIBLE) == REASON_UNPARSEABLE_OUTPUT


async def test_tier2_uses_the_expected_protocol_version_parameter(install_transport) -> None:
    # The version arrives as a parameter, so this module never imports the trainer
    # package.
    script = _healthy_tier2_script()
    script[f"docker manifest inspect {_REGISTRY}/physicalai-trainer-cuda:protocol-3"] = _ok("{}")
    script["docker image inspect"] = _ok("3\n")
    transport = install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server(), protocol_version=3)

    assert transport.ran("physicalai-trainer-cuda:protocol-3")
    assert _outcome(result, CheckKey.PROTOCOL_COMPATIBLE) is CheckOutcome.PASSED


async def test_tier2_connect_failure_skips_every_check(install_transport) -> None:
    install_transport(FakeTransport(connect_error=SshConnectionError(ALIAS, reason="timeout")))

    result = await run_tier2_preflight(_server())

    assert {check.key for check in result.checks} == set(_TIER2_KEYS)
    assert all(check.outcome is CheckOutcome.SKIPPED for check in result.checks)
    assert result.passed is True


async def test_tier2_connection_lost_midway_skips_the_rest(install_transport) -> None:
    transport = install_transport(FakeTransport(_healthy_tier2_script(), fail_after=1))

    result = await run_tier2_preflight(_server())

    assert _outcome(result, CheckKey.IMAGE_RESOLVED) is CheckOutcome.PASSED
    assert _outcome(result, CheckKey.CONTAINER_DEVICE_PROBE) is CheckOutcome.SKIPPED
    assert _reason(result, CheckKey.CONTAINER_DEVICE_PROBE) == REASON_UNREACHABLE
    assert transport.closed is True


async def test_tier2_closes_the_transport(install_transport) -> None:
    transport = install_transport(FakeTransport(_healthy_tier2_script()))

    await run_tier2_preflight(_server())

    assert transport.closed is True


async def test_tier2_signature_check_is_never_blocking(install_transport) -> None:
    install_transport(FakeTransport(_healthy_tier2_script()))

    result = await run_tier2_preflight(_server())

    check = result.check(CheckKey.IMAGE_SIGNATURE)
    assert check is not None
    assert check.blocking is False


def test_trainer_image_ref_is_built_from_constants() -> None:
    assert trainer_image_ref(_REGISTRY, DeviceType.CUDA, "protocol-1") == _CUDA_IMAGE
    assert trainer_image_ref(f"{_REGISTRY}/", DeviceType.XPU, "latest").endswith("physicalai-trainer-xpu:latest")


async def test_tier2_device_probe_mid_pull_is_skipped_not_failed(install_transport) -> None:
    # Defense-in-depth for the race between `_image_present_locally` reporting
    # present and the `docker run` a few lines later - e.g. another process
    # evicting the image in between, forcing Docker to pull it inline. The raw
    # progress output must never read as the compute probe, or the device,
    # having failed.
    script = _healthy_tier2_script()
    script["docker run"] = _fail(
        "Unable to find image 'ghcr.io/open-edge-platform/physicalai-trainer-cuda:protocol-1' locally\n"
        "protocol-1: Pulling from open-edge-platform/physicalai-trainer-cuda\n"
        "26c307b5e35a: Pulling fs layer\n"
        "b07d7dc8cffa: Pulling fs layer\n"
    )
    install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    assert _outcome(result, CheckKey.CONTAINER_DEVICE_PROBE) is CheckOutcome.SKIPPED
    assert _reason(result, CheckKey.CONTAINER_DEVICE_PROBE) == REASON_IMAGE_PULLING
    assert result.passed is True


async def test_tier2_device_probe_mid_pull_also_skips_the_protocol_check(install_transport) -> None:
    # `docker image inspect` needs the image locally too, so running it right
    # after a mid-pull device probe would just produce a second, equally
    # misleading failure ("no protocol version") for the same root cause.
    script = _healthy_tier2_script()
    script["docker run"] = _fail("Unable to find image 'img' locally\nPulling from registry\n")
    install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    assert _outcome(result, CheckKey.PROTOCOL_COMPATIBLE) is CheckOutcome.SKIPPED
    assert _reason(result, CheckKey.PROTOCOL_COMPATIBLE) == REASON_IMAGE_PULLING


async def test_tier2_device_probe_starts_a_background_pull_when_image_is_absent(install_transport) -> None:
    # The primary path: the image simply is not cached locally yet (a cold
    # host, or a protocol bump CI has not published an image for). Rather than
    # `docker run` pulling it inline - tying the transfer to this check's
    # short timeout and to the SSH connection Tier 2 is about to close - the
    # pull is handed to a detached, `nohup`-backed process that keeps going
    # after this check returns.
    script = _healthy_tier2_script()
    script["docker image inspect"] = _fail("Error: No such image: img", exit_status=1)
    script["sh -c test -f"] = _fail("", exit_status=1)  # no pull already running
    script["sh -c nohup docker pull"] = _ok("")
    transport = install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    assert _outcome(result, CheckKey.CONTAINER_DEVICE_PROBE) is CheckOutcome.SKIPPED
    assert _reason(result, CheckKey.CONTAINER_DEVICE_PROBE) == REASON_IMAGE_PULLING
    assert "Started pulling" in (_detail_for(result, CheckKey.CONTAINER_DEVICE_PROBE) or "")
    assert _outcome(result, CheckKey.PROTOCOL_COMPATIBLE) is CheckOutcome.SKIPPED
    assert _reason(result, CheckKey.PROTOCOL_COMPATIBLE) == REASON_IMAGE_PULLING
    assert not transport.ran("docker run")
    assert transport.ran("nohup docker pull")
    assert result.passed is True


async def test_tier2_device_probe_does_not_start_a_second_pull_already_in_flight(install_transport) -> None:
    # Repeated "Test connection" clicks while a pull is downloading must not
    # each kick off a competing pull for the same image.
    script = _healthy_tier2_script()
    script["docker image inspect"] = _fail("Error: No such image: img", exit_status=1)
    script["sh -c test -f"] = _ok("")  # a pull is already running
    transport = install_transport(FakeTransport(script))

    result = await run_tier2_preflight(_server())

    assert _outcome(result, CheckKey.CONTAINER_DEVICE_PROBE) is CheckOutcome.SKIPPED
    assert "Still pulling" in (_detail_for(result, CheckKey.CONTAINER_DEVICE_PROBE) or "")
    assert not transport.ran("nohup docker pull")
