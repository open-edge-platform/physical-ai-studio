# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Two-tier preflight for SSH-provisioned remote training servers.

Tier 1 is cheap and bounded, so it can gate a create/update request. Tier 2
resolves and inspects the trainer image and runs a one-shot GPU container, so it
is an explicit action and never runs inline in a request handler.

**Tier 1 performs no image work at all.** Its registry check is an
unauthenticated ``HEAD`` against the registry's ``/v2/`` API root, which
separates "this host cannot reach the registry" from "the pull failed midway" -
the latter being Tier 2's job. Nothing in :func:`run_tier1_preflight` resolves a
manifest, pulls a layer, or starts a container.

Every check is blocking except ``GPU_FREE``: a busy GPU is a transient state, not
a misconfiguration, and a user must be able to register or edit a target while a
job runs on it. A busy GPU reports ``WARNING``; a probe that cannot answer
reports ``SKIPPED``. Neither blocks.

Neither entry point raises for a server that fails its checks. A failure is a
``FAILED`` check, so the caller can surface every cause at once instead of only
the first one.

The transport is reached through :data:`transport_factory`, so a caller can
substitute a fake without patching ``asyncssh``.
"""

import hashlib
import re
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Final
from urllib.parse import urlsplit

from loguru import logger

from exceptions import (
    SshAgentRequiredError,
    SshAuthenticationError,
    SshConnectionError,
    SshHostAliasNotFoundError,
    SshHostKeyMismatchError,
    SshHostKeyUnknownError,
)
from schemas.hardware import DeviceType
from schemas.remote_server import SSH_SERVER_DEVICE_TYPES, RemoteServer
from schemas.ssh_preflight import CheckKey, CheckOutcome, PreflightCheck, PreflightResult, PreflightTier
from services.ssh.sanitize import sanitize_output
from services.ssh.transport import CommandResult, SshTransport, open_transport
from services.ssh_config_reader import resolve_alias
from settings import Settings, get_settings

# Both tiers obtain their transport here, so a test or a future provisioning path
# can substitute a fake without patching asyncssh. Use `set_transport_factory` to
# replace it.
transport_factory: Callable[[str], SshTransport] = open_transport

# Every Ssh* failure a connect attempt can raise. Listed once so both tiers catch
# exactly the same set and an unexpected exception type is never swallowed.
_CONNECT_FAILURES: Final = (
    SshHostAliasNotFoundError,
    SshHostKeyUnknownError,
    SshHostKeyMismatchError,
    SshAgentRequiredError,
    SshAuthenticationError,
    SshConnectionError,
)

# Reason codes are stable machine-readable strings the UI groups and labels on.
# Renaming one is a breaking change.
REASON_ALIAS_NOT_FOUND: Final = "alias_not_found"
REASON_HOST_KEY_UNKNOWN: Final = "host_key_unknown"
REASON_HOST_KEY_MISMATCH: Final = "host_key_mismatch"
REASON_AGENT_REQUIRED: Final = "ssh_agent_required"
REASON_AUTH_FAILED: Final = "authentication_failed"
REASON_UNREACHABLE: Final = "unreachable"
REASON_NOT_ATTEMPTED: Final = "not_attempted"
REASON_COMMAND_FAILED: Final = "command_failed"
REASON_DOCKER_UNAVAILABLE: Final = "docker_unavailable"
REASON_INSUFFICIENT_DISK: Final = "insufficient_disk"
REASON_UNPARSEABLE_OUTPUT: Final = "unparseable_output"
REASON_DRIVER_MISSING: Final = "driver_missing"
REASON_REGISTRY_UNREACHABLE: Final = "registry_unreachable"
REASON_GPU_BUSY: Final = "gpu_busy"
REASON_NO_SIGNAL: Final = "no_signal"
REASON_UNSUPPORTED_DEVICE: Final = "unsupported_device_type"
REASON_PROTOCOL_TAG_UNRESOLVED: Final = "protocol_tag_unresolved"
REASON_IMAGE_UNRESOLVED: Final = "image_unresolved"
REASON_TOOL_MISSING: Final = "tool_missing"
REASON_DEVICE_UNAVAILABLE: Final = "device_unavailable"
REASON_PROTOCOL_MISMATCH: Final = "protocol_mismatch"
REASON_PROTOCOL_UNKNOWN: Final = "protocol_unknown"
# The image was not yet present locally when the probe ran, so it was handed
# to a background pull. Distinct from REASON_DEVICE_UNAVAILABLE: this is a
# multi-gigabyte transfer still in flight, not a broken accelerator.
REASON_IMAGE_PULLING: Final = "image_pulling"

# Which probe answered a check, so the UI can show how a result was obtained.
METHOD_SSH_CONFIG: Final = "ssh-config"
METHOD_SSH_CONNECT: Final = "ssh-connect"
METHOD_DOCKER: Final = "docker"
METHOD_DF: Final = "df"
METHOD_CURL: Final = "curl"
METHOD_NVIDIA_SMI: Final = "nvidia-smi"
METHOD_XPU_SMI: Final = "xpu-smi"
METHOD_RENDER_NODE: Final = "render-node"
METHOD_DOCKER_MANIFEST: Final = "docker-manifest"
METHOD_COSIGN: Final = "cosign"
METHOD_CONTAINER: Final = "container"

# `df -B1 -P` prints one header line, then rows of:
# filesystem 1-blocks used available capacity mounted-on
_DF_AVAILABLE_COLUMN: Final = 3
_DF_MIN_COLUMNS: Final = 5

# Intel PCI vendor id, for the XPU render-node fallback probe.
_INTEL_VENDOR_ID: Final = "0x8086"

# Fraction of device memory in use above which an accelerator counts as busy.
# Used only where per-process attribution is unavailable (the XPU case).
_GPU_BUSY_MEMORY_FRACTION: Final = 0.3

# Cap on `detail`, so a check can never carry an essay from a remote host.
_DETAIL_MAX_CHARS: Final = 240

_BYTES_PER_GIB: Final = 1024**3


def _now() -> datetime:
    """Current UTC time, matching how the rest of the backend stamps records."""
    return datetime.now(UTC)


def _detail(text: str) -> str | None:
    """Sanitize and shorten operator-facing text taken from remote output."""
    cleaned = sanitize_output(text, max_line_chars=_DETAIL_MAX_CHARS, max_total_chars=_DETAIL_MAX_CHARS)
    return cleaned.strip() or None


def _failure_detail(result: CommandResult) -> str | None:
    """Best available explanation for a failed command."""
    return _detail(result.stderr or result.stdout)


_XPU_DEVICE_NAME_PATTERN: Final = re.compile(r"Device Name:\s*(.+?)\s*\|?\s*$")


def _xpu_discovery_detail(result: CommandResult) -> str:
    """Return a short device-name summary from `xpu-smi discovery`'s table output."""
    for line in result.stdout.splitlines():
        match = _XPU_DEVICE_NAME_PATTERN.search(line)
        if match:
            return match.group(1).strip()
    return result.first_line()


class _CheckRecorder:
    """Accumulates one tier's checks, timing each one individually."""

    def __init__(self, tier: PreflightTier) -> None:
        self.tier = tier
        self.checks: list[PreflightCheck] = []
        self._started = perf_counter()

    @property
    def recorded(self) -> set[CheckKey]:
        """Keys already recorded, so a partial tier can be completed."""
        return {check.key for check in self.checks}

    def add(
        self,
        key: CheckKey,
        outcome: CheckOutcome,
        *,
        blocking: bool = True,
        reason_code: str | None = None,
        detail: str | None = None,
        method: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Record one check outcome and start timing the next."""
        elapsed = round((perf_counter() - self._started) * 1000) if duration_ms is None else duration_ms
        self.checks.append(
            PreflightCheck(
                key=key,
                tier=self.tier,
                outcome=outcome,
                blocking=blocking,
                checked_at=_now(),
                reason_code=reason_code,
                detail=detail,
                method=method,
                duration_ms=max(0, elapsed),
            )
        )
        self._started = perf_counter()

    def skip(self, *keys: CheckKey, blocking: bool = True, reason_code: str = REASON_NOT_ATTEMPTED) -> None:
        """Record checks that were never attempted because a prerequisite failed."""
        for key in keys:
            self.add(key, CheckOutcome.SKIPPED, blocking=blocking, reason_code=reason_code, duration_ms=0)


def _result(server: RemoteServer, recorder: _CheckRecorder, started: float, checked_at: datetime) -> PreflightResult:
    """Assemble one tier's checks into a ``PreflightResult``."""
    return PreflightResult(
        remote_server_id=server.id,
        tiers_run=[recorder.tier],
        checks=recorder.checks,
        checked_at=checked_at,
        latency_ms=round((perf_counter() - started) * 1000),
    )


# --------------------------------------------------------------------------- #
# Connection stage, shared by both tiers                                      #
# --------------------------------------------------------------------------- #

# Maps a connect failure to (reason_code, whether the TCP dial got through). An
# auth or host-key failure proves reachability; a dial failure does not, and its
# dependent checks must then be SKIPPED rather than FAILED - Studio should not
# claim Docker is broken on a host it never reached.
_CONNECT_ERROR_CLASSES: Final[tuple[tuple[type[Exception], str, bool], ...]] = (
    (SshHostAliasNotFoundError, REASON_ALIAS_NOT_FOUND, False),
    (SshHostKeyUnknownError, REASON_HOST_KEY_UNKNOWN, True),
    (SshHostKeyMismatchError, REASON_HOST_KEY_MISMATCH, True),
    (SshAgentRequiredError, REASON_AGENT_REQUIRED, True),
    (SshAuthenticationError, REASON_AUTH_FAILED, True),
    (SshConnectionError, REASON_UNREACHABLE, False),
)

_HOST_KEY_REASONS: Final = frozenset({REASON_HOST_KEY_UNKNOWN, REASON_HOST_KEY_MISMATCH})


def _classify_connect_error(error: Exception) -> tuple[str, bool]:
    """Return one connect failure's reason code and whether the host answered."""
    for error_type, reason_code, reachable in _CONNECT_ERROR_CLASSES:
        if isinstance(error, error_type):
            return reason_code, reachable
    return REASON_UNREACHABLE, False


def _record_connect_failure(recorder: _CheckRecorder, error: Exception) -> None:
    """Record REACHABLE, AUTHENTICATED and HOST_KEY_VERIFIED from one failure.

    A single connect attempt answers all three, and which exception came back says
    which of them got as far as being tested.
    """
    reason_code, reachable = _classify_connect_error(error)
    host_key_failed = reason_code in _HOST_KEY_REASONS

    if reachable:
        recorder.add(CheckKey.REACHABLE, CheckOutcome.PASSED, method=METHOD_SSH_CONNECT)
    else:
        recorder.add(CheckKey.REACHABLE, CheckOutcome.FAILED, reason_code=reason_code, method=METHOD_SSH_CONNECT)

    if host_key_failed:
        # Verification happens before authentication, so authentication was never
        # attempted - reporting it as failed would send the user chasing keys.
        recorder.add(CheckKey.HOST_KEY_VERIFIED, CheckOutcome.FAILED, reason_code=reason_code)
        recorder.skip(CheckKey.AUTHENTICATED, reason_code=reason_code)
        return

    if not reachable:
        recorder.skip(CheckKey.AUTHENTICATED, CheckKey.HOST_KEY_VERIFIED, reason_code=reason_code)
        return

    recorder.add(CheckKey.HOST_KEY_VERIFIED, CheckOutcome.PASSED, method=METHOD_SSH_CONNECT)
    recorder.add(CheckKey.AUTHENTICATED, CheckOutcome.FAILED, reason_code=reason_code)


def _record_alias_check(recorder: _CheckRecorder, server: RemoteServer, settings: Settings) -> bool:
    """Record ``ALIAS_RESOLVED`` and return whether the alias is usable.

    Resolution goes through the read-only SSH config reader, which rejects a
    wildcard-only match: a pattern stanza is not a usable target.
    """
    resolved = resolve_alias(settings.ssh_config_path, server.ssh_host_alias)
    if resolved.found:
        recorder.add(CheckKey.ALIAS_RESOLVED, CheckOutcome.PASSED, method=METHOD_SSH_CONFIG)
        return True
    recorder.add(
        CheckKey.ALIAS_RESOLVED,
        CheckOutcome.FAILED,
        reason_code=REASON_ALIAS_NOT_FOUND,
        detail="Alias is absent from the SSH config, or matches only a wildcard stanza.",
        method=METHOD_SSH_CONFIG,
    )
    return False


# --------------------------------------------------------------------------- #
# Tier 1 checks                                                               #
# --------------------------------------------------------------------------- #


async def _check_docker(recorder: _CheckRecorder, transport: SshTransport) -> None:
    """Record whether the SSH user can talk to the Docker daemon.

    Queries the *server* version rather than ``docker --version``: the client
    binary existing says nothing about the socket being reachable by this user,
    which is the failure this check is for.
    """
    result = await transport.run_command(["docker", "version", "--format", "{{.Server.Version}}"])
    if result.ok:
        recorder.add(CheckKey.DOCKER_USABLE, CheckOutcome.PASSED, detail=result.first_line(), method=METHOD_DOCKER)
        return
    recorder.add(
        CheckKey.DOCKER_USABLE,
        CheckOutcome.FAILED,
        reason_code=REASON_DOCKER_UNAVAILABLE,
        detail=_failure_detail(result) or "The Docker daemon did not respond.",
        method=METHOD_DOCKER,
    )


def _parse_free_bytes(stdout: str) -> int | None:
    """Parse available bytes out of ``df -B1 -P`` output.

    Returns ``None`` when no data row parses, so an unexpected ``df`` reports an
    unparseable-output failure rather than a confidently wrong number.
    """
    for line in stdout.splitlines()[1:]:
        columns = line.split()
        if len(columns) < _DF_MIN_COLUMNS:
            continue
        try:
            return int(columns[_DF_AVAILABLE_COLUMN])
        except ValueError:
            continue
    return None


async def _check_disk(recorder: _CheckRecorder, transport: SshTransport, settings: Settings) -> None:
    """Record whether the host has room for the trainer image plus a job."""
    result = await transport.run_command(["df", "-B1", "-P", "/var/lib/docker"])
    if not result.ok:
        # A rootless or relocated data root means /var/lib/docker need not exist;
        # the root filesystem is the useful approximation, not a failure.
        result = await transport.run_command(["df", "-B1", "-P", "/"])
    if not result.ok:
        recorder.add(
            CheckKey.DISK_SPACE,
            CheckOutcome.FAILED,
            reason_code=REASON_COMMAND_FAILED,
            detail=_failure_detail(result) or "Could not measure free disk space.",
            method=METHOD_DF,
        )
        return

    free_bytes = _parse_free_bytes(result.stdout)
    if free_bytes is None:
        recorder.add(
            CheckKey.DISK_SPACE,
            CheckOutcome.FAILED,
            reason_code=REASON_UNPARSEABLE_OUTPUT,
            detail="Could not parse free disk space from df output.",
            method=METHOD_DF,
        )
        return

    required = settings.ssh_min_free_disk_bytes
    free_gib = free_bytes / _BYTES_PER_GIB
    if free_bytes >= required:
        recorder.add(CheckKey.DISK_SPACE, CheckOutcome.PASSED, detail=f"{free_gib:.1f} GiB free", method=METHOD_DF)
        return
    recorder.add(
        CheckKey.DISK_SPACE,
        CheckOutcome.FAILED,
        reason_code=REASON_INSUFFICIENT_DISK,
        detail=f"{free_gib:.1f} GiB free, {required / _BYTES_PER_GIB:.1f} GiB required",
        method=METHOD_DF,
    )


async def _probe_intel_render_node(transport: SshTransport) -> CommandResult:
    """Probe for an Intel render node, the XPU fallback signal.

    Two signals are required: a ``/dev/dri/renderD*`` node must exist, and some
    DRM device must report the Intel PCI vendor id. Both the script and the vendor
    id are application constants, and the transport shell-quotes every element, so
    the remote shell only ever expands the globs in this fixed text.
    """
    return await transport.run_command(
        [
            "sh",
            "-c",
            'ls /dev/dri/renderD* >/dev/null 2>&1 && grep -qil "$1" /sys/class/drm/*/device/vendor',
            "sh",
            _INTEL_VENDOR_ID,
        ]
    )


async def _check_driver_cuda(recorder: _CheckRecorder, transport: SshTransport) -> str | None:
    """Record ``DRIVER_PRESENT`` for a CUDA host."""
    result = await transport.run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if result.ok and result.first_line():
        recorder.add(
            CheckKey.DRIVER_PRESENT,
            CheckOutcome.PASSED,
            detail=result.first_line(),
            method=METHOD_NVIDIA_SMI,
        )
        return METHOD_NVIDIA_SMI
    recorder.add(
        CheckKey.DRIVER_PRESENT,
        CheckOutcome.FAILED,
        reason_code=REASON_DRIVER_MISSING,
        detail=_failure_detail(result) or "nvidia-smi is not available on the remote host.",
        method=METHOD_NVIDIA_SMI,
    )
    return None


async def _check_driver_xpu(recorder: _CheckRecorder, transport: SshTransport) -> str | None:
    """Record ``DRIVER_PRESENT`` for an XPU host.

    ``xpu-smi`` first, then the render-node fallback: ``xpu-smi`` is frequently
    absent on hosts whose XPUs work perfectly, so requiring it would reject valid
    targets. Tier 2's in-container ``torch.xpu.is_available()`` is the
    authoritative answer; Tier 1 needs only enough signal to reject a host with no
    Intel accelerator at all.
    """
    smi = await transport.run_command(["xpu-smi", "discovery"])
    if smi.ok:
        recorder.add(
            CheckKey.DRIVER_PRESENT, CheckOutcome.PASSED, detail=_xpu_discovery_detail(smi), method=METHOD_XPU_SMI
        )
        return METHOD_XPU_SMI

    render = await _probe_intel_render_node(transport)
    if render.ok:
        recorder.add(
            CheckKey.DRIVER_PRESENT,
            CheckOutcome.PASSED,
            detail="Intel render node present.",
            method=METHOD_RENDER_NODE,
        )
        return METHOD_RENDER_NODE

    recorder.add(
        CheckKey.DRIVER_PRESENT,
        CheckOutcome.FAILED,
        reason_code=REASON_DRIVER_MISSING,
        detail="Neither xpu-smi nor an Intel render node was found.",
        method=METHOD_RENDER_NODE,
    )
    return None


async def _check_driver(recorder: _CheckRecorder, transport: SshTransport, device_type: DeviceType) -> str | None:
    """Record ``DRIVER_PRESENT`` and return the method that answered."""
    if device_type is DeviceType.CUDA:
        return await _check_driver_cuda(recorder, transport)
    return await _check_driver_xpu(recorder, transport)


def registry_probe_url(registry: str) -> str:
    """Return the registry-API URL whose reachability Tier 1 checks.

    Only the registry host matters: this is a reachability probe, and resolving a
    repository or tag is Tier 2's job.

    Args:
        registry: Configured registry, e.g. ``ghcr.io/open-edge-platform``.

    Returns:
        The ``https://<host>/v2/`` URL to issue a ``HEAD`` against.
    """
    candidate = registry if "://" in registry else f"https://{registry}"
    host = urlsplit(candidate).netloc or registry.split("/", 1)[0]
    return f"https://{host}/v2/"


async def _check_registry(recorder: _CheckRecorder, transport: SshTransport, settings: Settings) -> None:
    """Record whether the remote host can reach the trainer image registry.

    Probed *from the remote host*, because that is where the eventual
    ``docker pull`` runs - a registry reachable from the Studio machine but
    firewalled on the GPU box is exactly the misconfiguration worth catching
    before a job is accepted.

    A ``HEAD`` and nothing more: no manifest, no layer, no image. ``/v2/`` answers
    ``401`` to an anonymous client on GHCR, which still proves reachability, so any
    HTTP response counts as reachable and only a transport-level failure counts as
    unreachable.
    """
    url = registry_probe_url(settings.trainer_image_registry)
    result = await transport.run_command(
        ["curl", "--head", "--silent", "--show-error", "--max-time", "10", "--output", "/dev/null", url]
    )
    if result.ok:
        recorder.add(CheckKey.REGISTRY_REACHABLE, CheckOutcome.PASSED, detail=url, method=METHOD_CURL)
        return
    recorder.add(
        CheckKey.REGISTRY_REACHABLE,
        CheckOutcome.FAILED,
        reason_code=REASON_REGISTRY_UNREACHABLE,
        detail=_failure_detail(result) or f"Could not reach {url}",
        method=METHOD_CURL,
    )


async def _check_gpu_free_cuda(recorder: _CheckRecorder, transport: SshTransport) -> None:
    """Record CUDA GPU occupancy. Never blocking, and "busy" is never FAILED."""
    apps = await transport.run_command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    if not apps.ok:
        recorder.add(
            CheckKey.GPU_FREE,
            CheckOutcome.SKIPPED,
            blocking=False,
            reason_code=REASON_COMMAND_FAILED,
            detail=_failure_detail(apps) or "Could not query GPU compute processes.",
            method=METHOD_NVIDIA_SMI,
        )
        return

    processes = [line for line in apps.stdout.splitlines() if line.strip()]
    if processes:
        recorder.add(
            CheckKey.GPU_FREE,
            CheckOutcome.WARNING,
            blocking=False,
            reason_code=REASON_GPU_BUSY,
            detail=f"{len(processes)} process(es) currently hold the GPU.",
            method=METHOD_NVIDIA_SMI,
        )
        return
    recorder.add(CheckKey.GPU_FREE, CheckOutcome.PASSED, blocking=False, method=METHOD_NVIDIA_SMI)


def _parse_memory_fraction(stdout: str) -> float | None:
    """Parse the first two numbers on a ``used,total`` line into a fraction."""
    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", stdout)]
    if len(numbers) < 2 or numbers[1] <= 0:
        return None
    return numbers[0] / numbers[1]


async def _check_gpu_free_xpu(recorder: _CheckRecorder, transport: SshTransport, driver_method: str | None) -> None:
    """Record XPU occupancy, best effort.

    XPU offers no per-process attribution comparable to ``nvidia-smi``, so this
    falls back to a memory-utilization heuristic. With no usable signal the check
    is ``SKIPPED``, never ``PASSED``: reporting a busy accelerator as free would
    send a job straight into an out-of-memory failure.
    """
    if driver_method != METHOD_XPU_SMI:
        recorder.add(
            CheckKey.GPU_FREE,
            CheckOutcome.SKIPPED,
            blocking=False,
            reason_code=REASON_NO_SIGNAL,
            detail="xpu-smi is not available, so occupancy cannot be determined.",
            method=METHOD_RENDER_NODE,
        )
        return

    stats = await transport.run_command(["xpu-smi", "stats", "-d", "0"])
    if not stats.ok:
        recorder.add(
            CheckKey.GPU_FREE,
            CheckOutcome.SKIPPED,
            blocking=False,
            reason_code=REASON_COMMAND_FAILED,
            detail=_failure_detail(stats) or "Could not query xpu-smi statistics.",
            method=METHOD_XPU_SMI,
        )
        return

    fraction = _parse_memory_fraction(stats.stdout)
    if fraction is None:
        recorder.add(
            CheckKey.GPU_FREE,
            CheckOutcome.SKIPPED,
            blocking=False,
            reason_code=REASON_NO_SIGNAL,
            detail="Could not parse XPU memory usage.",
            method=METHOD_XPU_SMI,
        )
        return
    if fraction >= _GPU_BUSY_MEMORY_FRACTION:
        recorder.add(
            CheckKey.GPU_FREE,
            CheckOutcome.WARNING,
            blocking=False,
            reason_code=REASON_GPU_BUSY,
            detail=f"{fraction:.0%} of XPU memory is in use.",
            method=METHOD_XPU_SMI,
        )
        return
    recorder.add(CheckKey.GPU_FREE, CheckOutcome.PASSED, blocking=False, method=METHOD_XPU_SMI)


async def _check_gpu_free(
    recorder: _CheckRecorder,
    transport: SshTransport,
    device_type: DeviceType,
    driver_method: str | None,
) -> None:
    """Record GPU occupancy for the configured device type."""
    if device_type is DeviceType.CUDA:
        await _check_gpu_free_cuda(recorder, transport)
        return
    await _check_gpu_free_xpu(recorder, transport, driver_method)


# Tier 1 checks that need an open connection, in the order they run.
_TIER1_REMOTE_KEYS: Final = (
    CheckKey.DOCKER_USABLE,
    CheckKey.DISK_SPACE,
    CheckKey.DRIVER_PRESENT,
    CheckKey.REGISTRY_REACHABLE,
)


def _complete_tier1(recorder: _CheckRecorder, reason_code: str) -> None:
    """Record every not-yet-attempted Tier 1 check as skipped."""
    recorded = recorder.recorded
    recorder.skip(*(key for key in _TIER1_REMOTE_KEYS if key not in recorded), reason_code=reason_code)
    if CheckKey.GPU_FREE not in recorded:
        recorder.skip(CheckKey.GPU_FREE, blocking=False, reason_code=reason_code)


async def _run_tier1_remote_checks(
    recorder: _CheckRecorder,
    transport: SshTransport,
    server: RemoteServer,
    settings: Settings,
) -> None:
    """Run the Tier 1 checks that need an open connection."""
    recorder.add(CheckKey.REACHABLE, CheckOutcome.PASSED, method=METHOD_SSH_CONNECT)
    recorder.add(CheckKey.HOST_KEY_VERIFIED, CheckOutcome.PASSED, method=METHOD_SSH_CONNECT)
    recorder.add(CheckKey.AUTHENTICATED, CheckOutcome.PASSED, method=METHOD_SSH_CONNECT)

    await _check_docker(recorder, transport)
    await _check_disk(recorder, transport, settings)
    driver_method = await _check_driver(recorder, transport, server.device_type)
    await _check_registry(recorder, transport, settings)
    await _check_gpu_free(recorder, transport, server.device_type, driver_method)


async def run_tier1_preflight(server: RemoteServer) -> PreflightResult:
    """Run the cheap, save-gating preflight against one server.

    Resolves the alias, opens one bounded SSH connection, and runs four short
    remote probes plus a non-blocking occupancy check. Performs no image
    resolution, no pull, and starts no container: the registry check is a ``HEAD``
    against the registry API root.

    Does not raise for a server that fails its checks. Callers decide from
    :attr:`~schemas.ssh_preflight.PreflightResult.blocking_failures` whether to
    reject a save, so the user sees every cause at once.

    Args:
        server: The server whose configuration to verify.

    Returns:
        One check per Tier 1 key, each with its own outcome and duration.
    """
    settings = get_settings()
    started = perf_counter()
    checked_at = _now()
    recorder = _CheckRecorder(PreflightTier.TIER_1)

    if not _record_alias_check(recorder, server, settings):
        recorder.skip(
            CheckKey.REACHABLE,
            CheckKey.AUTHENTICATED,
            CheckKey.HOST_KEY_VERIFIED,
            reason_code=REASON_ALIAS_NOT_FOUND,
        )
        _complete_tier1(recorder, REASON_ALIAS_NOT_FOUND)
        return _result(server, recorder, started, checked_at)

    if server.device_type not in SSH_SERVER_DEVICE_TYPES:
        # The schema rejects this on save, but a record predating that validation
        # must not silently pass a driver check that cannot be performed.
        recorder.skip(
            CheckKey.REACHABLE,
            CheckKey.AUTHENTICATED,
            CheckKey.HOST_KEY_VERIFIED,
            reason_code=REASON_UNSUPPORTED_DEVICE,
        )
        recorder.add(
            CheckKey.DRIVER_PRESENT,
            CheckOutcome.FAILED,
            reason_code=REASON_UNSUPPORTED_DEVICE,
            detail=f"No trainer image exists for device type '{server.device_type.value}'.",
        )
        _complete_tier1(recorder, REASON_UNSUPPORTED_DEVICE)
        return _result(server, recorder, started, checked_at)

    transport = transport_factory(server.ssh_host_alias)
    try:
        await transport.connect()
    except _CONNECT_FAILURES as error:
        _record_connect_failure(recorder, error)
        _complete_tier1(recorder, _classify_connect_error(error)[0])
        return _result(server, recorder, started, checked_at)

    try:
        await _run_tier1_remote_checks(recorder, transport, server, settings)
    except SshConnectionError as error:
        # The connection dropped partway through: the remaining checks are
        # unknown, not failed.
        logger.warning(
            "SSH connection lost during Tier 1 preflight for alias '{}': {}",
            server.ssh_host_alias,
            error.message,
        )
        _complete_tier1(recorder, REASON_UNREACHABLE)
    finally:
        await transport.close()

    return _result(server, recorder, started, checked_at)


# --------------------------------------------------------------------------- #
# Tier 2 checks                                                               #
# --------------------------------------------------------------------------- #

# Protocol version this backend speaks. Callers pass the compiled-in value from
# the trainer package; importing it here would couple the SSH service to the
# trainer, which the preflight must be able to run without.
DEFAULT_PROTOCOL_VERSION: Final = 1

# Label the trainer images carry their protocol version in, set by
# `docker/Dockerfile.trainer` from TRAINER_API_PROTOCOL_VERSION.
PROTOCOL_LABEL: Final = "org.open-edge-platform.physicalai.trainer.api-protocol"

_TIER2_KEYS: Final = (
    CheckKey.IMAGE_RESOLVED,
    CheckKey.IMAGE_SIGNATURE,
    CheckKey.CONTAINER_DEVICE_PROBE,
    CheckKey.PROTOCOL_COMPATIBLE,
)

# IMAGE_SIGNATURE is advisory: the images are signed at publish time, so a host
# without cosign is informative, not broken.
_TIER2_NON_BLOCKING: Final = frozenset({CheckKey.IMAGE_SIGNATURE})


def trainer_image_ref(registry: str, device_type: DeviceType, tag: str) -> str:
    """Build a trainer image reference from application constants.

    Args:
        registry: Configured registry.
        device_type: The server's accelerator.
        tag: Image tag.

    Returns:
        The ``<registry>/physicalai-trainer-<device>:<tag>`` reference.
    """
    return f"{registry.rstrip('/')}/physicalai-trainer-{device_type.value}:{tag}"


def protocol_tag(protocol_version: int) -> str:
    """Return the protocol-pinned image tag for a protocol version."""
    return f"protocol-{protocol_version}"


def _complete_tier2(recorder: _CheckRecorder, reason_code: str) -> None:
    """Record every not-yet-attempted Tier 2 check as skipped."""
    recorded = recorder.recorded
    for key in _TIER2_KEYS:
        if key not in recorded:
            recorder.skip(key, blocking=key not in _TIER2_NON_BLOCKING, reason_code=reason_code)


async def _resolve_image(
    recorder: _CheckRecorder,
    transport: SshTransport,
    server: RemoteServer,
    protocol_version: int,
    image_ref_hint: str | None,
) -> str | None:
    """Resolve the trainer image, recording ``IMAGE_RESOLVED``.

    Prefers the protocol-pinned tag and falls back to ``latest``. The fallback is
    a ``WARNING``, not a failure: a protocol bump can land in Studio before CI has
    published a matching trainer image, and blocking every verification until then
    would be worse than telling the user which tag was actually used. Only failing
    both tags is fatal.

    Args:
        recorder: Recorder for this tier.
        transport: Open transport to the server.
        server: The server being verified.
        protocol_version: Protocol version this backend speaks.
        image_ref_hint: Image reference to try ahead of the pinned tag.

    Returns:
        The resolved image reference, or ``None`` when nothing resolved.
    """
    registry = get_settings().trainer_image_registry
    preferred = image_ref_hint or trainer_image_ref(registry, server.device_type, protocol_tag(protocol_version))

    result = await transport.run_command(["docker", "manifest", "inspect", preferred])
    if result.ok:
        recorder.add(CheckKey.IMAGE_RESOLVED, CheckOutcome.PASSED, detail=preferred, method=METHOD_DOCKER_MANIFEST)
        return preferred

    fallback = trainer_image_ref(registry, server.device_type, "latest")
    if fallback != preferred:
        fallback_result = await transport.run_command(["docker", "manifest", "inspect", fallback])
        if fallback_result.ok:
            logger.warning("Trainer image '{}' did not resolve; falling back to '{}'", preferred, fallback)
            recorder.add(
                CheckKey.IMAGE_RESOLVED,
                CheckOutcome.WARNING,
                reason_code=REASON_PROTOCOL_TAG_UNRESOLVED,
                detail=f"{preferred} did not resolve; using {fallback}",
                method=METHOD_DOCKER_MANIFEST,
            )
            return fallback

    recorder.add(
        CheckKey.IMAGE_RESOLVED,
        CheckOutcome.FAILED,
        reason_code=REASON_IMAGE_UNRESOLVED,
        detail=_failure_detail(result) or f"Could not resolve {preferred}",
        method=METHOD_DOCKER_MANIFEST,
    )
    return None


async def _check_signature(recorder: _CheckRecorder, transport: SshTransport, image_ref: str) -> None:
    """Verify the image signature when ``cosign`` is available on the host.

    Defense in depth rather than required infrastructure, so a host without
    ``cosign`` is ``SKIPPED`` and a failed verification is a non-blocking
    ``WARNING``: the publish-time signature is the primary control.
    """
    available = await transport.run_command(["cosign", "version"])
    if not available.ok:
        recorder.add(
            CheckKey.IMAGE_SIGNATURE,
            CheckOutcome.SKIPPED,
            blocking=False,
            reason_code=REASON_TOOL_MISSING,
            detail="cosign is not installed on the remote host, so the signature was not verified there.",
            method=METHOD_COSIGN,
        )
        return

    verified = await transport.run_command(["cosign", "verify", image_ref])
    if verified.ok:
        recorder.add(CheckKey.IMAGE_SIGNATURE, CheckOutcome.PASSED, blocking=False, method=METHOD_COSIGN)
        return
    recorder.add(
        CheckKey.IMAGE_SIGNATURE,
        CheckOutcome.WARNING,
        blocking=False,
        reason_code=REASON_COMMAND_FAILED,
        detail=_failure_detail(verified) or "cosign could not verify the image signature.",
        method=METHOD_COSIGN,
    )


async def resolve_render_group_gid(transport: SshTransport) -> str | None:
    """Return the host GID that owns the first Intel render node, or ``None``.

    The trainer container always runs as a fixed non-root UID/GID (see
    ``docker_ops.build_run_argv``'s ``--user 10001:10001``), which is never a
    member of the host's render group by default. ``--device /dev/dri`` alone
    passes the device node through but leaves it unreadable by that user - the
    node's group ownership still gates access, and without ``--group-add
    <gid>`` the container's ``torch.xpu.is_available()`` reports zero devices
    even though the host driver and render node are both fine. The GID is
    discovered per host, never hardcoded, because it is not portable across
    distributions.
    """
    result = await transport.run_command(
        ["sh", "-c", "stat -c %g $(ls /dev/dri/renderD* 2>/dev/null | head -n1) 2>/dev/null"]
    )
    gid = result.first_line().strip()
    return gid or None


def _device_run_args(device_type: DeviceType, render_gid: str | None = None) -> list[str]:
    """Return the ``docker run`` flags that expose the accelerator.

    Derived from the configured device type, never from user-supplied text.
    ``render_gid``, when known, is added via ``--group-add`` so the
    container's non-root user can actually read/write the render node -
    without it the device node is present but access is denied.
    """
    if device_type is DeviceType.CUDA:
        return ["--gpus", "all"]
    args = ["--device", "/dev/dri"]
    if render_gid:
        args.extend(["--group-add", render_gid])
    return args


def _device_probe_expression(device_type: DeviceType) -> str:
    """Return the in-container device-availability expression."""
    accelerator = "cuda" if device_type is DeviceType.CUDA else "xpu"
    return f"import torch; print(torch.{accelerator}.is_available())"


# Substrings Docker's own client prints while it pulls an image inline for a
# `docker run` whose image is not yet cached locally. Matched against a failed
# probe's output as a defense-in-depth fallback for the race between
# `_image_present_locally` and the `docker run` a few lines later - e.g.
# another process evicting the image in between. The normal path never hits
# this: the presence check below routes a genuinely absent image to a
# background pull instead of an inline one.
_IMAGE_PULL_MARKERS: Final = ("unable to find image", "pulling fs layer", "pulling from")


def _is_pulling_image(result: CommandResult) -> bool:
    """True when a failed command's output shows Docker mid-pull, not broken."""
    text = f"{result.stdout}\n{result.stderr}".lower()
    return any(marker in text for marker in _IMAGE_PULL_MARKERS)


async def _image_present_locally(transport: SshTransport, image_ref: str) -> bool:
    """True when the image is already cached in the remote Docker image store."""
    result = await transport.run_command(["docker", "image", "inspect", image_ref])
    return result.ok


def _pull_state_paths(image_ref: str) -> tuple[str, str]:
    """Return the (pid file, log file) paths a background pull for this image uses.

    Named from a hash of the image reference rather than the reference itself,
    so the path is stable across calls and never carries `/`, `:`, or other
    reference characters into a filename.
    """
    digest = hashlib.sha256(image_ref.encode()).hexdigest()[:16]
    # These are a remote-host path (run over SSH on the target machine), not a local temp-file access.
    return (
        f"/tmp/physicalai-pull-{digest}.pid",  # noqa: S108 # nosec B108
        f"/tmp/physicalai-pull-{digest}.log",  # noqa: S108 # nosec B108
    )


async def _pull_already_running(transport: SshTransport, pidfile: str) -> bool:
    """True when a previously started background pull's PID is still alive."""
    result = await transport.run_command(
        ["sh", "-c", 'test -f "$1" && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null', "sh", pidfile]
    )
    return result.ok


async def _start_background_pull(transport: SshTransport, image_ref: str, pidfile: str, logfile: str) -> None:
    """Launch ``docker pull`` detached from this SSH session so it outlives it.

    ``nohup`` plus stdio redirected away from the exec channel is what makes
    this survive: Tier 2 closes its SSH connection right after this check
    returns, which would otherwise SIGHUP a foreground pull mid-transfer,
    aborting it and discarding an already-mostly-downloaded image. Progress
    and PID land in host-local files under ``/tmp`` so a later check can tell
    "still pulling" from "pull finished or died" without needing this SSH
    connection to still be open.
    """
    await transport.run_command(
        [
            "sh",
            "-c",
            'nohup docker pull "$1" >"$2" 2>&1 </dev/null & echo "$!" >"$3"',
            "sh",
            image_ref,
            logfile,
            pidfile,
        ]
    )


async def _check_device_probe(
    recorder: _CheckRecorder,
    transport: SshTransport,
    device_type: DeviceType,
    image_ref: str,
) -> bool:
    """Run a one-shot container and record whether it sees the accelerator.

    The authoritative check, and the reason Tier 2 exists: a driver visible on the
    host still tells you nothing about whether the container runtime passes the
    device through.

    Checks the image is already cached locally before running anything: a
    `docker run` against a cold image pulls it inline, tying the transfer to
    this check's short timeout and to this SSH connection's lifetime - both of
    which end long before a multi-gigabyte image finishes. An absent image is
    instead handed to a detached background pull that keeps going after this
    check (and the whole preflight) returns.

    Returns:
        True when the image was still being pulled when this probe ran, so the
        caller can skip the protocol check rather than have it fail against an
        image that is not there yet.
    """
    if not await _image_present_locally(transport, image_ref):
        pidfile, logfile = _pull_state_paths(image_ref)
        already_running = await _pull_already_running(transport, pidfile)
        if not already_running:
            await _start_background_pull(transport, image_ref, pidfile, logfile)

        verb = "Still pulling" if already_running else "Started pulling"
        recorder.add(
            CheckKey.CONTAINER_DEVICE_PROBE,
            CheckOutcome.SKIPPED,
            reason_code=REASON_IMAGE_PULLING,
            detail=f"{verb} {image_ref} in the background. Run Test connection again once it finishes.",
            method=METHOD_CONTAINER,
        )
        return True

    render_gid = None if device_type is DeviceType.CUDA else await resolve_render_group_gid(transport)
    argv = [
        "docker",
        "run",
        "--rm",
        *_device_run_args(device_type, render_gid),
        "--entrypoint",
        "python",
        image_ref,
        "-c",
        _device_probe_expression(device_type),
    ]
    result = await transport.run_command(argv, timeout=get_settings().ssh_preflight_timeout_s)
    if result.ok and result.first_line().lower() == "true":
        recorder.add(CheckKey.CONTAINER_DEVICE_PROBE, CheckOutcome.PASSED, method=METHOD_CONTAINER)
        return False
    if _is_pulling_image(result):
        recorder.add(
            CheckKey.CONTAINER_DEVICE_PROBE,
            CheckOutcome.SKIPPED,
            reason_code=REASON_IMAGE_PULLING,
            detail=f"Started pulling {image_ref}. Run Test connection again once the pull finishes.",
            method=METHOD_CONTAINER,
        )
        return True
    recorder.add(
        CheckKey.CONTAINER_DEVICE_PROBE,
        CheckOutcome.FAILED,
        reason_code=REASON_DEVICE_UNAVAILABLE,
        detail=_failure_detail(result) or "The container did not report the device as available.",
        method=METHOD_CONTAINER,
    )
    return False


async def _check_protocol(
    recorder: _CheckRecorder,
    transport: SshTransport,
    image_ref: str,
    protocol_version: int,
) -> None:
    """Compare the image's advertised protocol version against the expected one.

    Read from the image label rather than by standing the trainer up and calling
    ``/health``: the label is baked in at build time by the same CI that sets the
    trainer's runtime value, so reading it needs no container, no port, and no
    tunnel. Provisioning re-verifies against the live ``/health`` response before
    uploading a dataset.
    """
    label_format = f'{{{{index .Config.Labels "{PROTOCOL_LABEL}"}}}}'
    result = await transport.run_command(["docker", "image", "inspect", "--format", label_format, image_ref])
    reported = result.first_line() if result.ok else ""
    if not reported or reported == "<no value>":
        recorder.add(
            CheckKey.PROTOCOL_COMPATIBLE,
            CheckOutcome.FAILED,
            reason_code=REASON_PROTOCOL_UNKNOWN,
            detail="The image advertises no trainer protocol version.",
            method=METHOD_DOCKER,
        )
        return

    try:
        image_protocol = int(reported)
    except ValueError:
        recorder.add(
            CheckKey.PROTOCOL_COMPATIBLE,
            CheckOutcome.FAILED,
            reason_code=REASON_UNPARSEABLE_OUTPUT,
            detail=_detail(f"Unparseable protocol version: {reported}"),
            method=METHOD_DOCKER,
        )
        return

    if image_protocol == protocol_version:
        recorder.add(
            CheckKey.PROTOCOL_COMPATIBLE,
            CheckOutcome.PASSED,
            detail=f"protocol {image_protocol}",
            method=METHOD_DOCKER,
        )
        return
    recorder.add(
        CheckKey.PROTOCOL_COMPATIBLE,
        CheckOutcome.FAILED,
        reason_code=REASON_PROTOCOL_MISMATCH,
        detail=f"The image speaks protocol {image_protocol}; this backend speaks {protocol_version}.",
        method=METHOD_DOCKER,
    )


async def run_tier2_preflight(
    server: RemoteServer,
    image_ref_hint: str | None = None,
    protocol_version: int = DEFAULT_PROTOCOL_VERSION,
) -> PreflightResult:
    """Run the expensive verification tier against one server.

    Invoked only by an explicit verify action: it inspects the trainer image in
    the registry and starts a one-shot container, neither of which belongs inside
    a create/update request. ``protocol_version`` is a parameter so this module
    stays independent of the trainer package.

    Like Tier 1, does not raise for a server that fails its checks.

    Args:
        server: The server to verify.
        image_ref_hint: Image reference to try before the protocol-pinned tag.
        protocol_version: Trainer API protocol version this backend speaks.

    Returns:
        One check per Tier 2 key.
    """
    started = perf_counter()
    checked_at = _now()
    recorder = _CheckRecorder(PreflightTier.TIER_2)

    transport = transport_factory(server.ssh_host_alias)
    try:
        await transport.connect()
    except _CONNECT_FAILURES as error:
        _complete_tier2(recorder, _classify_connect_error(error)[0])
        return _result(server, recorder, started, checked_at)

    try:
        image_ref = await _resolve_image(recorder, transport, server, protocol_version, image_ref_hint)
        if image_ref is None:
            _complete_tier2(recorder, REASON_IMAGE_UNRESOLVED)
            return _result(server, recorder, started, checked_at)

        await _check_signature(recorder, transport, image_ref)
        pulling = await _check_device_probe(recorder, transport, server.device_type, image_ref)
        if pulling:
            # The protocol check reads a label off the local image via `docker
            # image inspect`; with the pull still in flight that image is not
            # there yet, and running it now would report a spurious
            # "no protocol version" failure instead of the real cause.
            recorder.skip(CheckKey.PROTOCOL_COMPATIBLE, reason_code=REASON_IMAGE_PULLING)
        else:
            await _check_protocol(recorder, transport, image_ref, protocol_version)
    except SshConnectionError as error:
        logger.warning(
            "SSH connection lost during Tier 2 preflight for alias '{}': {}",
            server.ssh_host_alias,
            error.message,
        )
        _complete_tier2(recorder, REASON_UNREACHABLE)
    finally:
        await transport.close()

    return _result(server, recorder, started, checked_at)


def set_transport_factory(factory: Callable[[str], SshTransport]) -> None:
    """Replace the transport factory both tiers use."""
    global transport_factory  # noqa: PLW0603 - the module-level factory is the documented seam.
    transport_factory = factory


def reset_transport_factory() -> None:
    """Restore the real transport factory."""
    set_transport_factory(open_transport)
