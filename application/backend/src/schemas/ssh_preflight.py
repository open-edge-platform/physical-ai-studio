# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Structured results for the two-tier SSH preflight.

Tier 1 is cheap and gates a save. Tier 2 pulls a multi-gigabyte image and runs a
one-shot GPU container, so it is an explicit action and never runs inline in a
create/update request.

Every result is tagged with its tier and its own ``checked_at`` so the UI can
group them and never present a stale Tier 2 result as current. ``blocking``
separates a real misconfiguration from a reported-but-transient condition: GPU
occupancy is reported and never blocks a save.
"""

from datetime import datetime
from enum import IntEnum, StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class PreflightTier(IntEnum):
    """Which tier a check belongs to."""

    # Cheap, bounded, gates create/update.
    TIER_1 = 1
    # Expensive (registry pull, one-shot GPU container). Explicitly invoked.
    TIER_2 = 2


class CheckOutcome(StrEnum):
    """Result of a single preflight check."""

    PASSED = "passed"
    FAILED = "failed"
    # Ran, and the answer is a condition the user should see but which does not
    # make the server unusable. GPU occupancy is the canonical case.
    WARNING = "warning"
    # Not attempted, because a prerequisite check failed or the tier was not run.
    SKIPPED = "skipped"


class CheckKey(StrEnum):
    """Stable identifiers for each preflight check.

    Shared with the UI, which groups and labels results by these keys. Renaming
    one is a breaking change.
    """

    # --- Tier 1 -------------------------------------------------------------
    ALIAS_RESOLVED = "alias_resolved"
    REACHABLE = "reachable"
    AUTHENTICATED = "authenticated"
    HOST_KEY_VERIFIED = "host_key_verified"
    DOCKER_USABLE = "docker_usable"
    DISK_SPACE = "disk_space"
    DRIVER_PRESENT = "driver_present"
    REGISTRY_REACHABLE = "registry_reachable"
    GPU_FREE = "gpu_free"
    # --- Tier 2 -------------------------------------------------------------
    IMAGE_RESOLVED = "image_resolved"
    IMAGE_SIGNATURE = "image_signature"
    CONTAINER_DEVICE_PROBE = "container_device_probe"
    PROTOCOL_COMPATIBLE = "protocol_compatible"


class PreflightCheck(BaseModel):
    """One preflight check's outcome.

    ``detail`` is operator-facing text. It is sanitized and length-capped before
    it lands here: remote command output is environment-influenced, not trusted.
    """

    key: CheckKey
    tier: PreflightTier
    outcome: CheckOutcome
    blocking: bool = Field(
        description="Whether a FAILED outcome for this check prevents saving the server.",
    )
    checked_at: datetime
    # Stable machine-readable cause, e.g. "alias_not_found", "host_key_unknown".
    reason_code: str | None = None
    # Short human-readable explanation. Sanitized; never contains a key path,
    # host key, raw SSH exception text, or remote command text.
    detail: str | None = None
    # Which detection method answered, e.g. "nvidia-smi", "xpu-smi", "render-node".
    method: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)


class PreflightResult(BaseModel):
    """The outcome of running one or both preflight tiers against a server."""

    remote_server_id: UUID | None = None
    tiers_run: list[PreflightTier] = Field(default_factory=list)
    checks: list[PreflightCheck] = Field(default_factory=list)
    checked_at: datetime
    latency_ms: int | None = Field(default=None, ge=0)

    @property
    def blocking_failures(self) -> list[PreflightCheck]:
        """Checks whose failure must prevent a save."""
        return [check for check in self.checks if check.blocking and check.outcome is CheckOutcome.FAILED]

    @property
    def passed(self) -> bool:
        """True when no blocking check failed."""
        return not self.blocking_failures

    def check(self, key: CheckKey) -> PreflightCheck | None:
        """Return the result for one check, when it was run."""
        return next((candidate for candidate in self.checks if candidate.key is key), None)


class RemoteServerStatus(BaseModel):
    """Structured status for the remote-server status endpoint.

    Combines the last recorded preflight with live in-use information, keeping
    each check's own tier and timestamp so the UI never renders a stale Tier 2
    result as current.
    """

    remote_server_id: UUID
    status: str = Field(description="Rolled-up health: healthy, degraded, unreachable, or unknown.")
    device_type: str = Field(description="The server's configured device type, served from the DB record.")
    checks: list[PreflightCheck] = Field(default_factory=list)
    checked_at: datetime | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    reason_code: str | None = None
    in_use_by_job_id: UUID | None = None
    waiting_for_gpu: bool = False
