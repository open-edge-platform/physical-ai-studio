# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for pure-Python (no `cosign` binary) image signature verification.

Unlike the rest of `tests/services/ssh/`, these hit the real network: the
real `ghcr.io` registry and the real Sigstore public-good instance (Fulcio
root, Rekor, the TUF metadata CDN), against this project's own actual
published, `cosign`-signed trainer images. That is the whole point of this
module - to prove `services.ssh.oci_registry` and `services.ssh.sigstore_verify`
work against real infrastructure, not just a scripted fake - so it is marked
`integration` and skipped by a plain `uv run pytest` the same as any other
network-dependent test in this suite.
"""

from __future__ import annotations

import pytest

from services.ssh import sigstore_verify
from settings import Settings

pytestmark = pytest.mark.integration

_TRAINER_IMAGE = "ghcr.io/open-edge-platform/physicalai-trainer-cuda:protocol-1"
_DEFAULT_SETTINGS = Settings()


async def test_verifies_the_real_published_trainer_image() -> None:
    """The actual `protocol-1` trainer image Studio resolves for SSH provisioning verifies clean."""
    await sigstore_verify.verify_signature(
        _TRAINER_IMAGE,
        identity_regexp=_DEFAULT_SETTINGS.cosign_certificate_identity_regexp,
        oidc_issuer=_DEFAULT_SETTINGS.cosign_oidc_issuer,
    )


async def test_rejects_the_real_image_under_an_unrelated_identity() -> None:
    """A real, validly-signed image still fails a policy pinned to the wrong identity."""
    with pytest.raises(sigstore_verify.SignatureVerificationError):
        await sigstore_verify.verify_signature(
            _TRAINER_IMAGE,
            identity_regexp=r"https://github\.com/some-other-org/some-other-repo/\.github/workflows/.+",
            oidc_issuer=_DEFAULT_SETTINGS.cosign_oidc_issuer,
        )


async def test_reports_unavailable_for_an_unreachable_registry() -> None:
    """A registry that cannot be reached at all raises the distinct 'unavailable' error."""
    with pytest.raises(sigstore_verify.SignatureUnavailableError):
        await sigstore_verify.verify_signature(
            "registry.invalid.example/does-not-exist:latest",
            identity_regexp=_DEFAULT_SETTINGS.cosign_certificate_identity_regexp,
            oidc_issuer=_DEFAULT_SETTINGS.cosign_oidc_issuer,
        )
