# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Container image signature verification using the `sigstore` PyPI package.

No external binary needed, on this backend or any remote trainer server:
`cosign` signs images by attaching a Sigstore bundle to the registry (see
`services.ssh.oci_registry`), and `sigstore-python` (PyPI: `sigstore`) can
verify that bundle's certificate chain and Rekor inclusion proof in Python.

`sigstore-python` doesn't know how to find a signature in a registry (that's
`cosign`'s convention); `oci_registry` supplies that half, this module the
cryptographic half.

Raises its own two exception types, rather than a backend-specific one, so
it stays usable from both a fail-closed caller
(`services.ssh.docker_ops.verify_image_signature`) and an advisory,
never-blocking caller (`services.ssh.preflight`).
"""

from __future__ import annotations

import json
import re
from typing import Final

from cryptography.x509 import (
    Certificate,
    ExtensionNotFound,
    RFC822Name,
    SubjectAlternativeName,
    UniformResourceIdentifier,
)
from sigstore.errors import Error as SigstoreError
from sigstore.errors import VerificationError as SigstoreVerificationError
from sigstore.models import Bundle
from sigstore.verify import Verifier
from sigstore.verify.policy import AllOf, OIDCIssuer

from services.ssh.oci_registry import RegistryUnreachableError, fetch_referrer_blob, resolve_digest

# `cosign sign --new-bundle-format` wraps even a plain signature (not an
# attestation) in a DSSE envelope with this predicate type and an empty
# predicate. Requiring it here stops an SBOM/provenance attestation - stored
# the same way in the registry - from being accepted as a signature.
_COSIGN_SIGN_PREDICATE_TYPE: Final = "https://sigstore.dev/cosign/sign/v1"

# The artifact type `cosign`'s new bundle format publishes a signature under.
SIGSTORE_BUNDLE_MEDIA_TYPE: Final = "application/vnd.dev.sigstore.bundle.v0.3+json"


class SignatureUnavailableError(Exception):
    """Verification could not be attempted at all.

    Covers the registry being unreachable and Sigstore's own Fulcio/Rekor/TUF
    infrastructure being unreachable - the direct equivalent of the old
    "cosign is not installed" case. Callers that want a fail-open escape
    hatch for this specific class of failure (and only this class) should
    catch it separately from `SignatureVerificationError`.
    """


class SignatureVerificationError(Exception):
    """Verification was attempted and failed.

    Covers no signature bundle existing for this image at all (the direct
    equivalent of `cosign verify` reporting "no matching signatures"), a
    certificate/identity/issuer mismatch, a broken Rekor inclusion proof, and
    a signed subject digest that does not match the image being verified.
    Always meaningful - never downgrade this to a warning outside of a
    deliberately advisory check (see `services.ssh.preflight`).
    """


_verifier: Verifier | None = None


def _production_verifier() -> Verifier:
    """Return a process-wide `Verifier` against Sigstore's public-good instance.

    Constructing one fetches and caches TUF trust-root metadata, so it is
    built once and reused rather than per verification.
    """
    global _verifier  # noqa: PLW0603 - process-wide cache, not a test seam.
    if _verifier is None:
        _verifier = Verifier.production()
    return _verifier


class _IdentityRegexp:
    """Matches a certificate's SAN against a pattern.

    Mirrors `cosign verify --certificate-identity-regexp`: cosign's own flag
    of the same name, matched against the same SAN types (email and URI).
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = re.compile(pattern)

    def verify(self, cert: Certificate) -> None:
        """Verify `cert` against the policy. Raises `SigstoreVerificationError` on failure."""
        try:
            san_ext = cert.extensions.get_extension_for_class(SubjectAlternativeName).value
        except ExtensionNotFound as error:
            raise SigstoreVerificationError(
                f"certificate has no Subject Alternative Name extension to match against {self._pattern.pattern!r}"
            ) from error
        candidates = set(san_ext.get_values_for_type(RFC822Name))
        candidates.update(san_ext.get_values_for_type(UniformResourceIdentifier))
        if not any(self._pattern.fullmatch(candidate) for candidate in candidates):
            raise SigstoreVerificationError(
                f"no certificate SAN matched {self._pattern.pattern!r} (got {sorted(candidates)!r})"
            )


async def verify_signature(image_ref: str, *, identity_regexp: str, oidc_issuer: str) -> None:
    """Verify `image_ref`'s Sigstore-bundle signature, pinned to a certificate identity/issuer.

    `image_ref` may be a tag or a digest reference; a tag is resolved to its
    digest first so the rest of verification (fetching the bundle, checking
    the signed subject) is always digest-scoped.

    Args:
        image_ref: `<registry>/<repository>(@sha256:<hex>|:<tag>)`.
        identity_regexp: Pattern the certificate's SAN must fully match.
        oidc_issuer: The exact OIDC issuer the certificate must carry.

    Raises:
        SignatureUnavailableError: The registry or Sigstore's own
            infrastructure could not be reached.
        SignatureVerificationError: No signature bundle exists for this
            image, or verification failed.
    """
    try:
        digest_reference = await resolve_digest(image_ref)
        bundle_bytes = await fetch_referrer_blob(digest_reference, artifact_media_type=SIGSTORE_BUNDLE_MEDIA_TYPE)
    except RegistryUnreachableError as error:
        raise SignatureUnavailableError(f"could not reach the registry: {error}") from error

    if bundle_bytes is None:
        raise SignatureVerificationError(f"no signature bundle found for '{digest_reference}'")

    policy = AllOf([_IdentityRegexp(identity_regexp), OIDCIssuer(oidc_issuer)])

    try:
        bundle = Bundle.from_json(bundle_bytes)
        # `verify_dsse`'s returned "predicate type" is the DSSE envelope's payload
        # type, not the statement's own `predicateType` field (checked below,
        # which distinguishes a signature from an SBOM/provenance attestation).
        _, payload = _production_verifier().verify_dsse(bundle, policy)
    except SigstoreVerificationError as error:
        raise SignatureVerificationError(str(error)) from error
    except SigstoreError as error:
        # NetworkError/TUFError/RootError: Sigstore's own infrastructure was
        # unreachable, not a verification failure.
        raise SignatureUnavailableError(str(error)) from error

    _check_statement(digest_reference, payload)


def _check_statement(digest_reference: str, payload: bytes) -> None:
    """Confirm the signed in-toto statement is a plain signature over this exact image digest.

    Belt-and-suspenders on top of fetching the bundle by digest and type:
    * `predicateType` must be `cosign`'s plain-signature predicate, not an
      SBOM/provenance attestation stored the same way.
    * The statement's subject digest must match the image being verified.
    """
    try:
        statement = json.loads(payload)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as error:
        raise SignatureVerificationError(f"the signed statement is not valid JSON: {error}") from error
    if not isinstance(statement, dict):
        raise SignatureVerificationError(f"the signed statement is not a JSON object (got {type(statement).__name__})")
    if statement.get("predicateType") != _COSIGN_SIGN_PREDICATE_TYPE:
        raise SignatureVerificationError(f"unexpected predicate type {statement.get('predicateType')!r}")

    _, _, digest = digest_reference.rpartition("@")
    expected = digest.removeprefix("sha256:")
    subjects = statement.get("subject") or []
    if not any(subject.get("digest", {}).get("sha256") == expected for subject in subjects):
        raise SignatureVerificationError(f"the signed statement's subject digest does not match '{digest_reference}'")
