# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Minimal OCI Distribution client for fetching a signature's referrer artifact.

Exists only to answer one question: "does the registry hold a Sigstore bundle
that refers to this exact image digest?" It never pulls a layer of the image
itself.

GHCR (this project's registry) does not implement the OCI 1.1 Referrers API
(confirmed: it 404s on `GET /v2/<repo>/referrers/<digest>`), so this client
uses the fallback convention both Docker Buildx and modern `cosign` use
instead: a referrer targeting digest `sha256:<hex>` is pushed under the tag
`sha256-<hex>` (dash, not colon - tags cannot contain a colon), as an OCI
index whose entries `subject`-reference that digest. This client fetches that
index, finds the entry whose artifact type matches, and returns its one blob.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Final

import httpx

_MANIFEST_ACCEPT: Final = (
    "application/vnd.oci.image.index.v1+json,"
    "application/vnd.docker.distribution.manifest.list.v2+json,"
    "application/vnd.oci.image.manifest.v1+json,"
    "application/vnd.docker.distribution.manifest.v2+json"
)

_BEARER_CHALLENGE: Final = re.compile(
    r'Bearer realm="(?P<realm>[^"]+)"(?:,service="(?P<service>[^"]*)")?(?:,scope="(?P<scope>[^"]*)")?'
)

_REQUEST_TIMEOUT_S: Final = 15.0


class RegistryUnreachableError(Exception):
    """The registry (or its auth endpoint) could not be reached at all.

    Distinct from "no matching referrer found" (a `None` return, which means
    the registry answered but this image has no such artifact): this is a
    connectivity failure. Signature verification always fails closed on this,
    the same as on any other verification failure - see
    `services.ssh.sigstore_verify.verify_signature`.
    """


def _split_digest_reference(digest_reference: str) -> tuple[str, str, str]:
    """Split `<registry>/<repository>@sha256:<hex>` into its three parts."""
    repository_and_tag, _, digest = digest_reference.partition("@")
    registry, _, repository = repository_and_tag.partition("/")
    if not registry or not repository or not digest:
        raise ValueError(f"not a digest reference: {digest_reference!r}")
    return registry, repository, digest


def _split_reference(image_ref: str) -> tuple[str, str, str]:
    """Split `<registry>/<repository>(@sha256:<hex>|:<tag>)` into its three parts."""
    if "@" in image_ref:
        return _split_digest_reference(image_ref)
    repository_and_tag, _, tag = image_ref.rpartition(":")
    registry, _, repository = repository_and_tag.partition("/")
    if not registry or not repository or not tag:
        raise ValueError(f"not an image reference: {image_ref!r}")
    return registry, repository, tag


class _RegistryClient:
    """A short-lived, single-repository client that authenticates once and reuses the token."""

    def __init__(self, http: httpx.AsyncClient, base_url: str) -> None:
        self._http = http
        self._base_url = base_url
        self._token: str | None = None

    async def _authenticate(self, challenge: str) -> None:
        """Exchange a `WWW-Authenticate: Bearer ...` challenge for a token.

        Requests no scope beyond what the challenge itself names (anonymous
        pull), matching how `docker`/`cosign` authenticate against a public
        repository. Leaves `self._token` unset if the challenge is not a
        bearer challenge (e.g. `Basic`) - private-registry credentials are
        out of scope for this client.
        """
        match = _BEARER_CHALLENGE.search(challenge)
        if match is None:
            return
        params = {key: value for key, value in match.groupdict().items() if value is not None}
        realm = params.pop("realm")
        response = await self._http.get(realm, params=params, timeout=_REQUEST_TIMEOUT_S)
        response.raise_for_status()
        body = response.json()
        token = body.get("token") or body.get("access_token")
        self._token = token if isinstance(token, str) else None

    async def get(self, path: str, *, accept: str) -> httpx.Response:
        """GET a path under this repository, authenticating once on a 401 challenge."""
        url = f"{self._base_url}/{path}"
        headers = {"Accept": accept}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        response = await self._http.get(url, headers=headers, timeout=_REQUEST_TIMEOUT_S)
        if response.status_code != httpx.codes.UNAUTHORIZED or self._token is not None:
            return response

        challenge = response.headers.get("WWW-Authenticate")
        if not challenge:
            return response
        await self._authenticate(challenge)
        if self._token is None:
            return response
        headers["Authorization"] = f"Bearer {self._token}"
        return await self._http.get(url, headers=headers, timeout=_REQUEST_TIMEOUT_S)


def _referrer_entries(fallback_tag: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the index entries to check, tolerating a registry that answers with a bare manifest."""
    entries = manifest.get("manifests")
    if entries is not None:
        return entries
    # Not an index: the fallback tag resolved directly to a single manifest.
    return [{"digest": fallback_tag, "mediaType": manifest.get("mediaType", _MANIFEST_ACCEPT)}]


async def resolve_digest(image_ref: str) -> str:
    """Resolve a tag or digest reference to `<registry>/<repository>@sha256:<hex>`.

    A digest reference is returned unchanged (already content-addressed, so
    there is nothing to resolve). A tag reference is resolved via the
    registry's `Docker-Content-Digest` response header - the digest a
    conformant registry is required to return, and the same value `docker
    pull`/`docker manifest inspect` treat as authoritative - falling back to
    hashing the exact manifest bytes received if a registry omits the header.

    Raises:
        RegistryUnreachableError: The registry could not be reached.
    """
    if "@" in image_ref:
        return image_ref

    registry, repository, tag = _split_reference(image_ref)
    try:
        async with httpx.AsyncClient(follow_redirects=True) as http:
            client = _RegistryClient(http, f"https://{registry}/v2/{repository}")
            response = await client.get(f"manifests/{tag}", accept=_MANIFEST_ACCEPT)
            response.raise_for_status()
            digest = response.headers.get("Docker-Content-Digest")
            if not digest:
                digest = f"sha256:{hashlib.sha256(response.content).hexdigest()}"
    except httpx.HTTPError as error:
        raise RegistryUnreachableError(f"could not reach {registry}: {error}") from error

    return f"{registry}/{repository}@{digest}"


async def fetch_referrer_blob(digest_reference: str, *, artifact_media_type: str) -> bytes | None:
    """Return the sole blob of the referrer artifact matching `artifact_media_type`.

    Args:
        digest_reference: `<registry>/<repository>@sha256:<hex>` - the exact
            digest whose referrers are being searched.
        artifact_media_type: The referrer's expected `artifactType` (or, for a
            registry that only preserves it on the child manifest, that
            manifest's `config.mediaType`).

    Returns:
        The raw blob bytes, or `None` if the registry has no referrer of this
        type for this digest (including "no referrers at all").

    Raises:
        RegistryUnreachableError: The registry could not be reached.
    """
    registry, repository, digest = _split_digest_reference(digest_reference)
    fallback_tag = digest.replace(":", "-", 1)

    try:
        async with httpx.AsyncClient(follow_redirects=True) as http:
            client = _RegistryClient(http, f"https://{registry}/v2/{repository}")

            index_response = await client.get(f"manifests/{fallback_tag}", accept=_MANIFEST_ACCEPT)
            if index_response.status_code == httpx.codes.NOT_FOUND:
                return None
            index_response.raise_for_status()

            for entry in _referrer_entries(fallback_tag, index_response.json()):
                child_response = await client.get(
                    f"manifests/{entry['digest']}", accept=entry.get("mediaType", _MANIFEST_ACCEPT)
                )
                if child_response.status_code != httpx.codes.OK:
                    continue
                child = child_response.json()
                child_type = child.get("artifactType") or child.get("config", {}).get("mediaType")
                if child_type != artifact_media_type:
                    continue

                layers = child.get("layers") or []
                if not layers:
                    continue
                blob_response = await client.get(f"blobs/{layers[0]['digest']}", accept="*/*")
                if blob_response.status_code != httpx.codes.OK:
                    continue
                return blob_response.content
    except httpx.HTTPError as error:
        raise RegistryUnreachableError(f"could not reach {registry}: {error}") from error

    return None
