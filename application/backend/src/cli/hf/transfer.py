# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared resumable transfer state and Hub I/O; resource workflows live beside commands."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import click

from cli.hf.inspection import CARDS_AND_LICENSES, hub_api, repo_inventory, safe_path, token
from cli.hf.schemas import Inspection, JsonObject, ResourceKind, TransferResult, to_json
from cli.hf.state import ExportSource, ExportState, FileFingerprint, ImportState, TransferState

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from multiprocessing.connection import Connection

    from huggingface_hub import HfApi


class TransferError(ValueError):
    """An interrupted operation with a machine-readable recovery command."""

    def __init__(self, cause: BaseException, state: TransferState) -> None:
        self.resume_command = state.resume_command
        self.transfer_id = state.transfer_id
        super().__init__(f"{str(cause) or 'Transfer interrupted'}\nResume: {self.resume_command}")


@contextmanager
def locked_transfer[T: TransferState](state: T) -> Iterator[T]:
    """Lock a transfer and wrap failures with its recovery command."""
    from huggingface_hub.utils import WeakFileLock

    click.echo(f"Transfer: {state.transfer_id}\nResume: {state.resume_command}", err=True)
    try:
        with WeakFileLock(state.directory / "transfer.lock", timeout=0.1):
            # Read again under the lock: another invocation may have advanced state.
            yield state.load(UUID(state.transfer_id), state.kind)
    except (Exception, KeyboardInterrupt) as exc:
        raise TransferError(exc, state) from exc


def check_disk_space(path: Path, required: int) -> None:
    """Enforce import limits and reserve the configured free-space headroom."""
    from settings import get_settings

    settings = get_settings()
    if required > settings.data_import_max_uncompressed_bytes:
        raise ValueError("Transfer exceeds DATA_IMPORT_MAX_UNCOMPRESSED_BYTES")
    existing = path
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < required + settings.data_import_min_free_bytes:
        raise ValueError("Insufficient disk space (including DATA_IMPORT_MIN_FREE_BYTES headroom)")


def download_files(state: ImportState, parent: Path) -> Path:
    """Resume the pinned payload into staging on the destination filesystem."""
    from huggingface_hub import snapshot_download

    from settings import get_settings

    settings = get_settings()
    inspection = state.inspection
    if inspection.size_bytes > settings.data_import_max_uncompressed_bytes:
        raise ValueError("Transfer exceeds DATA_IMPORT_MAX_UNCOMPRESSED_BYTES")
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".hf-{state.transfer_id}"
    staging.mkdir(exist_ok=True)
    prefix = f"{inspection.subfolder}/" if inspection.subfolder else ""
    names = [f"{prefix}{name}" for name in inspection.files]
    existing_bytes = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
    check_disk_space(staging, max(0, inspection.size_bytes - existing_bytes))
    click.echo(f"Downloading {state.repo_id} @ {state.revision}", err=True)
    snapshot_download(
        state.repo_id,
        repo_type=state.kind,
        revision=state.revision,
        local_dir=staging,
        allow_patterns=names,
        token=token(),
    )
    return staging / (inspection.subfolder or "")


def fingerprint_files(root: Path, names: list[str]) -> dict[str, FileFingerprint]:
    """Capture source file identities without reading multi-GB payloads."""
    result = {}
    for name in names:
        path = root / safe_path(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe source file: {name}")
        stat = path.stat()
        result[name] = FileFingerprint(stat.st_size, stat.st_mtime_ns, stat.st_ino)
    return result


def _remote_destination(
    api: HfApi, kind: ResourceKind, repo_id: str, names: list[str], update: bool, *, revision: str | None = None
) -> set[str]:
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        remote = set(api.list_repo_files(repo_id, repo_type=kind, revision=revision))
    except RepositoryNotFoundError:
        return set()
    payload = remote - CARDS_AND_LICENSES - {".gitattributes", "studio.json"}
    if (payload or "studio.json" in remote) and not update:
        raise ValueError("Destination is nonempty. Use --update to explicitly authorize overwriting it.")
    stale = payload - set(names)
    if stale:
        raise ValueError(f"Update would retain stale files: {sorted(stale)}. Export to a new repository instead.")
    return remote


def check_source(state: ExportState, current: ExportSource, inspection: Inspection) -> None:
    """Reject source mutations and incompatible payload inventories."""
    if current != state.source or fingerprint_files(current.path, state.files) != state.fingerprints:
        raise ValueError("Export source changed. Start a new transfer rather than resuming mixed content.")
    inspection.require_importable()
    current_files = inspection.files
    ignored = CARDS_AND_LICENSES | {"studio.json"}
    if set(current_files) - ignored != set(state.files) - ignored:
        raise ValueError("Export source file inventory changed. Start a new transfer.")


def prepare_export(
    kind: ResourceKind,
    repo_id: str,
    asset_id: str,
    source: ExportSource,
    inspection: Inspection,
    public: bool,
    update: bool,
) -> ExportState:
    """Select export files, preserving remote cards and rejecting stale payloads."""
    inspection.require_importable()
    names = [name for name in inspection.files if name != "studio.json"]
    remote = _remote_destination(hub_api(), kind, repo_id, names, update)
    preserved = remote & CARDS_AND_LICENSES
    names = [name for name in names if name not in preserved]
    return ExportState(
        kind=kind,
        repo_id=repo_id,
        asset_id=asset_id,
        source=source,
        files=names,
        fingerprints=fingerprint_files(source.path, names),
        public=public,
        update=update,
    )


def run_export(saved: ExportState, validate_source: Callable[[ExportState], None]) -> JsonObject:
    """Upload unchanged source files and retain only the completed receipt."""
    with locked_transfer(saved) as state:
        if state.phase != "completed":
            validate_source(state)
            _upload_payload(state)
            validate_source(state)
            state.result = _complete_upload(state)
            state.phase = "completed"
            state.save()
        # A completed receipt must remain usable without retained multi-GB payloads.
        shutil.rmtree(state.directory / "payload", ignore_errors=True)
    return state.receipt


def _upload_payload(state: ExportState) -> None:
    root = state.source.path
    payload = state.directory / "payload"
    payload.mkdir(exist_ok=True)
    for name in state.files:
        target = payload / name
        if target.exists():
            if target.stat().st_size != state.fingerprints[name].size:
                raise ValueError(f"Staged export file changed: {name}. Start a new transfer.")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.hf-copy")
        temporary.unlink(missing_ok=True)
        try:
            os.link(root / name, temporary)
        except OSError:
            check_disk_space(target.parent, (root / name).stat().st_size)
            shutil.copy2(root / name, temporary)
        temporary.replace(target)
    api = hub_api()
    kind, repo_id = state.kind, state.repo_id
    if state.phase == "prepared":
        _remote_destination(api, kind, repo_id, state.files, state.update)
        api.create_repo(repo_id, repo_type=kind, private=not state.public, exist_ok=True)
        state.phase = "uploading"
        state.save()
    descriptor, head = _remote_descriptor(api, kind, repo_id)
    if descriptor.get("transfer_id") == state.transfer_id and descriptor.get("status") == "complete":
        return
    if descriptor.get("status") == "uploading" and descriptor.get("transfer_id") != state.transfer_id:
        raise ValueError("Another transfer owns this incomplete remote export. Resume that transfer first.")
    if descriptor.get("transfer_id") != state.transfer_id:
        # Resume state does not authorize overwriting a replacement export. Check
        # the same revision used as the parent of the ownership claim below.
        _remote_destination(api, kind, repo_id, state.files, state.update, revision=head)
    metadata = _export_metadata(state, "uploading")
    api.upload_file(
        repo_id=repo_id,
        repo_type=kind,
        path_in_repo="studio.json",
        path_or_fileobj=json.dumps(metadata).encode(),
        commit_message="Mark Studio export in progress",
        parent_commit=head,
    )
    click.echo(f"Uploading {len(state.files)} files to {repo_id}. Ctrl-C retains resume state.", err=True)
    _large_upload(repo_id, kind, payload, state.files)


def _complete_upload(state: ExportState) -> TransferResult:
    api = hub_api()
    kind, repo_id = state.kind, state.repo_id
    descriptor, head = _remote_descriptor(api, kind, repo_id)
    if descriptor.get("transfer_id") != state.transfer_id:
        raise ValueError("Remote export ownership changed; refusing to publish completion metadata")
    _, sizes = repo_inventory(api, kind, repo_id, head)
    if any(sizes.get(name) != state.fingerprints[name].size for name in state.files):
        raise ValueError("Remote payload is missing files or has unexpected sizes; resume before publishing completion")
    if descriptor.get("status") == "complete":
        return _export_result(state, head)
    metadata = _export_metadata(state, "complete")
    commit = api.upload_file(
        repo_id=repo_id,
        repo_type=kind,
        path_in_repo="studio.json",
        path_or_fileobj=json.dumps(metadata, indent=2).encode(),
        commit_message="Complete Studio export",
        parent_commit=head,
    )
    return _export_result(state, commit.oid)


def _export_metadata(state: ExportState, status: Literal["uploading", "complete"]) -> JsonObject:
    return {
        **to_json(state.source.metadata),
        "schema_version": 1,
        "kind": state.kind,
        "status": status,
        "transfer_id": state.transfer_id,
        "files": [*state.files],
    }


def _export_result(state: ExportState, revision: str) -> TransferResult:
    return TransferResult(kind=state.kind, repo_id=state.repo_id, id=state.asset_id, revision=revision, url=state.url)


def _remote_descriptor(api: HfApi, kind: ResourceKind, repo_id: str) -> tuple[JsonObject, str]:
    from huggingface_hub import hf_hub_download

    from cli.hf.inspection import MAX_METADATA_BYTES, read_json_file

    sha, sizes = repo_inventory(api, kind, repo_id)
    if "studio.json" not in sizes:
        return {}, sha
    size = sizes["studio.json"]
    if size is None or size > MAX_METADATA_BYTES:
        raise ValueError("Remote studio.json is oversized; refusing to overwrite it")
    path = hf_hub_download(repo_id, "studio.json", repo_type=kind, revision=sha, token=token())
    return read_json_file(Path(path)), sha


def _upload_worker(
    repo_id: str, kind: ResourceKind, payload: str, names: list[str], credential: str | bool, pipe: Connection
) -> None:
    from huggingface_hub import HfApi

    try:
        with redirect_stdout(sys.stderr):
            HfApi(token=credential).upload_large_folder(
                repo_id,
                folder_path=payload,
                repo_type=kind,
                allow_patterns=names,
                print_report=False,
            )
        pipe.send(None)
    except (Exception, KeyboardInterrupt) as exc:
        pipe.send(str(exc) or "Upload interrupted")
    finally:
        pipe.close()


def _large_upload(repo_id: str, kind: ResourceKind, payload: Path, names: list[str]) -> None:
    """Isolate SDK worker threads so cancellation cannot leave uploads running."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_upload_worker, args=(repo_id, kind, str(payload), names, token(), sender))
    process.start()
    sender.close()
    try:
        try:
            error = receiver.recv()
        except EOFError as exc:
            raise ValueError("Hub upload worker exited before reporting completion") from exc
        process.join()
        if error is not None or process.exitcode:
            raise ValueError(error or f"Hub upload worker exited with code {process.exitcode}")
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        receiver.close()
