# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared resumable transfer state and Hub I/O; resource workflows live beside commands."""

import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import click

from cli.hf.inspection import hub_api, safe_path, token


class TransferError(ValueError):
    """An interrupted operation with a machine-readable recovery command."""

    def __init__(self, cause: BaseException, state: dict[str, Any]) -> None:
        self.resume_command = _resume_command(state)
        self.transfer_id = state["transfer_id"]
        super().__init__(f"{str(cause) or 'Transfer interrupted'}\nResume: {self.resume_command}")


def _base() -> Path:
    from settings import get_settings

    return get_settings().cache_dir / "hf" / "transfers"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _save(directory: Path, state: dict[str, Any]) -> None:
    _atomic_json(directory / "transfer.json", state)


def _load(resume: UUID, kind: str, operation: str) -> tuple[Path, dict[str, Any]]:
    directory = _base() / str(resume)
    try:
        state = json.loads((directory / "transfer.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read transfer {resume}. Use the original STORAGE_DIR.") from exc
    if state.get("schema_version") != 1 or state.get("kind") != kind or state.get("operation") != operation:
        raise ValueError("Transfer version, resource kind, or operation does not match this command")
    return directory, state


def _new(kind: str, operation: str, **options: Any) -> tuple[Path, dict[str, Any]]:
    transfer_id = str(uuid4())
    state = {
        "schema_version": 1,
        "transfer_id": transfer_id,
        "kind": kind,
        "operation": operation,
        "phase": "prepared",
        **options,
    }
    directory = _base() / transfer_id
    _save(directory, state)
    return directory, state


def _resume_command(state: dict[str, Any]) -> str:
    return f"physicalai-studio hf {state['kind']} {state['operation']} --resume {state['transfer_id']}"


@contextmanager
def _transfer(directory: Path, state: dict[str, Any]) -> Iterator[None]:
    from huggingface_hub.utils import WeakFileLock

    command = _resume_command(state)
    click.echo(f"Transfer: {state['transfer_id']}\nResume: {command}", err=True)
    try:
        with WeakFileLock(directory / "transfer.lock", timeout=0.1):
            # Read again under the lock: another invocation may have advanced state.
            state.update(json.loads((directory / "transfer.json").read_text()))
            yield
    except (Exception, KeyboardInterrupt) as exc:
        raise TransferError(exc, state) from exc


def _require_importable(inspection: dict[str, Any]) -> None:
    if not inspection["importable"]:
        raise ValueError("Resource cannot be imported: " + "; ".join(inspection["errors"]))


def _space(path: Path, required: int) -> None:
    from settings import get_settings

    settings = get_settings()
    if required > settings.data_import_max_uncompressed_bytes:
        raise ValueError("Transfer exceeds DATA_IMPORT_MAX_UNCOMPRESSED_BYTES")
    existing = path
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < required + settings.data_import_min_free_bytes:
        raise ValueError("Insufficient disk space (including DATA_IMPORT_MIN_FREE_BYTES headroom)")


def _download_files(state: dict[str, Any], parent: Path) -> Path:
    """Resume the pinned payload into staging on the destination filesystem."""
    from huggingface_hub import snapshot_download

    from settings import get_settings

    settings = get_settings()
    inspection = state["inspection"]
    if inspection["size_bytes"] > settings.data_import_max_uncompressed_bytes:
        raise ValueError("Transfer exceeds DATA_IMPORT_MAX_UNCOMPRESSED_BYTES")
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".hf-{state['transfer_id']}"
    staging.mkdir(exist_ok=True)
    prefix = f"{inspection['subfolder']}/" if inspection["subfolder"] else ""
    names = [f"{prefix}{name}" for name in inspection["files"]]
    existing_bytes = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
    _space(staging, max(0, inspection["size_bytes"] - existing_bytes))
    click.echo(f"Downloading {state['repo_id']} @ {state['revision']}", err=True)
    snapshot_download(
        state["repo_id"],
        repo_type=state["kind"],
        revision=state["revision"],
        local_dir=staging,
        allow_patterns=names,
        token=token(),
    )
    return staging / inspection["subfolder"]


def _fingerprint(root: Path, names: list[str]) -> dict[str, list[int]]:
    result = {}
    for name in names:
        path = root / safe_path(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe source file: {name}")
        stat = path.stat()
        result[name] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    return result


def _remote_destination(api: Any, kind: str, repo_id: str, names: list[str], update: bool) -> set[str]:
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        remote = set(api.list_repo_files(repo_id, repo_type=kind))
    except RepositoryNotFoundError:
        return set()
    payload = remote - {".gitattributes", "README.md", "LICENSE", "LICENSE.md", "studio.json"}
    if (payload or "studio.json" in remote) and not update:
        raise ValueError("Destination is nonempty. Use --update to explicitly authorize overwriting it.")
    stale = payload - set(names)
    if stale:
        raise ValueError(f"Update would retain stale files: {sorted(stale)}. Export to a new repository instead.")
    return remote


def _check_source(state: dict[str, Any], current: dict[str, Any], inspection: dict[str, Any]) -> None:
    if current != state["source"] or _fingerprint(Path(current["path"]), state["files"]) != state["fingerprints"]:
        raise ValueError("Export source changed. Start a new transfer rather than resuming mixed content.")
    _require_importable(inspection)
    current_files = inspection["files"]
    ignored = {"studio.json", "README.md", "LICENSE", "LICENSE.md"}
    if sorted(set(current_files) - ignored) != sorted(set(state["files"]) - ignored):
        raise ValueError("Export source file inventory changed. Start a new transfer.")


def _upload_payload(directory: Path, state: dict[str, Any]) -> None:
    root = Path(state["source"]["path"])
    payload = directory / "payload"
    payload.mkdir(exist_ok=True)
    for name in state["files"]:
        target = payload / name
        if target.exists():
            if target.stat().st_size != state["fingerprints"][name][0]:
                raise ValueError(f"Staged export file changed: {name}. Start a new transfer.")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.hf-copy")
        temporary.unlink(missing_ok=True)
        try:
            os.link(root / name, temporary)
        except OSError:
            _space(target.parent, (root / name).stat().st_size)
            shutil.copy2(root / name, temporary)
        temporary.replace(target)
    api = hub_api()
    kind, repo_id = state["kind"], state["repo_id"]
    if state["phase"] == "prepared":
        _remote_destination(api, kind, repo_id, state["files"], state["update"])
        api.create_repo(repo_id, repo_type=kind, private=not state["public"], exist_ok=True)
        state["phase"] = "uploading"
        _save(directory, state)
    descriptor, head = _remote_descriptor(api, kind, repo_id)
    if descriptor.get("transfer_id") == state["transfer_id"] and descriptor.get("status") == "complete":
        return
    if descriptor.get("status") == "uploading" and descriptor.get("transfer_id") != state["transfer_id"]:
        raise ValueError("Another transfer owns this incomplete remote export. Resume that transfer first.")
    metadata = {
        "schema_version": 1,
        "kind": kind,
        "status": "uploading",
        "transfer_id": state["transfer_id"],
        **state["source"]["metadata"],
        "files": state["files"],
    }
    api.upload_file(
        repo_id=repo_id,
        repo_type=kind,
        path_in_repo="studio.json",
        path_or_fileobj=json.dumps(metadata).encode(),
        commit_message="Mark Studio export in progress",
        parent_commit=head,
    )
    click.echo(f"Uploading {len(state['files'])} files to {repo_id}. Ctrl-C retains resume state.", err=True)
    _large_upload(repo_id, kind, payload, state["files"])


def _complete_upload(state: dict[str, Any]) -> dict[str, Any]:
    api = hub_api()
    kind, repo_id = state["kind"], state["repo_id"]
    descriptor, head = _remote_descriptor(api, kind, repo_id)
    if descriptor.get("transfer_id") != state["transfer_id"]:
        raise ValueError("Remote export ownership changed; refusing to publish completion metadata")
    inventory = api.repo_info(repo_id, repo_type=kind, revision=head, files_metadata=True)
    sizes = {item.rfilename: item.size for item in inventory.siblings}
    if any(sizes.get(name) != state["fingerprints"][name][0] for name in state["files"]):
        raise ValueError("Remote payload is missing files or has unexpected sizes; resume before publishing completion")
    if descriptor.get("status") == "complete":
        return _export_result(state, head)
    metadata = {
        "schema_version": 1,
        "kind": kind,
        "status": "complete",
        "transfer_id": state["transfer_id"],
        **state["source"]["metadata"],
        "files": state["files"],
    }
    commit = api.upload_file(
        repo_id=repo_id,
        repo_type=kind,
        path_in_repo="studio.json",
        path_or_fileobj=json.dumps(metadata, indent=2).encode(),
        commit_message="Complete Studio export",
        parent_commit=head,
    )
    return _export_result(state, commit.oid)


def _export_result(state: dict[str, Any], revision: str) -> dict[str, Any]:
    return {
        "kind": state["kind"],
        "repo_id": state["repo_id"],
        "id": state["asset_id"],
        "revision": revision,
        "url": state["url"],
    }


def _remote_descriptor(api: Any, kind: str, repo_id: str) -> tuple[dict[str, Any], str]:
    from huggingface_hub import hf_hub_download

    from cli.hf.inspection import MAX_METADATA_BYTES, read_json_file

    info = api.repo_info(repo_id, repo_type=kind, files_metadata=True)
    descriptor = next((item for item in info.siblings if item.rfilename == "studio.json"), None)
    if descriptor is None:
        return {}, info.sha
    if descriptor.size is None or descriptor.size > MAX_METADATA_BYTES:
        raise ValueError("Remote studio.json is oversized; refusing to overwrite it")
    path = hf_hub_download(repo_id, "studio.json", repo_type=kind, revision=info.sha, token=token())
    return read_json_file(Path(path)), info.sha


def _upload_worker(repo_id: str, kind: str, payload: str, names: list[str], credential: str | bool, pipe: Any) -> None:
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


def _large_upload(repo_id: str, kind: str, payload: Path, names: list[str]) -> None:
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
