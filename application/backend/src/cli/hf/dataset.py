# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset commands and their inspection/import/export workflows."""

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import click

from cli.hf.output import invoke


@click.group()
def dataset() -> None:
    """Inspect, import, and publish LeRobot v3 datasets on the Hub.

    REPO_ID is OWNER/REPO. Import needs an existing local project, not an environment
    UUID: it creates a temporary empty environment. For local folders, use the
    separate datasets import-dir command, which has different requirements.
    """


@dataset.command("inspect")
@click.argument("repo_id")
@click.option("--revision", default="main", show_default=True, help="Source branch, tag, or commit SHA.")
@click.option("--subfolder", help="Dataset root within the repository.")
@click.option("--json", "as_json", is_flag=True, help="Emit one JSON object on stdout; diagnostics go to stderr.")
def inspect_dataset(repo_id: str, revision: str, subfolder: str | None, as_json: bool) -> None:
    """Inspect OWNER/REPO without downloading Parquet shards or videos.

    Reads bounded JSON metadata and file listings. JSON includes features,
    cameras, size estimate, resolved revision SHA, warnings, and importable.
    Exit code 0 means inspection succeeded, not necessarily importable=true.
    Metadata compatibility does not verify every data row or decode videos.

    \b
    Example:
      physicalai-studio hf dataset inspect OWNER/DATASET --json
    """
    from cli.hf.inspection import inspect_remote_dataset

    invoke(inspect_remote_dataset, as_json=as_json, repo_id=repo_id, revision=revision, subfolder=subfolder)


@dataset.command("import")
@click.argument("repo_id", required=False)
@click.option("--project-id", type=click.UUID, help="Target project (required for a new import).")
@click.option("--name", help="Override the dataset display name.")
@click.option("--revision", help="Source branch, tag, or commit (default: main).")
@click.option("--subfolder", help="Dataset root within the repository.")
@click.option("--resume", type=click.UUID, help="Resume a transfer UUID; use only this option and optionally --json.")
@click.option("--dry-run", is_flag=True, help="Read metadata and check project/disk space; do not download payload.")
@click.option("--json", "as_json", is_flag=True, help="Emit one JSON object on stdout; diagnostics go to stderr.")
def import_dataset(  # noqa: PLR0913, PLR0917
    repo_id: str | None,
    project_id: UUID | None,
    name: str | None,
    revision: str | None,
    subfolder: str | None,
    resume: UUID | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Download a LeRobot v3.0 dataset from OWNER/REPO into local Studio storage.

    A new import requires REPO_ID and --project-id. Creates a new dataset and
    temporary empty environment atomically; no robots or cameras are configured.
    No environment or existing dataset UUID is needed. LeRobot v2 conversion is not
    supported. Inspect first and pin its returned commit with --revision.

    Resume uses saved options; omit REPO_ID and other transfer options except
    --json. Preserve staging/cache files to reuse downloaded data.

    \b
    Examples (replace UUID placeholders):
      physicalai-studio hf dataset import OWNER/DATASET --project-id PROJECT_UUID --dry-run --json
      physicalai-studio hf dataset import --resume TRANSFER_UUID --json
    """
    invoke(
        _import_dataset,
        as_json=as_json,
        repo_id=repo_id,
        project_id=project_id,
        name=name,
        revision=revision,
        subfolder=subfolder,
        resume=resume,
        dry_run=dry_run,
    )


@dataset.command("export")
@click.argument("repo_id", required=False)
@click.option("--dataset-id", type=click.UUID, help="Local Studio dataset UUID (required for a new export).")
@click.option(
    "--public", is_flag=True, help="Create a public repo instead of private; existing visibility is unchanged."
)
@click.option(
    "--update", is_flag=True, help="Authorize overwriting a nonempty repo; stale remote files still cause failure."
)
@click.option("--resume", type=click.UUID, help="Resume a transfer UUID; use only this option and optionally --json.")
@click.option(
    "--dry-run", is_flag=True, help="Check local files and remote destination without creating a repo or uploading."
)
@click.option("--json", "as_json", is_flag=True, help="Emit one JSON object on stdout; diagnostics go to stderr.")
def export_dataset(
    repo_id: str | None,
    dataset_id: UUID | None,
    public: bool,
    update: bool,
    resume: UUID | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Publish an existing local dataset to OWNER/REPO in native LeRobot layout.

    A new export requires REPO_ID and --dataset-id; the local record supplies the
    project. New repositories default to private. Existing payloads require
    --update, but stale shards are not silently merged or deleted.

    Stop dataset recording before export; this CLI does not lock recording.
    Resume requires unchanged source files. Omit REPO_ID and other transfer
    options when using --resume (except --json).

    \b
    Examples (replace UUID placeholders):
      physicalai-studio hf dataset export OWNER/DATASET --dataset-id DATASET_UUID --dry-run --json
      physicalai-studio hf dataset export --resume TRANSFER_UUID --json
    """
    invoke(
        _export_dataset,
        as_json=as_json,
        repo_id=repo_id,
        dataset_id=dataset_id,
        public=public,
        update=update,
        resume=resume,
        dry_run=dry_run,
    )


def _import_dataset(
    repo_id: str | None,
    project_id: UUID | None,
    name: str | None,
    revision: str | None,
    subfolder: str | None,
    resume: UUID | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Import a LeRobot dataset, creating its temporary environment at registration."""
    import asyncio

    from cli.hf import transfer
    from cli.hf.inspection import inspect_remote_dataset
    from cli.hf.storage import validate_target

    if resume:
        if any(value is not None for value in (repo_id, project_id, name, revision, subfolder)) or dry_run:
            raise ValueError("--resume cannot be combined with new transfer options")
        directory, state = transfer._load(resume, "dataset", "import")
    else:
        if not repo_id or not project_id:
            raise ValueError("A new import requires REPO_ID and --project-id")
        if name is not None and (not name.strip() or len(name) > 255):
            raise ValueError("--name must contain 1 to 255 characters")
        asyncio.run(validate_target(str(project_id)))
        inspection = inspect_remote_dataset(repo_id, revision or "main", subfolder)
        transfer._require_importable(inspection)
        if not inspection["size_complete"]:
            raise ValueError("Hub did not report all payload sizes; cannot enforce import disk limits")
        from settings import get_settings

        transfer._space(get_settings().datasets_dir, inspection["size_bytes"])
        if dry_run:
            return {**inspection, "dry_run": True, "project_id": str(project_id)}
        directory, state = transfer._new(
            "dataset",
            "import",
            repo_id=repo_id,
            project_id=str(project_id),
            name=name,
            revision=inspection["revision"],
            inspection=inspection,
            asset_id=str(uuid4()),
            environment_id=str(uuid4()),
        )
    with transfer._transfer(directory, state):
        if state["phase"] != "completed":
            state["result"] = _download_dataset(directory, state)
            state["phase"] = "completed"
            transfer._save(directory, state)
    return {**state["result"], "transfer_id": state["transfer_id"], "resume_command": transfer._resume_command(state)}


def _download_dataset(directory: Path, state: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    from cli.hf import transfer
    from cli.hf.inspection import inspect_local_dataset
    from cli.hf.storage import register_dataset_import, validate_dataset_payload, validate_target
    from settings import get_settings

    parent = get_settings().datasets_dir
    destination = parent / state["asset_id"]
    asyncio.run(validate_target(state["project_id"]))
    if not destination.exists():
        root = transfer._download_files(state, parent)
        transfer._require_importable(inspect_local_dataset(root))
        validate_dataset_payload(root)
        root.rename(destination)
        state["phase"] = "downloaded"
        transfer._save(directory, state)
    verified = inspect_local_dataset(destination)
    transfer._require_importable(verified)
    validate_dataset_payload(destination)
    return asyncio.run(register_dataset_import(state, destination, verified))


def _export_dataset(
    repo_id: str | None,
    dataset_id: UUID | None,
    public: bool,
    update: bool,
    resume: UUID | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Publish a LeRobot dataset, validating its task/episode shard references."""
    import asyncio
    import shutil

    from cli.hf import transfer
    from cli.hf.inspection import inspect_local_dataset
    from cli.hf.storage import dataset_export_source, validate_dataset_payload

    if resume:
        if repo_id is not None or dataset_id is not None or public or update or dry_run:
            raise ValueError("--resume cannot be combined with new transfer options")
        directory, state = transfer._load(resume, "dataset", "export")
    else:
        from huggingface_hub.utils import validate_repo_id

        if not repo_id or not dataset_id:
            raise ValueError("A new export requires REPO_ID and --dataset-id")
        validate_repo_id(repo_id)
        source = asyncio.run(dataset_export_source(str(dataset_id)))
        inspection = inspect_local_dataset(Path(source["path"]))
        transfer._require_importable(inspection)
        validate_dataset_payload(Path(source["path"]))
        names = [name for name in inspection["files"] if name != "studio.json"]
        remote = transfer._remote_destination(transfer.hub_api(), "dataset", repo_id, names, update)
        preserved = remote & {"README.md", "LICENSE", "LICENSE.md"}
        names = [name for name in names if name not in preserved]
        fingerprints = transfer._fingerprint(Path(source["path"]), names)
        if dry_run:
            return {
                "dry_run": True,
                "kind": "dataset",
                "repo_id": repo_id,
                "files": names,
                "size_bytes": inspection["size_bytes"],
                "private_if_created": not public,
            }
        directory, state = transfer._new(
            "dataset",
            "export",
            repo_id=repo_id,
            asset_id=str(dataset_id),
            source=source,
            files=names,
            fingerprints=fingerprints,
            public=public,
            update=update,
        )
    state.setdefault("url", f"https://huggingface.co/datasets/{state['repo_id']}")
    with transfer._transfer(directory, state):
        if state["phase"] != "completed":
            _check_dataset_source(state)
            transfer._upload_payload(directory, state)
            _check_dataset_source(state)
            state["result"] = transfer._complete_upload(state)
            state["phase"] = "completed"
            transfer._save(directory, state)
        shutil.rmtree(directory / "payload", ignore_errors=True)
    return {**state["result"], "transfer_id": state["transfer_id"], "resume_command": transfer._resume_command(state)}


def _check_dataset_source(state: dict[str, Any]) -> None:
    import asyncio

    from cli.hf import transfer
    from cli.hf.inspection import inspect_local_dataset
    from cli.hf.storage import dataset_export_source

    current = asyncio.run(dataset_export_source(state["asset_id"]))
    transfer._check_source(state, current, inspect_local_dataset(Path(current["path"])))
