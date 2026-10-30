# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model commands and their inspection/import/export workflows."""

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import click

from cli.hf.output import invoke


@click.group()
def model() -> None:
    """Inspect, import, and publish Studio models on the Hub.

    REPO_ID is OWNER/REPO. Import registers a new local model in an existing
    project and requires a suitable dataset in that project. Export publishes
    existing files; it does not convert a checkpoint to another inference format.
    """


@model.command("inspect")
@click.argument("repo_id")
@click.option("--revision", default="main", show_default=True, help="Source branch, tag, or commit SHA.")
@click.option("--subfolder", help="Model root within the repository.")
@click.option("--json", "as_json", is_flag=True, help="Emit one JSON object on stdout; diagnostics go to stderr.")
def inspect_model(repo_id: str, revision: str, subfolder: str | None, as_json: bool) -> None:
    """Inspect OWNER/REPO without downloading weights or checkpoints.

    Checks Studio manifests, backends, and selected files. JSON includes the
    resolved revision SHA, subfolder, size estimate, warnings, and importable.
    Exit code 0 means inspection succeeded, not necessarily importable=true.
    Metadata compatibility does not prove successful inference.

    \b
    Example:
      physicalai-studio hf model inspect OWNER/MODEL --json
    """
    from cli.hf.inspection import inspect_remote_model

    invoke(inspect_remote_model, as_json=as_json, repo_id=repo_id, revision=revision, subfolder=subfolder)


@model.command("import")
@click.argument("repo_id", required=False)
@click.option("--project-id", type=click.UUID, help="Target project (required for a new import).")
@click.option("--dataset-id", type=click.UUID, help="Local dataset in the target project (required for a new import).")
@click.option("--name", help="Override the model display name.")
@click.option("--revision", help="Source branch, tag, or commit (default: main).")
@click.option("--subfolder", help="Model root within the repository.")
@click.option("--resume", type=click.UUID, help="Resume a transfer UUID; use only this option and optionally --json.")
@click.option(
    "--dry-run", is_flag=True, help="Read metadata and check local associations/disk space; do not download payload."
)
@click.option("--json", "as_json", is_flag=True, help="Emit one JSON object on stdout; diagnostics go to stderr.")
def import_model(  # noqa: PLR0913, PLR0917
    repo_id: str | None,
    project_id: UUID | None,
    dataset_id: UUID | None,
    name: str | None,
    revision: str | None,
    subfolder: str | None,
    resume: UUID | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Download OWNER/REPO and register a new model in local Studio storage.

    A new import requires REPO_ID, --project-id, and --dataset-id. The dataset must
    belong to the target project and match the model's intended environment.
    Supports Studio policy packages, not arbitrary Transformers/checkpoint-only
    repositories. Inspect first and pin its returned commit with --revision.

    Resume uses the saved project, dataset, revision, and name; omit REPO_ID and
    all other transfer options. Preserve staging/cache files between attempts.

    \b
    Examples (replace UUID placeholders):
      physicalai-studio hf model import OWNER/MODEL \\
        --project-id PROJECT_UUID --dataset-id DATASET_UUID --dry-run --json
      physicalai-studio hf model import --resume TRANSFER_UUID --json
    """
    invoke(
        _import_model,
        as_json=as_json,
        repo_id=repo_id,
        project_id=project_id,
        dataset_id=dataset_id,
        name=name,
        revision=revision,
        subfolder=subfolder,
        resume=resume,
        dry_run=dry_run,
    )


@model.command("export")
@click.argument("repo_id", required=False)
@click.option("--model-id", type=click.UUID, help="Local Studio model UUID (required for a new export).")
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
def export_model(
    repo_id: str | None,
    model_id: UUID | None,
    public: bool,
    update: bool,
    resume: UUID | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Publish an existing local model to OWNER/REPO without format conversion.

    A new export requires REPO_ID and --model-id; the local record supplies the
    project. New repositories default to private. Existing payloads require
    --update, but stale-file merges are rejected. Review hparams for secrets.

    Do not train or modify the model during upload. Resume uses retained upload
    state and requires unchanged source files. Omit REPO_ID and other options
    when using --resume (except --json).

    \b
    Examples (replace UUID placeholders):
      physicalai-studio hf model export OWNER/MODEL --model-id MODEL_UUID --dry-run --json
      physicalai-studio hf model export --resume TRANSFER_UUID --json
    """
    invoke(
        _export_model,
        as_json=as_json,
        repo_id=repo_id,
        model_id=model_id,
        public=public,
        update=update,
        resume=resume,
        dry_run=dry_run,
    )


def _import_model(  # noqa: PLR0913, PLR0917
    repo_id: str | None,
    project_id: UUID | None,
    dataset_id: UUID | None,
    name: str | None,
    revision: str | None,
    subfolder: str | None,
    resume: UUID | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Import a Studio model with an explicit local dataset association."""
    import asyncio

    from cli.hf import transfer
    from cli.hf.inspection import inspect_remote_model
    from cli.hf.storage import uuid_string, validate_target

    if resume:
        if any(value is not None for value in (repo_id, project_id, dataset_id, name, revision, subfolder)) or dry_run:
            raise ValueError("--resume cannot be combined with new transfer options")
        directory, state = transfer._load(resume, "model", "import")
    else:
        if not repo_id or not project_id:
            raise ValueError("A new import requires REPO_ID and --project-id")
        if not dataset_id:
            raise ValueError("A model import requires --dataset-id from the target project")
        if name is not None and (not name.strip() or len(name) > 255):
            raise ValueError("--name must contain 1 to 255 characters")
        asyncio.run(validate_target(str(project_id), uuid_string(dataset_id)))
        inspection = inspect_remote_model(repo_id, revision or "main", subfolder)
        transfer._require_importable(inspection)
        if not inspection["size_complete"]:
            raise ValueError("Hub did not report all payload sizes; cannot enforce import disk limits")
        from settings import get_settings

        settings = get_settings()
        transfer._space(settings.models_dir, inspection["size_bytes"])
        if dry_run:
            return {**inspection, "dry_run": True, "project_id": str(project_id), "dataset_id": uuid_string(dataset_id)}
        directory, state = transfer._new(
            "model",
            "import",
            repo_id=repo_id,
            project_id=str(project_id),
            dataset_id=uuid_string(dataset_id),
            name=name,
            revision=inspection["revision"],
            inspection=inspection,
            asset_id=str(uuid4()),
        )
    with transfer._transfer(directory, state):
        if state["phase"] != "completed":
            state["result"] = _download_model(directory, state)
            state["phase"] = "completed"
            transfer._save(directory, state)
    return {**state["result"], "transfer_id": state["transfer_id"], "resume_command": transfer._resume_command(state)}


def _download_model(directory: Path, state: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    from cli.hf import transfer
    from cli.hf.inspection import inspect_local_model
    from cli.hf.storage import register_model_import, validate_target
    from settings import get_settings

    parent = get_settings().models_dir
    destination = parent / state["asset_id"]
    asyncio.run(validate_target(state["project_id"], state["dataset_id"]))
    if not destination.exists():
        root = transfer._download_files(state, parent)
        verified = inspect_local_model(root)
        transfer._require_importable(verified)
        if state["inspection"]["detected_format"] == "studio_exports":
            _normalize_exports(root, destination)
        else:
            root.rename(destination)
        state["phase"] = "downloaded"
        transfer._save(directory, state)
    if state["inspection"]["detected_format"] == "studio_exports":
        _finish_normalization(destination)
    verified = inspect_local_model(destination)
    transfer._require_importable(verified)
    return asyncio.run(register_model_import(state, destination, verified))


def _normalize_exports(root: Path, destination: Path) -> None:
    # Move the complete tree first. A crash before normalization finishes is
    # recoverable because the same normalization runs on the next invocation.
    root.rename(destination)
    _finish_normalization(destination)


def _finish_normalization(destination: Path) -> None:
    from cli.hf import transfer
    from cli.hf.inspection import BACKENDS, read_json_file

    exports = destination / "exports"
    for backend in BACKENDS:
        if (destination / backend).is_dir():
            exports.mkdir(exist_ok=True)
            (destination / backend).rename(exports / backend)
    descriptor = destination / "studio.json"
    if descriptor.is_file():
        metadata = read_json_file(descriptor)
        metadata["files"] = [
            f"exports/{name}" if name.split("/")[0] in BACKENDS else name for name in metadata["files"]
        ]
        transfer._atomic_json(descriptor, metadata)


def _export_model(
    repo_id: str | None,
    model_id: UUID | None,
    public: bool,
    update: bool,
    resume: UUID | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Publish a Studio model, with model-specific source and manifest checks."""
    import asyncio
    import shutil

    from cli.hf import transfer
    from cli.hf.inspection import inspect_local_model
    from cli.hf.storage import model_export_source

    if resume:
        if repo_id is not None or model_id is not None or public or update or dry_run:
            raise ValueError("--resume cannot be combined with new transfer options")
        directory, state = transfer._load(resume, "model", "export")
    else:
        from huggingface_hub.utils import validate_repo_id

        if not repo_id or not model_id:
            raise ValueError("A new export requires REPO_ID and --model-id")
        validate_repo_id(repo_id)
        source = asyncio.run(model_export_source(str(model_id)))
        inspection = inspect_local_model(Path(source["path"]))
        transfer._require_importable(inspection)
        names = [name for name in inspection["files"] if name != "studio.json"]
        remote = transfer._remote_destination(transfer.hub_api(), "model", repo_id, names, update)
        # Publish local cards/licenses to new repos, but never overwrite existing ones.
        preserved = remote & {"README.md", "LICENSE", "LICENSE.md"}
        names = [name for name in names if name not in preserved]
        fingerprints = transfer._fingerprint(Path(source["path"]), names)
        if dry_run:
            return {
                "dry_run": True,
                "kind": "model",
                "repo_id": repo_id,
                "files": names,
                "size_bytes": inspection["size_bytes"],
                "private_if_created": not public,
            }
        directory, state = transfer._new(
            "model",
            "export",
            repo_id=repo_id,
            asset_id=str(model_id),
            source=source,
            files=names,
            fingerprints=fingerprints,
            public=public,
            update=update,
        )
    state.setdefault("url", f"https://huggingface.co/{state['repo_id']}")
    with transfer._transfer(directory, state):
        if state["phase"] != "completed":
            _check_model_source(state)
            transfer._upload_payload(directory, state)
            _check_model_source(state)
            state["result"] = transfer._complete_upload(state)
            state["phase"] = "completed"
            transfer._save(directory, state)
        # Completed transfers need only their receipt, not retained multi-GB payloads.
        shutil.rmtree(directory / "payload", ignore_errors=True)
    return {**state["result"], "transfer_id": state["transfer_id"], "resume_command": transfer._resume_command(state)}


def _check_model_source(state: dict[str, Any]) -> None:
    import asyncio

    from cli.hf import transfer
    from cli.hf.inspection import inspect_local_model
    from cli.hf.storage import model_export_source

    current = asyncio.run(model_export_source(state["asset_id"]))
    transfer._check_source(state, current, inspect_local_model(Path(current["path"])))
