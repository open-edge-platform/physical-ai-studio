# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local database boundaries for the standalone HF commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from cli.hf.schemas import DATA_PATH, VIDEO_PATH, AssetMetadata, DatasetInfo, Inspection, TransferResult, feature_flags
from cli.hf.state import ExportSource, ImportState

if TYPE_CHECKING:
    from collections.abc import Mapping


async def validate_target(project_id: str, dataset_id: str | None = None) -> None:
    """Validate local associations before starting a network transfer."""
    from db import get_async_db_session_ctx
    from db.schema import DatasetDB, ProjectDB

    async with get_async_db_session_ctx() as session:
        if await session.get(ProjectDB, project_id) is None:
            raise ValueError(f"Project {project_id} does not exist in this Studio database")
        if dataset_id:
            dataset = await session.get(DatasetDB, dataset_id)
            if dataset is None or dataset.project_id != project_id:
                raise ValueError("--dataset-id must identify a dataset in the target project")


async def model_export_source(model_id: str) -> ExportSource:
    """Read a model and its portable metadata; reject active training jobs."""
    from db import get_async_db_session_ctx
    from db.schema import JobDB, ModelDB
    from settings import get_settings

    settings = get_settings()
    async with get_async_db_session_ctx() as session:
        record = await session.get(ModelDB, model_id)
        if record is None:
            raise ValueError(f"Local model {model_id} does not exist")
        job = await session.get(JobDB, record.train_job_id) if record.train_job_id else None
        if job and job.status in {"pending", "running"}:
            raise ValueError("Cannot export a model while its training job is active")
        path = Path(record.path)
        metadata = AssetMetadata(
            name=record.name,
            source_id=record.id,
            source_project_id=record.project_id,
            policy=record.policy,
            version=record.version,
            source_dataset_id=record.dataset_id,
            properties=feature_flags(cast("Mapping[str, object]", record.properties)),
        )
        _validate_source_path(path, settings.models_dir)
        return ExportSource(path.resolve(), metadata)


async def dataset_export_source(dataset_id: str) -> ExportSource:
    """Read a dataset and its portable task/name metadata."""
    from db import get_async_db_session_ctx
    from db.schema import DatasetDB
    from settings import get_settings

    settings = get_settings()
    async with get_async_db_session_ctx() as session:
        record = await session.get(DatasetDB, dataset_id)
        if record is None:
            raise ValueError(f"Local dataset {dataset_id} does not exist")
        path = settings.datasets_dir / record.id
        _validate_source_path(path, settings.datasets_dir)
        return ExportSource(
            path.resolve(),
            AssetMetadata(
                name=record.name,
                source_id=record.id,
                source_project_id=record.project_id,
                default_task=record.default_task,
            ),
        )


def _validate_source_path(path: Path, parent: Path) -> None:
    if path.is_symlink() or not path.resolve().is_relative_to(parent.resolve()) or not path.is_dir():
        raise ValueError("Resource directory is missing or outside managed storage")


def _import_name(state: ImportState, metadata: AssetMetadata) -> str:
    name = state.name or metadata.name or state.repo_id.rsplit("/", 1)[-1]
    if not isinstance(name, str) or not name.strip() or len(name) > 255:
        raise ValueError("Asset name must contain 1 to 255 characters")
    return name


async def register_model_import(state: ImportState, destination: Path, inspection: Inspection) -> TransferResult:
    """Register a model once without inventing training history or foreign links."""
    from db import get_async_db_session_ctx
    from db.schema import ModelDB

    asset_id = state.asset_id
    project_id = state.project_id
    dataset_id = state.dataset_id
    await validate_target(project_id, dataset_id)
    metadata = inspection.studio or AssetMetadata()
    if inspection.model is None or inspection.model.policy is None:
        raise ValueError("Model inspection must identify a supported policy")
    name = _import_name(state, metadata)
    async with get_async_db_session_ctx() as session:
        existing = await session.get(ModelDB, asset_id)
        if existing is not None:
            if existing.project_id != project_id or Path(existing.path).resolve() != destination.resolve():
                raise ValueError("Saved import ID conflicts with an existing resource")
        else:
            session.add(
                ModelDB(
                    id=asset_id,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    name=name,
                    path=str(destination),
                    policy=inspection.model.policy,
                    properties=feature_flags(metadata.properties),
                    version=metadata.version,
                    # Do not invent training history or restore foreign job/snapshot IDs.
                    train_job_id=None,
                    snapshot_id=None,
                    parent_model_id=None,
                )
            )
        await session.commit()
    return TransferResult(
        kind="model",
        id=asset_id,
        project_id=project_id,
        path=str(destination),
        repo_id=state.repo_id,
        revision=state.revision,
    )


async def register_dataset_import(
    state: ImportState,
    destination: Path,
    inspection: Inspection,
) -> TransferResult:
    """Register a dataset and temporary empty environment in one transaction."""
    from db import get_async_db_session_ctx
    from db.schema import DatasetDB, ProjectEnvironmentDB

    asset_id, project_id = state.asset_id, state.project_id
    await validate_target(project_id)
    metadata = inspection.studio or AssetMetadata()
    name = _import_name(state, metadata)
    async with get_async_db_session_ctx() as session:
        existing = await session.get(DatasetDB, asset_id)
        if existing is not None:
            if existing.project_id != project_id or Path(existing.path).resolve() != destination.resolve():
                raise ValueError("Saved import ID conflicts with an existing resource")
        else:
            # TODO: Remove this placeholder environment creation once datasets can
            # be registered without an environment. Keep it in the same transaction.
            environment = ProjectEnvironmentDB(
                id=state.environment_id, project_id=project_id, name=f"Imported dataset {asset_id}"
            )
            session.add(environment)
            await session.flush()
            session.add(
                DatasetDB(
                    id=asset_id,
                    project_id=project_id,
                    environment_id=environment.id,
                    name=name,
                    path=str(destination),
                    default_task=metadata.default_task,
                )
            )
        await session.commit()
    return TransferResult(
        kind="dataset",
        id=asset_id,
        project_id=project_id,
        path=str(destination),
        repo_id=state.repo_id,
        revision=state.revision,
        environment_id=state.environment_id,
    )


def validate_dataset_payload(root: Path) -> None:
    """Check episode shard references without decoding videos or opening hardware."""
    import pandas as pd

    from cli.hf.inspection import read_json_file

    info = DatasetInfo.model_validate(read_json_file(root / "meta/info.json"))
    for field, template in (("data_path", DATA_PATH), ("video_path", VIDEO_PATH)):
        if getattr(info, field) != template:
            raise ValueError(f"Unsupported LeRobot {field} template: {getattr(info, field)}")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if tasks.empty:
        raise ValueError("Dataset task index is empty")
    paths = {"data": info.data_path} | {
        f"videos/{key}": info.video_path.replace("{video_key}", key) for key in info.cameras
    }
    for shard in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        frame = pd.read_parquet(shard)
        if frame.empty or "episode_index" not in frame:
            raise ValueError(f"Empty or invalid episode metadata: {shard.name}")
        for prefix, template in paths.items():
            columns = [f"{prefix}/chunk_index", f"{prefix}/file_index"]
            if not all(column in frame for column in columns):
                raise ValueError(f"Episode metadata is missing shard references: {columns}")
            for chunk, index in frame[columns].drop_duplicates().itertuples(index=False, name=None):
                path = root / template.format(chunk_index=int(chunk), file_index=int(index))
                if not path.is_file():
                    raise ValueError(f"Episode references missing payload {path.relative_to(root)}")
