"""Dataset management CLI commands."""

import asyncio
import sys
from pathlib import Path
from uuid import UUID

import click


@click.group()
def datasets() -> None:
    """Import LeRobot v3 datasets from local folders.

    For Hugging Face repositories, use hf dataset instead. Local-folder imports
    require an environment association; Hub imports create a temporary empty one.
    """


@datasets.command("import-dir")
@click.option(
    "--source-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Existing LeRobot v3 dataset directory containing meta/ and data/.",
)
@click.option("--project-id", required=True, type=click.UUID, help="Existing target Studio project UUID.")
@click.option(
    "--environment-id", required=True, type=click.UUID, help="Existing environment UUID in the target project."
)
@click.option("--dataset-name", required=True, type=str, help="Display name for the imported dataset.")
@click.option(
    "--default-task", type=str, default="", show_default=True, help="Default task description stored in Studio."
)
@click.option("--move/--copy", default=False, show_default=True, help="Move removes the source; copy keeps it intact.")
def import_dir(
    source_dir: Path,
    project_id: UUID,
    environment_id: UUID,
    dataset_name: str,
    default_task: str,
    move: bool,
) -> None:
    """Import a LeRobot v3 dataset directly from a local folder (copy or move).

    Unlike the UI's ZIP-upload import flow, this reads the dataset straight off disk,
    so it's suited for large local datasets already staged on the backend host.

    Requires an existing project and environment. Copies files by default;
    --move removes the source directory on success. Unlike hf dataset import,
    this command has no --dry-run, --json, or resumable-transfer interface.

    \b
    Example (replace the UUID placeholders):
      physicalai-studio datasets import-dir --source-dir ./dataset \\
        --project-id PROJECT_UUID --environment-id ENVIRONMENT_UUID --dataset-name "My data"
    """
    from db import get_async_db_session_ctx
    from schemas import Dataset
    from services.dataset_import_dir_service import DatasetImportDirService
    from services.dataset_service import DatasetService

    click.echo(f"Importing dataset from folder: {source_dir}")
    click.echo(f"Mode: {'move' if move else 'copy'}")

    async def _run_import() -> None:
        async def persist_dataset(dataset: Dataset) -> Dataset:
            async with get_async_db_session_ctx() as session:
                return await DatasetService(session).create_dataset(dataset)

        service = DatasetImportDirService(persist_dataset=persist_dataset)
        dataset = await service.import_dataset_directory(
            source_dir=source_dir,
            project_id=project_id,
            environment_id=environment_id,
            dataset_name=dataset_name,
            default_task=default_task,
            move=move,
        )
        click.echo("Dataset imported successfully!")
        click.echo(f"Dataset ID: {dataset.id}")
        click.echo(f"Dataset path: {dataset.path}")

    try:
        asyncio.run(_run_import())
    except Exception as e:
        click.echo(f"Dataset import failed: {e}")
        sys.exit(1)
