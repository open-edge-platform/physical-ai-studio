"""Model management CLI commands."""

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import click

if TYPE_CHECKING:
    from physicalai.policies.base import Policy


@click.group()
def models() -> None:
    """Import Studio models from local folders.

    For Hugging Face repositories, use hf model instead. These commands register
    models in the configured local Studio database, not a remote Studio server.
    """


@models.command("import-dir")
@click.option(
    "--source-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Existing Studio model directory or exports-only directory.",
)
@click.option("--project-id", required=True, type=click.UUID, help="Existing target Studio project UUID.")
@click.option("--dataset-id", required=True, type=click.UUID, help="Local dataset UUID in the target project.")
@click.option("--model-name", required=True, type=str, help="Display name for the imported model.")
@click.option("--move/--copy", default=False, show_default=True, help="Move removes the source; copy keeps it intact.")
@click.option("--base-model-id", type=click.UUID, default=None, help="Optional local parent model UUID for lineage.")
@click.option("--version", type=int, default=1, show_default=True, help="Version assigned to the imported model.")
def import_dir(
    source_dir: Path,
    project_id: UUID,
    dataset_id: UUID,
    model_name: str,
    move: bool,
    base_model_id: UUID | None,
    version: int,
) -> None:
    """Import a model from an existing folder (copy or move) trained by Physical AI Studio.

    The dataset_id should reference a dataset that uses the same environment as the
    original training. The inference UI uses this dataset to determine which environment
    to load when running the model.

    Copies files by default; --move removes the source directory on success.
    This command has no --dry-run, --json, or resumable-transfer interface.

    \b
    Example (replace the UUID placeholders):
      physicalai-studio models import-dir --source-dir ./model \\
        --project-id PROJECT_UUID --dataset-id DATASET_UUID --model-name "My model"
    """
    from db import get_async_db_session_ctx
    from schemas import Model, TrainJob
    from schemas.dataset import Dataset
    from services.dataset_service import DatasetService
    from services.job_service import JobService
    from services.model_import_service import ModelImportService
    from services.model_service import ModelService

    click.echo(f"Importing model from folder: {source_dir}")
    click.echo(f"Mode: {'move' if move else 'copy'}")

    async def _run_import() -> None:
        async def get_dataset(dataset_id: UUID) -> Dataset:
            async with get_async_db_session_ctx() as session:
                return await DatasetService(session).get_dataset_by_id(dataset_id)

        async def persist_import(job: TrainJob, model: Model) -> Model:
            async with get_async_db_session_ctx() as session:
                saved_job = await JobService(session).create_job(job)
                return await ModelService(session).create_model(model.model_copy(update={"train_job_id": saved_job.id}))

        service = ModelImportService(get_dataset=get_dataset, persist_import=persist_import)
        model = await service.import_model_directory(
            source_dir=source_dir,
            project_id=project_id,
            dataset_id=dataset_id,
            model_name=model_name,
            move=move,
            base_model_id=base_model_id,
            version=version,
        )
        click.echo("Model imported successfully!")
        click.echo(f"Model ID: {model.id}")
        click.echo(f"Model path: {model.path}")

    try:
        asyncio.run(_run_import())
    except Exception as e:
        click.echo(f"Model import failed: {e}")
        sys.exit(1)


def _export_requested_backends(policy: "Policy", model_dir: Path, requested: list[str]) -> None:
    """Export every requested backend, failing the CLI if any export is unavailable or fails."""
    from physicalai.export import ExportablePolicyMixin

    from training.job import EXPORTS_DIRNAME, _backend_name, _release_memory

    if not isinstance(policy, ExportablePolicyMixin):
        raise click.ClickException(f"Policy does not support export backends: {', '.join(requested)}")

    supported = {_backend_name(backend): backend for backend in policy.get_supported_export_backends()}
    failures: dict[str, str] = {}
    for name in dict.fromkeys(requested):
        if name not in supported:
            failures[name] = "unsupported by this policy"
            click.echo(f"Export to {name} failed: {failures[name]}", err=True)
            continue

        try:
            click.echo(f"Exporting to {name} format")
            policy.export(model_dir / EXPORTS_DIRNAME / name, backend=supported[name])
        except ImportError as exc:
            failures[name] = f"optional dependency missing ({exc})"
        except Exception as exc:
            failures[name] = str(exc)
        else:
            click.echo(f"Export to {name} succeeded")
        finally:
            # Keep conversion memory cleanup without training's best-effort error handling.
            _release_memory()

        if name in failures:
            click.echo(f"Export to {name} failed: {failures[name]}", err=True)

    if failures:
        details = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
        raise click.ClickException(f"Requested exports failed: {details}")


@models.command("export")
@click.argument("model_id", type=click.UUID)
@click.option(
    "--backend",
    "backends",
    multiple=True,
    type=click.Choice(("torch", "openvino", "onnx", "executorch"), case_sensitive=False),
    default=("torch", "openvino"),
    show_default=True,
    help="Export backend to produce. Specify the option more than once to select multiple backends.",
)
def export(model_id: UUID, backends: tuple[str, ...]) -> None:
    """Export a persisted model checkpoint to one or more deployment backends.

    By default this produces PyTorch and OpenVINO artifacts under the model's
    ``exports/`` directory. Existing artifacts are handled by the policy export
    implementation in the same way as exports performed after training.
    Unlike training, this command exits with an error if any requested backend
    is unsupported, unavailable, or fails to export.
    """
    from db import get_async_db_session_ctx
    from schemas import Model
    from services.model_service import ModelService
    from training.job import TrainingJobSpec, _load_policy_from_checkpoint, resolve_checkpoint

    async def get_model() -> Model:
        async with get_async_db_session_ctx() as session:
            return await ModelService(session).get_model_by_id(model_id)

    try:
        model = asyncio.run(get_model())
        model_dir = Path(model.path)
        checkpoint = resolve_checkpoint(model_dir)
        if not checkpoint.is_file():
            raise click.ClickException(f"Model checkpoint not found: {checkpoint}")

        requested = [backend.lower() for backend in backends]
        click.echo(f"Loading model {model.id} from {checkpoint}")
        policy = _load_policy_from_checkpoint(TrainingJobSpec(policy=model.policy), checkpoint)
        _export_requested_backends(policy, model_dir, requested)
        click.echo(f"Export finished for model {model.id}")
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Model export failed: {exc}") from exc
