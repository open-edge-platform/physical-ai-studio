"""Model management CLI commands."""

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import click

if TYPE_CHECKING:
    from physicalai.policies.base import Policy

    from schemas import Model, TrainJob
    from schemas.dataset import Dataset


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


NOISE_BACKUP_DIRNAME = "noise-input-backup"
NOISE_FIX_SUFFIX = " (noise fix)"


def _get_model_dir(model_id: UUID) -> Path:
    """Return the folder of the Studio model with this id."""
    from db import get_async_db_session_ctx
    from services.model_service import ModelService

    async def get_path() -> str:
        async with get_async_db_session_ctx() as session:
            return (await ModelService(session).get_model_by_id(model_id)).path

    return Path(asyncio.run(get_path())).expanduser()


def _resolve_openvino_export(target: str) -> tuple[Path, Path | None]:
    """Return the OpenVINO export folder TARGET names and, for a Studio model, the model folder."""
    from training.job import EXPORTS_DIRNAME

    path = Path(target).expanduser()
    if not path.exists():
        try:
            model_id = UUID(target)
        except ValueError:
            raise click.ClickException(f"Not a folder or a model UUID: {target}") from None
        try:
            path = _get_model_dir(model_id)
        except Exception as exc:
            raise click.ClickException(f"Model {model_id} not found: {exc}") from exc
    path = path.resolve()
    studio_export = path / EXPORTS_DIRNAME / "openvino"
    if studio_export.is_dir():
        return studio_export, path
    if path.name == "openvino" and path.parent.name == EXPORTS_DIRNAME:
        # A Studio model's export folder: treat it like the model itself.
        return path, path.parent.parent
    return path, None


def _studio_models() -> "list[Model]":
    """Return every Studio model, oldest first."""
    from db import get_async_db_session_ctx
    from services.model_service import ModelService

    async def get_models() -> "list[Model]":
        async with get_async_db_session_ctx() as session:
            return await ModelService(session).get_model_list()

    found = asyncio.run(get_models())
    return sorted(found, key=lambda model: (model.created_at is None, model.created_at))


def _noise_status(model_dir: Path) -> str | None:
    """Describe a Studio model's pi05 OpenVINO export, or return None if it has none."""
    from physicalai.policies.pi05.noise_input import has_backup, read_noise_shape

    from training.job import EXPORTS_DIRNAME

    export_dir = model_dir / EXPORTS_DIRNAME / "openvino"
    # A backup decides restorability on its own: an interrupted fix can leave the
    # export unreadable or looking unfixed.
    restorable = ", restorable" if has_backup(export_dir, model_dir / NOISE_BACKUP_DIRNAME) else ""
    try:
        noise_shape = read_noise_shape(export_dir)
    except (OSError, ValueError):
        return f"unreadable{restorable}" if restorable else None
    return ("needs fix" if noise_shape is None else "fixed") + restorable


def _choose_studio_model(*, restorable_only: bool) -> Path:
    """List the Studio pi05 models with an OpenVINO export and ask which one to use."""
    rows = []
    for model in _studio_models():
        model_dir = Path(model.path).expanduser()
        status = _noise_status(model_dir)
        if status is None or (restorable_only and not status.endswith(", restorable")):
            continue
        created = f", {model.created_at:%Y-%m-%d}" if model.created_at else ""
        rows.append((f"{model.name} (v{model.version}{created})  [{status}]  {model.id}", model_dir))
    if not rows:
        what = "fixed in place with a backup" if restorable_only else "with an OpenVINO export"
        raise click.ClickException(f"No Studio pi05 model {what}.")

    click.echo(
        "Studio pi05 models fixed in place with a backup:"
        if restorable_only
        else "Studio pi05 models with an OpenVINO export:"
    )
    for index, (label, _) in enumerate(rows, start=1):
        click.echo(f"  {index:>2}. {label}")
    choice = click.prompt("Restore model" if restorable_only else "Model to fix", type=click.IntRange(1, len(rows)))
    return rows[choice - 1][1]


async def _get_dataset(dataset_id: UUID) -> "Dataset":
    from db import get_async_db_session_ctx
    from services.dataset_service import DatasetService

    async with get_async_db_session_ctx() as session:
        return await DatasetService(session).get_dataset_by_id(dataset_id)


async def _persist_import(job: "TrainJob", model: "Model") -> "Model":
    """Save the import's job and model in one transaction, so a failure leaves neither."""
    from db import get_async_db_session_ctx
    from repositories.job_repo import JobRepository
    from repositories.model_repo import ModelRepository

    model = model.model_copy(update={"train_job_id": job.id})
    async with get_async_db_session_ctx() as session:
        session.add(JobRepository(session).to_schema(job))
        await session.flush()  # the model row references the job
        session.add(ModelRepository(session).to_schema(model))
        await session.commit()
    return model


def _copy_as_fixed_model(model_dir: Path, export_dir: Path) -> None:
    """Add the fixed export as a new Studio model next to the original, which stays as it is."""
    from physicalai.policies.pi05.noise_input import ConversionStatus, convert_openvino_export

    from services.model_import_service import ModelImportService
    from settings import get_settings

    matches = [model for model in _studio_models() if Path(model.path).expanduser().resolve() == model_dir.resolve()]
    if not matches:
        raise click.ClickException(f"--copy needs a Studio model; {model_dir} is not one")
    original = matches[0]
    if original.dataset_id is None:
        raise click.ClickException(f"Model {original.id} has no dataset; use --out and models import-dir instead")

    models_dir = get_settings().models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise-fix-", dir=models_dir))
    try:
        result = convert_openvino_export(export_dir, staging / "openvino")
        if result.status is ConversionStatus.ALREADY_CONVERTED:
            raise click.ClickException(f"{original.name} already takes the noise as input; nothing to copy")
        service = ModelImportService(get_dataset=_get_dataset, persist_import=_persist_import)
        copy = asyncio.run(
            service.import_model_directory(
                source_dir=staging,
                project_id=original.project_id,
                dataset_id=original.dataset_id,
                model_name=f"{original.name}{NOISE_FIX_SUFFIX}",
                move=True,
                version=original.version,
            )
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    click.echo(f"Added model: {copy.name}  {copy.id}")
    click.echo(f"Original left as it is: {original.name}  {original.id}")


@models.command("fix-pi05-noise")
@click.argument("target", required=False)
@click.option(
    "--copy",
    "as_copy",
    is_flag=True,
    help=f"Studio models: add the fixed version as a new model named '<name>{NOISE_FIX_SUFFIX}' "
    "and leave the original as it is.",
)
@click.option("--restore", is_flag=True, help="Undo an in-place fix from its backup.")
@click.option(
    "--out",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write the fixed export to this new folder instead of fixing in place.",
)
@click.option(
    "--backup/--no-backup",
    default=True,
    show_default=True,
    help=f"In place: keep the replaced files, in the model folder's {NOISE_BACKUP_DIRNAME}/ "
    "for a Studio model, else in EXPORT_DIR.orig/. --restore needs them.",
)
def fix_pi05_noise(target: str | None, as_copy: bool, restore: bool, output_dir: Path | None, backup: bool) -> None:
    """Make an older pi05 OpenVINO export draw fresh start noise on every call.

    pi05 models exported before the start noise became a model input sample it
    inside the graph, where OpenVINO's GPU plugin returns the same noise on every
    call. This rewrites the export so the runtime draws the noise instead. The
    weights stay as they are and no checkpoint is needed. Exports that already take
    the noise as input are left unchanged.

    TARGET is a Studio model UUID, a Studio model folder, or an OpenVINO export
    folder (with manifest.json). Without TARGET, the command lists the Studio
    pi05 models and asks which one to use.

    By default the model is fixed in place and keeps its id and name; reload it in
    a running Studio afterwards. --copy adds the fixed version as a new Studio
    model instead, to compare the two side by side. --restore undoes an in-place
    fix.

    \b
    Examples:
      physicalai-studio models fix-pi05-noise
      physicalai-studio models fix-pi05-noise MODEL_UUID
      physicalai-studio models fix-pi05-noise --copy MODEL_UUID
      physicalai-studio models fix-pi05-noise --restore
    """
    from physicalai.policies.pi05.noise_input import (
        ConversionStatus,
        RollbackError,
        convert_openvino_export,
        restore_openvino_export,
        runtime_draws_noise,
    )

    if as_copy + restore + (output_dir is not None) > 1:
        raise click.UsageError("Use only one of --copy, --restore and --out.")
    if target is None:
        target = str(_choose_studio_model(restorable_only=restore))
    export_dir, model_dir = _resolve_openvino_export(target)
    backup_dir = model_dir / NOISE_BACKUP_DIRNAME if model_dir is not None else None

    try:
        if restore:
            source = restore_openvino_export(export_dir, backup_dir)
            click.echo(f"Restored: {export_dir} (from {source})")
            return
        if as_copy:
            if model_dir is None:
                raise click.ClickException(f"--copy needs a Studio model; {target} is not one")
            _copy_as_fixed_model(model_dir, export_dir)
        else:
            result = convert_openvino_export(export_dir, output_dir, backup=backup, backup_dir=backup_dir)
            if result.status is ConversionStatus.ALREADY_CONVERTED:
                click.echo(f"Already takes the noise as input: {result.export_dir}")
            else:
                click.echo(f"Fixed: {result.export_dir} (noise_shape {list(result.noise_shape)})")
                if result.backup_dir is not None:
                    click.echo(f"Original files kept in: {result.backup_dir} (undo with --restore)")
    except (FileNotFoundError, FileExistsError, ValueError, RollbackError) as exc:
        raise click.ClickException(str(exc)) from exc

    if not runtime_draws_noise():
        click.echo(
            "Warning: the installed physicalai runtime cannot draw the noise yet "
            "(needs openvinotoolkit/physicalai#442). Upgrade physicalai before loading this model.",
            err=True,
        )
