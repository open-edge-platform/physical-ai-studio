import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from exceptions import InvalidArchiveError
from schemas import Model, TrainJob
from schemas.base_job import JobStatus
from schemas.dataset import Dataset
from schemas.job import TrainJobPayload
from services.dataset_service import DatasetService
from services.job_service import JobService
from services.model_service import ModelService
from settings import get_settings

# We assume the directory/zip is taken directly from Physical AI Studio, either
# by exporting the model from the UI, or by taking it from our storage dir
_REQUIRED_FILES = (
    "version_0/hparams.yaml",
    "version_0/metrics.csv",
    "exports/torch/manifest.json",
)
_TORCH_MANIFEST_PATH = "exports/torch/manifest.json"
_SUPPORTED_POLICIES = frozenset({"act", "smolvla", "pi05"})


class ModelReader(Protocol):
    """Abstract reader for model files (ZIP archive or directory)."""

    def file_exists(self, path: str) -> bool:
        """Check if a file exists at the given relative path."""
        ...

    def read_json(self, path: str) -> dict[str, Any] | None:
        """Read and parse a JSON file. Returns None if not found or invalid."""
        ...


class DirectoryModelReader:
    """ModelReader implementation backed by a filesystem directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def file_exists(self, path: str) -> bool:
        return (self._root / path).is_file()

    def read_json(self, path: str) -> dict[str, Any] | None:
        file_path = self._root / path
        if not file_path.is_file():
            return None
        try:
            with file_path.open(encoding="utf-8") as fobj:
                data = json.load(fobj)
        except (OSError, ValueError):
            return None
        if isinstance(data, dict):
            return data
        return None


class ModelImportService:
    async def import_model_directory(
        self,
        *,
        source_dir: Path,
        project_id: UUID,
        dataset_id: UUID,
        model_name: str,
        move: bool = False,
        base_model_id: UUID | None = None,
        version: int = 1,
    ) -> Model:
        """Import a model from a directory (copy or move)."""
        if not source_dir.exists() or not source_dir.is_dir():
            raise InvalidArchiveError(f"Model directory does not exist: {source_dir}")

        settings = get_settings()
        dataset = await DatasetService.get_dataset_by_id(dataset_id)
        if dataset.project_id != project_id:
            raise InvalidArchiveError("Dataset does not belong to the specified project")

        model_dir = settings.models_dir / str(uuid4())

        reader = DirectoryModelReader(source_dir)
        policy = self._inspect_model(reader)

        try:
            if move:
                await asyncio.to_thread(shutil.move, str(source_dir), str(model_dir))
            else:
                await asyncio.to_thread(shutil.copytree, source_dir, model_dir)

            return await self._finalize_import(
                model_dir=model_dir,
                dataset=dataset,
                model_name=model_name,
                policy=policy,
                base_model_id=base_model_id,
                version=version,
            )
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

    async def _finalize_import(
        self,
        *,
        model_dir: Path,
        dataset: Dataset,
        model_name: str,
        policy: str,
        base_model_id: UUID | None,
        version: int,
    ) -> Model:
        """Create job, and model record after files are in place."""
        project_id = dataset.project_id
        dataset_id = dataset.id

        job = TrainJob(
            project_id=project_id,
            payload=TrainJobPayload(
                project_id=project_id,
                dataset_id=dataset_id,
                policy=policy,
                model_name=model_name,
                max_steps=100,
                batch_size=1,
                auto_scale_batch_size=False,
                base_model_id=base_model_id,
                val_split=0.1,
                device=None,
            ),
            status=JobStatus.COMPLETED,
            message="Model import completed",
        )
        job = await JobService.create_job(job)

        model = Model(
            id=UUID(model_dir.name),
            project_id=project_id,
            dataset_id=dataset_id,
            path=str(model_dir),
            name=model_name,
            # Imported models don't have a snapshot: the provided dataset may differ
            # from what was actually used for training (possibly on another machine).
            snapshot_id=None,
            policy=policy,
            properties={},
            train_job_id=job.id,
            parent_model_id=base_model_id,
            version=version,
            created_at=None,
        )
        return await ModelService.create_model(model)

    def _inspect_model(self, reader: ModelReader) -> str:
        """Validate model structure and infer policy."""
        for required in _REQUIRED_FILES:
            if not reader.file_exists(required):
                raise InvalidArchiveError(f"Model is missing required file '{required}'")

        torch_manifest = self._read_manifest(reader, _TORCH_MANIFEST_PATH)
        self._validate_torch_artifact(torch_manifest, reader)
        return self._infer_policy(torch_manifest, _TORCH_MANIFEST_PATH)

    def _read_manifest(self, reader: ModelReader, path: str) -> dict[str, Any]:
        """Read and validate a manifest JSON file."""
        data = reader.read_json(path)
        if data is None:
            raise InvalidArchiveError(f"Model is missing required file '{path}'")
        if data.get("format") != "policy_package":
            raise InvalidArchiveError(f"Manifest '{path}' must declare format='policy_package'")
        return data

    def _validate_torch_artifact(self, torch_manifest: dict[str, Any], reader: ModelReader) -> None:
        """Validate that the torch artifact referenced in the manifest exists."""
        torch_artifact = self._extract_torch_artifact_path(torch_manifest, _TORCH_MANIFEST_PATH)
        artifact_path = f"exports/torch/{torch_artifact}"
        if not reader.file_exists(artifact_path):
            raise InvalidArchiveError(
                f"Manifest '{_TORCH_MANIFEST_PATH}' references missing torch artifact '{torch_artifact}'"
            )

    @staticmethod
    def _extract_torch_artifact_path(torch_manifest: dict[str, Any], label: str) -> str:
        """Extract and validate the torch artifact path from the manifest."""
        model_section = torch_manifest.get("model")
        if not isinstance(model_section, dict):
            raise InvalidArchiveError(f"Manifest '{label}' is missing object field 'model'")

        artifacts = model_section.get("artifacts")
        if not isinstance(artifacts, dict):
            raise InvalidArchiveError(f"Manifest '{label}' is missing object field 'model.artifacts'")

        torch_artifact = artifacts.get("torch")
        if not isinstance(torch_artifact, str) or not torch_artifact.strip():
            raise InvalidArchiveError(f"Manifest '{label}' is missing non-empty 'model.artifacts.torch' entry")

        artifact_path = Path(torch_artifact)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise InvalidArchiveError(f"Manifest '{label}' contains unsafe torch artifact path '{torch_artifact}'")

        return torch_artifact

    def _infer_policy(self, manifest: dict[str, Any], manifest_path: str) -> str:
        """Extract and validate the policy name from the manifest."""
        policy_section = manifest.get("policy")
        if not isinstance(policy_section, dict):
            raise InvalidArchiveError(f"Manifest '{manifest_path}' is missing 'policy' section")

        policy_name = policy_section.get("name")
        if not isinstance(policy_name, str) or not policy_name:
            raise InvalidArchiveError(f"Manifest '{manifest_path}' is missing 'policy.name'")

        if policy_name not in _SUPPORTED_POLICIES:
            raise InvalidArchiveError(
                f"Manifest '{manifest_path}' declares unsupported policy '{policy_name}'. "
                f"Supported policies are: {', '.join(sorted(_SUPPORTED_POLICIES))}"
            )

        return policy_name
