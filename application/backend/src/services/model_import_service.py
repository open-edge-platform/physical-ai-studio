import asyncio
import json
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from exceptions import InvalidArchiveError
from schemas import Model, TrainJob
from schemas.base_job import JobStatus
from schemas.dataset import Dataset
from schemas.job import LocalTrainJobPayload
from settings import get_settings

# Imported models must retain the checkpoint needed to resume training. Logger
# artifacts are optional and deployment exports may target any supported backend.
_REQUIRED_FILES = ("model.ckpt",)
_EXPORT_BACKENDS = ("torch", "openvino", "onnx", "executorch")
_SUPPORTED_POLICIES = frozenset({"act", "smolvla", "pi05", "rldx1", "molmoact2", "xr0"})


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
    def __init__(
        self,
        get_dataset: Callable[[UUID], Awaitable[Dataset]],
        persist_import: Callable[[TrainJob, Model], Awaitable[Model]],
    ) -> None:
        self._get_dataset = get_dataset
        self._persist_import = persist_import

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
        """Import a full model directory or a deployment ``exports`` directory."""
        if not source_dir.exists() or not source_dir.is_dir():
            raise InvalidArchiveError(f"Model directory does not exist: {source_dir}")

        settings = get_settings()
        dataset = await self._get_dataset(dataset_id)
        if dataset.project_id != project_id:
            raise InvalidArchiveError("Dataset does not belong to the specified project")

        model_dir = settings.models_dir / str(uuid4())

        reader = DirectoryModelReader(source_dir)
        source_is_exports_dir = self._is_exports_directory(reader)
        policy = self._inspect_model(reader, source_is_exports_dir=source_is_exports_dir)

        try:
            destination = model_dir / "exports" if source_is_exports_dir else model_dir
            if move:
                destination.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(shutil.move, str(source_dir), str(destination))
            else:
                await asyncio.to_thread(shutil.copytree, source_dir, destination)

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
            payload=LocalTrainJobPayload(
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
        return await self._persist_import(job, model)

    def _is_exports_directory(self, reader: ModelReader) -> bool:
        """Return whether the reader is rooted at an ``exports`` directory."""
        return any(reader.file_exists(f"{backend}/manifest.json") for backend in _EXPORT_BACKENDS)

    def _inspect_model(self, reader: ModelReader, *, source_is_exports_dir: bool) -> str:
        """Validate model structure and infer policy."""
        if not source_is_exports_dir:
            for required in _REQUIRED_FILES:
                if not reader.file_exists(required):
                    raise InvalidArchiveError(f"Model is missing required file '{required}'")

        for backend in _EXPORT_BACKENDS:
            manifest_path = f"{backend}/manifest.json" if source_is_exports_dir else f"exports/{backend}/manifest.json"
            if not reader.file_exists(manifest_path):
                continue
            manifest = self._read_manifest(reader, manifest_path)
            self._validate_backend_artifact(
                manifest,
                reader,
                backend,
                manifest_path,
                source_is_exports_dir=source_is_exports_dir,
            )
            return self._infer_policy(manifest, manifest_path)

        raise InvalidArchiveError("Model is missing a supported export manifest under 'exports/'")

    def _read_manifest(self, reader: ModelReader, path: str) -> dict[str, Any]:
        """Read and validate a manifest JSON file."""
        data = reader.read_json(path)
        if data is None:
            raise InvalidArchiveError(f"Model is missing required file '{path}'")
        if data.get("format") != "policy_package":
            raise InvalidArchiveError(f"Manifest '{path}' must declare format='policy_package'")
        return data

    def _validate_backend_artifact(
        self,
        manifest: dict[str, Any],
        reader: ModelReader,
        backend: str,
        manifest_path: str,
        *,
        source_is_exports_dir: bool,
    ) -> None:
        """Validate that a manifest references an existing backend artifact."""
        artifact = self._extract_backend_artifact_path(manifest, backend, manifest_path)
        artifact_path = f"{backend}/{artifact}" if source_is_exports_dir else f"exports/{backend}/{artifact}"
        if not reader.file_exists(artifact_path):
            raise InvalidArchiveError(f"Manifest '{manifest_path}' references missing {backend} artifact '{artifact}'")

    @staticmethod
    def _extract_backend_artifact_path(manifest: dict[str, Any], backend: str, label: str) -> str:
        """Extract and validate a backend artifact path from a manifest."""
        model_section = manifest.get("model")
        if not isinstance(model_section, dict):
            raise InvalidArchiveError(f"Manifest '{label}' is missing object field 'model'")

        artifacts = model_section.get("artifacts")
        if not isinstance(artifacts, dict):
            raise InvalidArchiveError(f"Manifest '{label}' is missing object field 'model.artifacts'")

        artifact = artifacts.get(backend)
        if not isinstance(artifact, str) or not artifact.strip():
            raise InvalidArchiveError(f"Manifest '{label}' is missing non-empty 'model.artifacts.{backend}' entry")

        artifact_path = Path(artifact)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise InvalidArchiveError(f"Manifest '{label}' contains unsafe {backend} artifact path '{artifact}'")

        return artifact

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
