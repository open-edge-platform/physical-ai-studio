import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, computed_field

from schemas.base import BaseIDModel, Field


class Model(BaseIDModel):
    name: str
    path: str
    policy: str
    properties: dict
    project_id: Annotated[UUID, Field(description="Project Unique identifier")]
    dataset_id: Annotated[UUID | None, Field(None, description="Dataset Unique identifier")]
    snapshot_id: Annotated[UUID | None, Field(None, description="Snapshot Unique identifier")]
    train_job_id: UUID | None = Field(None, description="ID of the training job that created this model")
    parent_model_id: UUID | None = Field(None, description="Parent model this was retrained from")
    version: int = Field(1, description="Model version, incremented on each retrain")
    created_at: datetime | None = Field(None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def available_backends(self) -> list[str]:
        exports_dir = Path(self.path) / "exports"
        if not exports_dir.is_dir():
            return []

        backends: list[str] = []
        for backend_dir in exports_dir.iterdir():
            if not backend_dir.is_dir():
                continue

            # Backend exports folder may be empty if export failed
            if not any(backend_dir.iterdir()):
                continue

            backends.append(backend_dir.name)

        return sorted(backends)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "",
                "name": "Dataset X/Y ACT Model",
                "path": "Path/to/model/ckpt",
                "properties": {},
                "policy": "act",
                "dataset_id": "",
                "project_id": "",
                "snapshot_id": "",
                "train_job_id": "0db0c16d-0d3c-4e0e-bc5a-ca710579e549",
                "parent_model_id": None,
                "version": 1,
                "created_at": "2021-06-29T16:24:30.928000+00:00",
            }
        }
    )


class IOFeature(BaseModel):
    name: str
    ftype: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None

    @classmethod
    def from_raw_list(cls, raw_features: Any) -> list["IOFeature"]:
        if not isinstance(raw_features, list):
            return []

        features: list[IOFeature] = []
        for raw_feature in raw_features:
            if not isinstance(raw_feature, dict):
                continue

            init_args = raw_feature.get("init_args")
            if not isinstance(init_args, dict):
                init_args = raw_feature

            name = init_args.get("name")
            if not isinstance(name, str) or not name:
                continue

            raw_shape = init_args.get("shape")
            shape = raw_shape if isinstance(raw_shape, list) and all(isinstance(v, int) for v in raw_shape) else None

            ftype = init_args.get("ftype")
            dtype = init_args.get("dtype")
            features.append(
                cls(
                    name=name,
                    ftype=ftype if isinstance(ftype, str) else None,
                    shape=shape,
                    dtype=dtype if isinstance(dtype, str) else None,
                )
            )

        return features


class BackendIOSpec(BaseModel):
    input_features: list[IOFeature]
    output_features: list[IOFeature]

    @classmethod
    def from_manifest(cls, manifest: Any) -> "BackendIOSpec | None":
        if not isinstance(manifest, dict):
            return None

        model_section = manifest.get("model")
        if not isinstance(model_section, dict):
            return None

        input_features = IOFeature.from_raw_list(model_section.get("input_features"))
        output_features = IOFeature.from_raw_list(model_section.get("output_features"))

        if not input_features and not output_features:
            return None

        return cls(
            input_features=input_features,
            output_features=output_features,
        )

    @classmethod
    def from_backend_dir(cls, backend_dir: Path) -> "BackendIOSpec | None":
        manifest_path = backend_dir / "manifest.json"
        if not manifest_path.is_file():
            return None

        try:
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            return None

        return cls.from_manifest(manifest)


class BackendExportDetail(BaseModel):
    type: str
    size_bytes: int
    file_count: int
    exported_at: datetime | None = None
    io_spec: BackendIOSpec | None = None

    @classmethod
    def from_backend_dir(cls, backend_dir: Path) -> "BackendExportDetail | None":
        if not backend_dir.is_dir():
            return None

        files = [f for f in backend_dir.rglob("*") if f.is_file()]

        if len(files) == 0:
            return None

        total_size = sum(f.stat().st_size for f in files)
        exported_at = datetime.fromtimestamp(backend_dir.stat().st_mtime)

        return cls(
            type=backend_dir.name,
            size_bytes=total_size,
            file_count=len(files),
            exported_at=exported_at,
            io_spec=BackendIOSpec.from_backend_dir(backend_dir),
        )


class TrainingSummary(BaseModel):
    max_steps: int | None = None
    batch_size: int | None = None
    auto_scale_batch_size: bool | None = None
    num_workers: int | str | None = None
    precision: str | None = None
    compile_model: bool | None = None
    val_split: float | None = None
    device_type: str | None = None


class ModelDetailResponse(BaseModel):
    model: Model
    exports: list[BackendExportDetail]
    training_summary: TrainingSummary | None = None
    hparams: dict | None = None
