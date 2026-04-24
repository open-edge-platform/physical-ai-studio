from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer
from pydantic_core.core_schema import SerializationInfo


class ManifestCameraEntry(BaseModel):
    """Recording schema entry describing a single camera stream."""

    name: str
    width: int | None = None
    height: int | None = None
    fps: int | None = None


class ManifestRobotEntry(BaseModel):
    """Recording schema entry describing a robot and its controllable joints."""

    name: str
    type: str | None = None
    joints: list[str] = Field(default_factory=list)


class DatasetManifestRecordingSchema(BaseModel):
    """Cameras and robots inferred from dataset source metadata."""

    cameras: list[ManifestCameraEntry] = Field(default_factory=list)
    robots: list[ManifestRobotEntry] = Field(default_factory=list)


class ImportStep(StrEnum):
    AWAITING_UPLOAD = "awaiting_upload"
    UPLOADED = "uploaded"
    DETECTING_SOURCE = "detecting_source"
    GENERATING_DRAFT_MANIFEST = "generating_draft_manifest"
    WAITING_FOR_USER_INPUT = "waiting_for_user_input"
    READY_TO_COMMIT = "ready_to_commit"
    REGISTERING_RESOURCE = "registering_resource"
    IMPORTING_RESOURCE = "importing_resource"
    COMPLETED = "completed"


class DatasetImportSource(StrEnum):
    LEROBOT_V3 = "lerobot_v3"
    UNKNOWN = "unknown"


class ImportValidationSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class ImportValidationMessage(BaseModel):
    severity: ImportValidationSeverity
    message: str


class ImportValidationReport(BaseModel):
    messages: list[ImportValidationMessage] = Field(default_factory=list)

    def add(self, severity: ImportValidationSeverity, message: str) -> None:
        self.messages.append(ImportValidationMessage(severity=severity, message=message))

    def add_error(self, message: str) -> None:
        self.messages.append(ImportValidationMessage(severity=ImportValidationSeverity.ERROR, message=message))

    def add_warning(self, message: str) -> None:
        self.messages.append(ImportValidationMessage(severity=ImportValidationSeverity.WARNING, message=message))

    @computed_field(return_type=bool)
    def is_valid(self) -> bool:
        return not any(message.severity == ImportValidationSeverity.ERROR for message in self.messages)


class DatasetManifestStatistics(BaseModel):
    episode_count: int = Field(default=0, ge=0)
    frame_count: int = Field(default=0, ge=0)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_type: DatasetImportSource = DatasetImportSource.UNKNOWN
    suggested_name: str | None = None
    statistics: DatasetManifestStatistics = Field(default_factory=DatasetManifestStatistics)
    dataset_schema: DatasetManifestRecordingSchema = Field(default_factory=DatasetManifestRecordingSchema)


class DatasetImportFinalizeInput(BaseModel):
    dataset_name: str
    environment_id: UUID
    default_task: str = ""

    @field_serializer("environment_id")
    def serialize_environment_id(self, environment_id: UUID, _info: SerializationInfo) -> str:
        return str(environment_id)


class DatasetImportJobPayload(BaseModel):
    step: ImportStep = ImportStep.AWAITING_UPLOAD
    result_dataset_id: UUID | None = None

    # Opaque staging identifier - resolve the archive path via resolve_payload_archive_path().
    archive_staging_id: UUID
    uploaded_archive_name: str | None = None
    source_hint: str = "auto"
    dataset_manifest_draft: DatasetManifest | None = None
    validation_report: ImportValidationReport | None = None
    finalize_input: DatasetImportFinalizeInput | None = None

    @field_serializer("result_dataset_id")
    def serialize_result_dataset_id(self, result_dataset_id: UUID | None, _info: SerializationInfo) -> str | None:
        return str(result_dataset_id) if result_dataset_id else None

    @field_serializer("archive_staging_id")
    def serialize_archive_staging_id(self, archive_staging_id: UUID, _info: SerializationInfo) -> str:
        return str(archive_staging_id)
