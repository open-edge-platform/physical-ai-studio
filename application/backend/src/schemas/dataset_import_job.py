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

    type: str | None = None
    joints: list[str] = Field(default_factory=list)


class DatasetManifestRecordingSchema(BaseModel):
    """Cameras and robots inferred from dataset format metadata."""

    cameras: list[ManifestCameraEntry] = Field(default_factory=list)
    robots: list[ManifestRobotEntry] = Field(default_factory=list)


class ImportStep(StrEnum):
    # User created import job and archive upload is expected next.
    AWAITING_ARCHIVE_UPLOAD = "awaiting_archive_upload"
    # Archive is uploaded and queued for worker-side format detection.
    QUEUED_FOR_DETECTION = "queued_for_detection"
    # Worker is determining dataset format adapter.
    DETECTING_FORMAT = "detecting_format"
    # Worker is building draft manifest and validation report.
    BUILDING_MANIFEST_DRAFT = "building_manifest_draft"
    # Draft is ready and UI should present finalization form.
    AWAITING_USER_REVIEW = "awaiting_user_review"
    # User finalized input and import is queued.
    QUEUED_FOR_IMPORT = "queued_for_import"
    # Worker is importing/extracting/persisting dataset.
    IMPORTING_DATASET = "importing_dataset"
    # Import complete
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
    statistics: DatasetManifestStatistics = Field(default_factory=DatasetManifestStatistics)
    dataset_schema: DatasetManifestRecordingSchema = Field(default_factory=DatasetManifestRecordingSchema)


class DatasetImportFinalizeInput(BaseModel):
    environment_id: UUID
    default_task: str = ""

    @field_serializer("environment_id")
    def serialize_environment_id(self, environment_id: UUID, _info: SerializationInfo) -> str:
        return str(environment_id)


class DatasetImportJobPayload(BaseModel):
    step: ImportStep = ImportStep.AWAITING_ARCHIVE_UPLOAD
    result_dataset_id: UUID | None = None

    # Opaque staging identifier - resolve the archive path via resolve_payload_archive_path().
    archive_staging_id: UUID
    uploaded_archive_name: str | None = None
    format_hint: str = "auto"
    # User-provided dataset name, captured at prepare time.
    dataset_name: str | None = None
    dataset_manifest_draft: DatasetManifest | None = None
    validation_report: ImportValidationReport | None = None
    finalize_input: DatasetImportFinalizeInput | None = None

    @field_serializer("result_dataset_id")
    def serialize_result_dataset_id(self, result_dataset_id: UUID | None, _info: SerializationInfo) -> str | None:
        return str(result_dataset_id) if result_dataset_id else None

    @field_serializer("archive_staging_id")
    def serialize_archive_staging_id(self, archive_staging_id: UUID, _info: SerializationInfo) -> str:
        return str(archive_staging_id)
