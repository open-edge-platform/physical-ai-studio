import http
from enum import StrEnum
from uuid import UUID


class ResourceType(StrEnum):
    """Enumeration for resource types."""

    PROJECT = "Project"
    ROBOT = "Robot"
    ROBOT_CALIBRATION = "Robot calibration"
    CAMERA = "Camera"
    ENVIRONMENT = "Environment"
    DATASET = "Dataset"
    MODEL = "Model"
    JOB = "JOB"
    JOB_FILE = "JOB_FILE"


class BaseException(Exception):
    """
    Base class for PhysicalAI exceptions with a predefined HTTP error code.

    :param message: str message providing short description of error
    :param error_code: str id of error
    :param http_status: int default http status code to return to user
    """

    def __init__(self, message: str, error_code: str, http_status: int) -> None:
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        super().__init__(message)


class ResourceNotFoundError(BaseException):
    """
    Exception raised when a resource could not be found in database.

    :param resource_id: ID of the resource that was not found
    """

    def __init__(self, resource_type: ResourceType, resource_id: str | UUID, message: str | None = None):
        msg = (
            message or f"The requested {resource_type} could not be found. {resource_type.title()} ID: `{resource_id}`."
        )

        super().__init__(
            message=msg,
            error_code=f"{resource_type}_not_found",
            http_status=http.HTTPStatus.NOT_FOUND,
        )


class DuplicateJobException(BaseException):
    """
    Exception raised when attempting to submit a duplicate job.

    :param message: str containing a custom message about the duplicate job.
    """

    def __init__(self, message: str = "A job with the same payload is already running or queued") -> None:
        super().__init__(message=message, error_code="duplicate_job", http_status=http.HTTPStatus.CONFLICT)


class ResourceInUseError(BaseException):
    """Exception raised when trying to delete a resource that is currently in use."""

    def __init__(self, resource_type: ResourceType, resource_id: str | UUID, message: str | None = None):
        msg = message or f"{resource_type} with ID {resource_id} cannot be deleted because it is in use."
        super().__init__(
            message=msg,
            error_code=f"{resource_type}_not_found",
            http_status=http.HTTPStatus.CONFLICT,
        )


class ResourceAlreadyExistsError(BaseException):
    """
    Exception raised when a resource already exists.

    :param resource_name: Name of the resource that was not found
    """

    def __init__(self, resource_name: str, detail: str) -> None:
        super().__init__(
            message=f"{resource_name} already exists. {detail}",
            error_code=f"{resource_name}_already_exists",
            http_status=http.HTTPStatus.CONFLICT,
        )


class UnsupportedDeviceError(BaseException):
    """Exception raised when a requested training device is not available on the system."""

    def __init__(self, device_type: str, supported: list[str]) -> None:
        supported_str = ", ".join(supported) if supported else "none"
        super().__init__(
            message=f"Device type '{device_type}' is not available for training. Supported devices: {supported_str}.",
            error_code="unsupported_device",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class InvalidJobStateError(BaseException):
    """Raised when a job action is not valid in the current state."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="invalid_job_state",
            http_status=http.HTTPStatus.CONFLICT,
        )


class DuplicateImportSourceError(BaseException):
    """Raised when importing an already imported source UUID."""

    def __init__(self, resource_kind: str, source_uuid: str) -> None:
        super().__init__(
            message=f"{resource_kind} with original source UUID `{source_uuid}` was already imported.",
            error_code="duplicate_import_source",
            http_status=http.HTTPStatus.CONFLICT,
        )


class ZipBombDetectedError(BaseException):
    """Raised when an uploaded archive is considered unsafe."""

    def __init__(self, message: str = "Uploaded archive was rejected by zip safety validation") -> None:
        super().__init__(
            message=message,
            error_code="zip_bomb_detected",
            http_status=http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )


class InvalidArchiveError(BaseException):
    """Raised when an uploaded archive is invalid or unreadable."""

    def __init__(self, message: str = "Uploaded archive is invalid or unreadable") -> None:
        super().__init__(
            message=message,
            error_code="invalid_archive",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class UploadTooLargeError(BaseException):
    """Raised when the HTTP upload exceeds the configured maximum size."""

    def __init__(self, message: str = "Uploaded file exceeds the maximum allowed size") -> None:
        super().__init__(
            message=message,
            error_code="upload_too_large",
            http_status=http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )


class InvalidResourceError(BaseException):
    """
    Exception raised when a resource is not what was expected.

    :param resource_name: Name of the resource that was not found
    """

    def __init__(self, resource_name: str, detail: str) -> None:
        super().__init__(
            message=f"{resource_name} invalid resource. {detail}",
            error_code=f"{resource_name}_invalid_resource",
            http_status=http.HTTPStatus.CONFLICT,
        )


class InsufficientDiskSpaceError(BaseException):
    """Raised when there is not enough free disk space to safely store the upload or extraction."""

    def __init__(self, message: str = "Insufficient disk space to process the upload") -> None:
        super().__init__(
            message=message,
            error_code="insufficient_disk_space",
            http_status=http.HTTPStatus.INSUFFICIENT_STORAGE,
        )


class RecordingLockError(BaseException):
    """Raised when a camera cannot be modified because it is locked by an active recording session."""

    def __init__(self, message: str = "Camera is in use by an active recording session.") -> None:
        super().__init__(
            message=message,
            error_code="recording_locked",
            http_status=423,
        )
