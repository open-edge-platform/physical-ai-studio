import http
from enum import StrEnum
from uuid import UUID


class ResourceType(StrEnum):
    """Enumeration for resource types."""

    PROJECT = "Project"
    DATASET = "Dataset"

class GetiBaseException(Exception):
    """
    Base class for Geti exceptions with a predefined HTTP error code.

    :param message: str message providing short description of error
    :param error_code: str id of error
    :param http_status: int default http status code to return to user
    """

    def __init__(self, message: str, error_code: str, http_status: int) -> None:
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        super().__init__(message)


class ResourceNotFoundError(GetiBaseException):
    """
    Exception raised when a resource could not be found in database.

    :param resource_id: ID of the resource that was not found
    """

    def __init__(self, resource_type: ResourceType, resource_id: str | UUID, message: str | None = None):
        msg = message or f"The requested {resource_type} could not be found. {resource_type.title()} ID: `{resource_id}`."

        super().__init__(
            message=msg,
            error_code=f"{resource_type}_not_found",
            http_status=http.HTTPStatus.NOT_FOUND,
        )

class ResourceInUseError(GetiBaseException):
    """Exception raised when trying to delete a resource that is currently in use."""

    def __init__(self, resource_type: ResourceType, resource_id: str | UUID, message: str | None = None):
        msg = message or f"{resource_type} with ID {resource_id} cannot be deleted because it is in use."
        super().__init__(
            message=msg,
            error_code=f"{resource_type}_not_found",
            http_status=http.HTTPStatus.CONFLICT,
        )

class ResourceAlreadyExistsError(GetiBaseException):
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
