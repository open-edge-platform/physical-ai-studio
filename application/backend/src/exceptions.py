import http
from uuid import UUID


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


class ResourceNotFoundException(GetiBaseException):
    """
    Exception raised when a resource could not be found in database.

    :param resource_id: ID of the resource that was not found
    """

    def __init__(self, resource_id: str | UUID, resource_name: str) -> None:
        super().__init__(
            message=f"The requested {resource_name} could not be found. {resource_name.title()} ID: `{resource_id}`.",
            error_code=f"{resource_name}_not_found",
            http_status=http.HTTPStatus.NOT_FOUND,
        )
