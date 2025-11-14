import http
from collections import defaultdict
from collections.abc import Sequence

import pydantic
from fastapi import FastAPI, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from exceptions import GetiBaseException


def handle_base_exception(request: Request, exception: Exception) -> Response:
    """
    Base exception handler
    """
    if not isinstance(exception, GetiBaseException):
        raise exception

    response = jsonable_encoder(
        {"error_code": exception.error_code, "message": exception.message, "http_status": exception.http_status}
    )
    headers: dict[str, str] | None = None
    # 204 skipped as No Content needs to be revalidated
    if exception.http_status not in [200, 201, 202, 203, 205, 206, 207, 208, 226] and request.method == "GET":
        headers = {"Cache-Control": "no-cache"}  # always revalidate
    if exception.http_status in [204, 304] or exception.http_status < 200:
        return Response(status_code=int(exception.http_status), headers=headers)
    return JSONResponse(content=response, status_code=int(exception.http_status), headers=headers)


async def handle_error(_request: Request, exception: Exception) -> JSONResponse:
    """
    Handler for internal server errors
    """
    logger.exception(f"Internal server error: {exception}")
    headers = {"Cache-Control": "no-cache"}  # always revalidate
    return JSONResponse(
        {"internal_server_error": "An internal server error occurred."},
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers=headers,
    )


async def handle_not_found(request: Request, _exception: Exception) -> JSONResponse:
    """Handler for 'endpoint not found' errors"""
    message = f"Endpoint {request.url} is not found"
    logger.info(message)
    headers = {"Cache-Control": "no-cache"}  # always revalidate
    return JSONResponse(
        {"endpoint_not_found_response": message},
        status_code=status.HTTP_404_NOT_FOUND,
        headers=headers,
    )


async def validation_exception_handler(_request: Request, exception: Exception) -> JSONResponse:
    """
    Converts a RequestValidationError to a better readable Bad request exception.
    """
    if not isinstance(exception, RequestValidationError):
        raise exception

    reformatted_message = defaultdict(list)
    for pydantic_error in exception.errors():
        # `loc` usually is a list with 2 items describing the location of the error.
        # The first item specifies if the error is a body, query or path parameter and
        # the second is the parameter name. Here, only the parameter name is used along
        # with a message explaining what the problem with the parameter is.
        loc, msg = pydantic_error["loc"], pydantic_error["msg"]
        filtered_loc = loc[1:] if loc[0] in ("body", "query", "path") else loc
        field_string = ".".join(str(filtered_loc))  # nested fields with dot-notation
        reformatted_message[field_string].append(msg)

    headers = {"Cache-Control": "no-cache"}  # always revalidate
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(
            {
                "error_code": "bad_request",
                "message": reformatted_message,
                "http_status": http.HTTPStatus.BAD_REQUEST.value,
            }
        ),
        headers=headers,
    )


async def pydantic_validation_exception_handler(_request: Request, exception: Exception) -> JSONResponse:
    """
    Converts a pydantic ValidationError to a better readable Bad request exception.
    """
    if not isinstance(exception, pydantic.ValidationError):
        raise exception

    def format_location(loc: Sequence[str | int]) -> str:
        """
        Format location path with proper dot notation and array indices.

        Example:
            format_location(['a', 0, 'b', 1, 'c']) -> 'a[0].b[1].c'
        """
        result = ""
        for i, item in enumerate(loc):
            if isinstance(item, int):
                result += f"[{item}]"
            else:
                result += f".{item}" if i > 0 else str(item)
        return result

    errors = [
        {"message": error["msg"], "type": error["type"], "location": format_location(error.get("loc", []))}
        for error in exception.errors()
    ]

    headers = {"Cache-Control": "no-cache"}  # always revalidate
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(
            {
                "error_code": "invalid_payload",
                "errors": errors,
                "http_status": http.HTTPStatus.BAD_REQUEST.value,
            }
        ),
        headers=headers,
    )


def register_application_exception_handlers(app: FastAPI) -> None:
    """
    Register application exception handlers
    """
    app.add_exception_handler(GetiBaseException, handle_base_exception)

    app.add_exception_handler(500, handle_error)
    app.add_exception_handler(404, handle_not_found)

    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(pydantic.ValidationError, pydantic_validation_exception_handler)
