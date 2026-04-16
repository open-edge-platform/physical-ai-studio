from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import RequestResponseEndpoint

from exceptions import UploadTooLargeError
from settings import get_settings


async def upload_size_guard_middleware(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Reject oversized dataset-upload requests before body processing."""
    raw = request.headers.get("content-length")

    if raw is not None:
        try:
            content_length = int(raw)
        except ValueError as error:
            raise UploadTooLargeError("Invalid Content-Length header") from error

        settings = get_settings()
        if content_length > settings.data_import_max_upload_bytes:
            upload_error = UploadTooLargeError(
                f"Upload size ({content_length} bytes) exceeds the maximum allowed upload size "
                f"({settings.data_import_max_upload_bytes} bytes)"
            )
            return JSONResponse(
                content=jsonable_encoder(
                    {
                        "error_code": upload_error.error_code,
                        "message": upload_error.message,
                        "http_status": upload_error.http_status,
                    }
                ),
                status_code=int(upload_error.http_status),
                headers={"Cache-Control": "no-cache"},
            )

    return await call_next(request)
