import json
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, WebSocket, status
from fastapi.responses import Response
from frame_source import FrameSourceFactory
from loguru import logger

from api.dependencies import CameraRegistryDep
from schemas.camera import SupportedCameraFormat
from schemas.project_camera import Camera as ProjectCamera
from schemas.project_camera import CameraAdapter
from workers.camera_worker import CameraWorker
from workers.transport.websocket_transport import WebSocketTransport

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


@router.get("/supported_formats/{driver}")
async def get_supported_formats(
    driver: str,
    fingerprint: str,
) -> list[SupportedCameraFormat]:
    """Returns the supported camera resolution and fps associated to the camera"""
    camera = FrameSourceFactory.create(driver if driver != "usb_camera" else "webcam", source=fingerprint)
    formats = camera.get_supported_formats()

    if formats is None:
        return []

    return [
        SupportedCameraFormat(width=format["width"], height=format["height"], fps=format["fps"]) for format in formats
    ]


def get_camera_from_query(websocket: WebSocket) -> ProjectCamera:
    """Parse camera from query parameters."""
    camera_param = websocket.query_params.get("camera")
    if not camera_param:
        raise ValueError("Missing 'camera' query parameter")

    try:
        camera_data = json.loads(camera_param)
        return CameraAdapter.validate_python(camera_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in camera parameter: {e}")
    except Exception as e:
        raise ValueError(f"Invalid camera configuration: {e}")


@router.get("/ws", tags=["WebSocket"], summary="Camera streaming (WebSocket)", status_code=426)
async def camera_websocket_openapi(
    camera: Annotated[str | None, Query(description="JSON-serialized ProjectCamera configuration")] = None,
) -> Response:
    """This endpoint requires a WebSocket connection. Use `ws://` or `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def camera_websocket(
    websocket: WebSocket,
    registry: CameraRegistryDep,
    camera: Annotated[ProjectCamera, Depends(get_camera_from_query)],
) -> None:
    """
    WebSocket endpoint for camera streaming.

    Query Parameters:
        camera: JSON serialized ProjectCamera

    Protocol:
        Client sends JSON messages:
            {"event": "disconnect"} - Request graceful disconnect
            {"event": "ping"} - Keep-alive check

        Server sends JSON-encoded messages with status updates:
            {"event": "status", "state": "running", ...}
    """
    await websocket.accept()

    worker_id = uuid4()
    camera.id = worker_id

    # Use WebSocket transport
    transport = WebSocketTransport(websocket)
    worker = CameraWorker(camera, transport)

    try:
        await registry.create_and_register(worker_id, worker)
        await worker.run()
    except ValueError as e:
        logger.error(f"Failed to register worker: {e}")
        try:
            await websocket.send_json(
                {
                    "event": "error",
                    "message": str(e),
                }
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        except Exception:
            logger.error(f"Could not close websocket: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in camera websocket: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as e:
            logger.error(f"Could not close websocket: {e}")
    finally:
        await registry.unregister(worker_id)
