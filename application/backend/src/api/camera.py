import json
from collections import defaultdict
from functools import cache
from typing import Annotated

from fastapi import APIRouter, Depends, Query, WebSocket
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect

from api.dependencies import SchedulerDep
from schemas.camera import SupportedCameraFormat
from schemas.project_camera import Camera as ProjectCamera
from schemas.project_camera import CameraAdapter
from workers.base import run_at_frequency
from workers.camera_worker import CameraWorker

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


def _formats_to_response(raw: list[tuple[int, int, int]]) -> list[SupportedCameraFormat]:
    """Group (w, h, fps) tuples into SupportedCameraFormat responses."""
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for w, h, fps in raw:
        grouped[(w, h)].append(fps)

    return [
        SupportedCameraFormat(width=w, height=h, fps=sorted(fps_list)) for (w, h), fps_list in sorted(grouped.items())
    ]


# Cache required on MacOS to avoid repeated cam.open() which may block camera stream
@cache
def _query_formats(driver: str, fingerprint: str) -> list[SupportedCameraFormat]:
    """Query real formats from a device via physicalai.capture."""
    if driver == "usb_camera":
        from physicalai.capture import UVCCamera

        return _formats_to_response(UVCCamera.query_formats(fingerprint))

    if driver == "realsense":
        from physicalai.capture.cameras.realsense import RealSenseCamera

        return _formats_to_response(RealSenseCamera.query_formats(fingerprint))

    if driver == "basler":
        # TODO: Replace with cached Basler hardware discovery once implementated in physicalai.capture
        return _formats_to_response([(640, 480, 30), (768, 480, 30), (1920, 1200, 30)])

    msg = f"Format discovery not supported for driver {driver!r}"
    raise ValueError(msg)


@router.get("/supported_formats/{driver}")
async def get_supported_formats(
    driver: str,
    fingerprint: str,
) -> list[SupportedCameraFormat]:
    """Returns the supported camera resolution and fps associated to the camera."""
    return _query_formats(driver, fingerprint)


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
    camera: Annotated[str | None, Query(description="JSON-serialized ProjectCamera configuration")] = None,  # noqa: ARG001
) -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def camera_websocket(
    websocket: WebSocket,
    scheduler: SchedulerDep,
    camera: Annotated[ProjectCamera, Depends(get_camera_from_query)],
) -> None:
    """
    WebSocket endpoint for camera streaming.

    Query Parameters:
        camera: JSON serialized ProjectCamera

    Protocol:
        Client sends JSON message on connect:
            camera: JSON serialized ProjectCamera

        Server sends jpeg encoded bytes
    """
    from utils.jpeg import encode_jpeg_rgb

    await websocket.accept()

    worker = None
    try:
        worker = CameraWorker(camera, scheduler.mp_stop_event)
        worker.start()
        while True:
            async with run_at_frequency(camera.payload.fps):
                frame = worker.get_frame()
                await websocket.send_bytes(encode_jpeg_rgb(frame))
    except WebSocketDisconnect:
        pass
    finally:
        if worker:
            worker.stop()
