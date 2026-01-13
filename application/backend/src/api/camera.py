from typing import Annotated

from fastapi import APIRouter, Depends
from frame_source import FrameSourceFactory

from api.dependencies import get_webrtc_manager
from schemas import Camera, CameraProfile
from schemas.camera import SupportedCameraFormat
from webrtc.manager import Answer, Offer, WebRTCManager

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


@router.post("/offer/camera")
async def offer_camera(
    offer: Offer,
    driver: str,
    camera: str,
    width: int,
    height: int,
    fps: int,
    webrtc_manager: Annotated[WebRTCManager, Depends(get_webrtc_manager)],
) -> Answer:
    """Create a WebRTC offer"""
    config = Camera(
        name=camera,
        fingerprint=camera,
        driver=driver,
        default_stream_profile=CameraProfile(
            width=width,
            height=height,
            fps=fps,
        ),
    )
    return await webrtc_manager.handle_offer(
        offer.sdp, offer.type, offer.webrtc_id, config, config.default_stream_profile
    )


@router.get("/supported_formats/{driver}")
async def get_supported_formats(
    driver: str,
    fingerprint: str,
) -> list[SupportedCameraFormat]:
    """Returns the supported camera resolution and fps associated to the camera"""
    camera = FrameSourceFactory.create(driver, source=fingerprint)
    formats = camera.get_supported_formats()

    return [
        SupportedCameraFormat(width=format["width"], height=format["height"], fps=format["fps"]) for format in formats
    ]
