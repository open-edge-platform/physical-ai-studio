from typing import Annotated

from fastapi import APIRouter, Depends

from schemas import Camera, CameraProfile
from api.dependencies import get_webrtc_manager
from webrtc.manager import Answer, Offer, WebRTCManager

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


@router.post("/offer/camera")
async def offer_camera(
    offer: Offer, driver: str, camera: str, width: int, height: int, fps: int, webrtc_manager: Annotated[WebRTCManager, Depends(get_webrtc_manager)]
) -> Answer:
    """Create a WebRTC offer"""
    config = Camera(
        name=camera,
        port_or_device_id=camera,
        driver=driver,
        default_stream_profile=CameraProfile(
            width=width,
            height=height,
            fps=fps,
        )
    )
    return await webrtc_manager.handle_offer(offer.sdp, offer.type, offer.webrtc_id, config, config.default_stream_profile)
