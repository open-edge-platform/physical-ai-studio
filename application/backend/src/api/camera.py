from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_webrtc_manager
from webrtc.manager import Answer, Offer, WebRTCManager

router = APIRouter()


@router.post("/offer/camera")
async def offer_camera(
    offer: Offer, camera: str, webrtc_manager: Annotated[WebRTCManager, Depends(get_webrtc_manager)]
) -> Answer:
    """Create a WebRTC offer"""
    return await webrtc_manager.handle_offer(offer.sdp, offer.type, offer.webrtc_id, camera)
