import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from api.dependencies import get_webrtc_manager
from schemas import Camera
from utils.camera import gen_camera_frames
from webrtc.manager import Answer, Offer, WebRTCManager

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


@router.post("/offer/camera")
async def offer_camera(
    offer: Offer, camera: str, webrtc_manager: Annotated[WebRTCManager, Depends(get_webrtc_manager)]
) -> Answer:
    """Create a WebRTC offer"""
    return await webrtc_manager.handle_offer(offer.sdp, offer.type, offer.webrtc_id, camera)


@router.websocket("/offer/camera/ws")
async def camera_feed_websocket(websocket: WebSocket) -> None:
    """Camera feed. Awaits json package with CameraConfig."""
    await websocket.accept()
    stop_event = asyncio.Event()
    print("Connected...")
    try:
        data = await websocket.receive_json("text")
        print(data)
        config = Camera(**data)
        await gen_camera_frames(websocket, stop_event, config)
    except WebSocketDisconnect:
        stop_event.set()
        print("Disconnected")
