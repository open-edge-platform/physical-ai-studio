from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from webrtc.manager import Answer, Offer, WebRTCManager

router = APIRouter()
manager = WebRTCManager()


@router.post("/offer/camera")
async def offer_camera(offer: Offer, camera: str) -> Answer:
    """Create a WebRTC offer"""
    return await manager.handle_offer(offer.sdp, offer.type, offer.webrtc_id, camera)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager for WebRTCManager"""
    # Startup
    print("🚀 Starting up WebRTCManager")
    yield
    # Shutdown
    print("🛑 Cleaning up WebRTCManager")
    await manager.cleanup()
