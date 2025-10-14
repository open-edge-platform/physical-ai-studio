import queue
from typing import TYPE_CHECKING

from aiortc import RTCPeerConnection, RTCSessionDescription
from pydantic import BaseModel

from webrtc.framesource_track import FrameSourceVideoStreamTrack

if TYPE_CHECKING:
    import numpy as np


class Offer(BaseModel):
    sdp: str
    type: str
    webrtc_id: str


class Answer(BaseModel):
    sdp: str
    type: str


class WebRTCManager:
    def __init__(self):
        self._pcs: dict[str, RTCPeerConnection] = {}
        self._frame_queues: dict[str, queue.Queue] = {}
        self._tracks: dict[str, FrameSourceVideoStreamTrack] = {}
        self._device_connections: dict[str, set[str]] = {}

    @staticmethod
    def identifier_for_driver_device(driver: str, device: str) -> str:
        return driver + "@" + device

    def get_track(self, driver: str, device: str) -> FrameSourceVideoStreamTrack:
        """
        Create or reuse a FrameSourceVideoStreamTrack for the given device.
        Each device corresponds to a queue feeding numpy frames.
        """
        identifier = self.identifier_for_driver_device(driver, device)
        if identifier not in self._tracks:
            # Create a new queue for incoming frames
            q: queue.Queue[np.ndarray] = queue.Queue(maxsize=10)
            self._frame_queues[identifier] = q
            self._tracks[identifier] = FrameSourceVideoStreamTrack(driver=driver, device=device)
            self._device_connections[identifier] = set()
        return self._tracks[identifier]

    async def handle_offer(self, sdp: str, type: str, webrtc_id: str, driver: str, device: str) -> Answer:
        pc = RTCPeerConnection()
        self._pcs[webrtc_id] = pc

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"🔌 {webrtc_id} connection state: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self.cleanup_peer(webrtc_id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type))
        track = self.get_track(driver, device)

        identifier = self.identifier_for_driver_device(driver, device)
        self._device_connections[identifier].add(webrtc_id)
        pc.addTrack(track)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return Answer(sdp=pc.localDescription.sdp, type=pc.localDescription.type)

    async def cleanup_peer(self, webrtc_id: str) -> None:
        """
        Clean up a single peer connection.
        If it's the last connection using a device, also clean up that device's track and queue.
        """
        pc = self._pcs.pop(webrtc_id, None)
        if pc:
            await pc.close()

            for device, connections in list(self._device_connections.items()):
                if webrtc_id in connections:
                    connections.remove(webrtc_id)
                    if not connections:
                        # Last connection using this device
                        self._tracks.pop(device, None)
                        self._frame_queues.pop(device, None)
                        print(f"🎥 Closed track for device {device} - no more active connections")

    async def cleanup(self) -> None:
        """
        Clean up everything: all peer connections and all FrameSourceVideoStreamTracks.
        Call this on server shutdown.
        """
        for pc in list(self._pcs.values()):
            await pc.close()
        self._pcs.clear()

        self._tracks.clear()
        self._frame_queues.clear()
        self._device_connections.clear()
