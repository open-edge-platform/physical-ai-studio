from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from pydantic import BaseModel


class Offer(BaseModel):
    sdp: str
    type: str
    webrtc_id: str


class Answer(BaseModel):
    sdp: str
    type: str


class WebRTCManager:
    def __init__(self):
        # active peer connections
        self._pcs: dict[str, RTCPeerConnection] = {}
        # one MediaPlayer per device (/dev/video2, /dev/video4, etc.)
        self._players: dict[str, MediaPlayer] = {}
        # track which webrtc connections are using which devices
        self._device_connections: dict[str, set[str]] = {}

    def get_player(self, device: str) -> MediaPlayer:
        """
        Create or reuse a MediaPlayer for the given device.
        Ensures the device is only opened once.
        """
        if device not in self._players:
            self._players[device] = MediaPlayer(device, options={"video_size": "640x480"})
            # Initialize the set of connections for this device
            self._device_connections[device] = set()
        return self._players[device]

    async def handle_offer(self, sdp: str, type: str, webrtc_id: str, device: str) -> Answer:
        pc = RTCPeerConnection()
        self._pcs[webrtc_id] = pc

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"🔌 {webrtc_id} connection state: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self.cleanup_peer(webrtc_id)

        # apply the remote description from the browser
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type))

        # add shared video track
        player = self.get_player(device)

        # Record which device this connection is using
        self._device_connections[device].add(webrtc_id)

        if player.video:
            pc.addTrack(player.video)

        # send answer back to browser
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return Answer(sdp=pc.localDescription.sdp, type=pc.localDescription.type)

    async def cleanup_peer(self, webrtc_id: str) -> None:
        """
        Clean up a single peer connection.
        If it's the last connection using a device, also clean up the device.
        """
        pc = self._pcs.pop(webrtc_id, None)
        if pc:
            await pc.close()

            # Check if this connection was associated with any device
            for device, connections in self._device_connections.items():
                if webrtc_id in connections:
                    connections.remove(webrtc_id)

                    # If this was the last connection using this device, clean up the player
                    if not connections:
                        player = self._players.pop(device, None)
                        if player and player.video:
                            player.video.stop()
                        print(f"🎥 Closed video player for device {device} - no more active connections")

    async def cleanup(self) -> None:
        """
        Clean up everything: all peer connections and all MediaPlayers.
        Call this on server shutdown.
        """
        # close all peer connections
        for pc in list(self._pcs.values()):
            await pc.close()
        self._pcs.clear()

        # close MediaPlayers (release /dev/videoX)
        for player in self._players.values():
            if player.video:
                player.video.stop()

        self._players.clear()
        self._device_connections.clear()
