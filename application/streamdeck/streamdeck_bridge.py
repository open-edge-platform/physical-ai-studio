# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "streamdeck>=0.10,<0.11", "pillow>=11,<13", "httpx>=0.28,<0.29",
#     "sounddevice>=0.5,<0.6", "faster-whisper>=1.2,<2", "numpy>=2,<3",
#     "av>=11,<19", # faster-whisper 1.2 passes metadata_errors, removed in PyAV 19.
# ]
# ///
"""Run recording and inference controls on a USB Stream Deck without the Elgato desktop app."""

from __future__ import annotations

import argparse
import contextlib
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx
from dictation import LocalDictation
from PIL import ImageDraw, ImageFont
from StreamDeck.DeviceManager import DeviceManager, ProbeError
from StreamDeck.ImageHelpers import PILHelper
from StreamDeck.Transport.Transport import TransportError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from StreamDeck.Devices.StreamDeck import StreamDeck

logger = logging.getLogger(__name__)

DISABLED = "#292929"
GREEN = "#126631"
RED = "#9c2424"
BLUE = "#174b85"
AMBER = "#845400"


class RecordingBridge:
    """Keep button availability tied to server state, never to an optimistic toggle."""

    def __init__(self, client: httpx.Client, session_id: str, task: str) -> None:
        self.client = client
        self.path = f"/api/runtime/sessions/{quote(session_id, safe='')}"
        self.task = task
        self.command_headers: dict[str, str] = {}
        self.state: dict[str, Any] | None = None
        self.error = ""
        self.discard_until = 0.0

    def refresh(self) -> None:
        """Read the selected session; a failed refresh disables every action."""
        try:
            response = self.client.get(f"{self.path}/state")
            response.raise_for_status()
            self.state = response.json()
        except (httpx.HTTPError, ValueError):
            self.state = None
            self.discard_until = 0.0
            raise
        if not (self.state.get("dataset") or {}).get("is_recording"):
            self.discard_until = 0.0

    def buttons(self, now: float) -> list[tuple[str, str]]:
        """Return six labels and backgrounds in logical row order."""
        if self.state is None:
            return [
                ("START\nEPISODE", DISABLED),
                ("ACCEPT\nSAVE", DISABLED),
                ("DISCARD\nEPISODE", DISABLED),
                ("OFFLINE", RED),
                ("SAVED\n--", DISABLED),
                ("RETRYING", AMBER),
            ]
        dataset = self.state.get("dataset")
        recording = bool(dataset and dataset["is_recording"])
        ready = bool(dataset and self.state.get("environment"))
        status = "RECORDING" if recording else "READY"
        if not ready:
            status = "NO\nDATASET" if not dataset else "NO\nROBOT"
        elif not recording and not self.task.strip():
            status = "NO\nTASK"
        return [
            ("START\nEPISODE", GREEN if ready and self.task.strip() and not recording else DISABLED),
            ("ACCEPT\nSAVE", BLUE if recording else DISABLED),
            ("CONFIRM\nDISCARD" if now < self.discard_until else "DISCARD\nEPISODE", AMBER if recording else DISABLED),
            (status, RED if recording else DISABLED),
            (f"SAVED\n{dataset['episodes_recorded']}" if dataset else "SAVED\n--", DISABLED),
            ("ERROR\nSEE LOG" if self.error else "CONNECTED", AMBER if self.error else GREEN),
        ]

    def press(self, key: int, now: float) -> None:
        """Send one explicit command; discard requires two presses within two seconds."""
        if self.state is None:
            return
        dataset = self.state.get("dataset")
        if not dataset:
            return
        recording = dataset["is_recording"]
        command: dict[str, Any]
        if key == 0 and not recording and self.task.strip() and self.state.get("environment"):
            command = {"command": "start_episode", "task": self.task}
        elif key == 1 and recording:
            command = {"command": "save_episode"}
        elif key == 2 and recording:
            if now >= self.discard_until:
                self.discard_until = now + 2.0
                return
            command = {"command": "discard_episode"}
        else:
            return
        self.discard_until = 0.0
        self.error = ""
        # Do not retry commands: a lost response does not mean the command failed.
        try:
            response = self.client.post(
                f"{self.path}/recording",
                json={**command, "request_id": uuid4().hex},
                headers=self.command_headers,
            )
            response.raise_for_status()
            ack = response.json()
            if not ack["ok"]:
                self.error = (ack.get("error") or {}).get("message", "Recording command failed")
                logger.error("%s", self.error)
        finally:
            # Require a fresh snapshot before another key can act, including after a timeout.
            self.state = None


class DeckDisplay:
    """Render only changed keys, leaving spare keys blank."""

    def __init__(self, deck: StreamDeck) -> None:
        self.deck = deck
        rows, columns = deck.key_layout()
        if rows < 2 or columns < 3:
            raise ValueError("A Stream Deck with at least two rows and three LCD keys is required.")
        self.keys = [0, 1, 2, columns, columns + 1, columns + 2]
        self.previous: list[tuple[str, str]] = []

    def render(self, buttons: list[tuple[str, str]]) -> None:
        """Draw a compact two-row layout on Mini, Neo and larger LCD decks."""
        for index, (label, color) in enumerate(buttons):
            if self.previous and self.previous[index] == (label, color):
                continue
            image = PILHelper.create_key_image(self.deck, background=color)
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=max(10, image.width // 8))
            draw.multiline_text(
                (image.width / 2, image.height / 2), label, font=font, anchor="mm", align="center", fill="white"
            )
            self.deck.set_key_image(self.keys[index], PILHelper.to_native_key_format(self.deck, image))
        self.previous = buttons


def run_deck(deck: StreamDeck, bridge: RecordingBridge) -> None:
    """Poll status at four Hz and drop input while commands or network requests run."""
    display = DeckDisplay(deck)
    presses: queue.Queue[tuple[int, float]] = queue.Queue(maxsize=1)
    accepting = threading.Event()
    ready_since = time.monotonic()
    offline = False

    def on_key(_deck: StreamDeck, key: int, pressed: bool) -> None:
        if pressed and accepting.is_set() and key in display.keys[:3]:
            with contextlib.suppress(queue.Full):
                presses.put_nowait((display.keys.index(key), time.monotonic()))

    deck.set_key_callback(on_key)
    try:
        display.render(bridge.buttons(time.monotonic()))
        while deck.is_open():
            accepting.clear()
            try:
                bridge.refresh()
                offline = False
            except (httpx.HTTPError, ValueError) as error:
                bridge.state = None
                bridge.discard_until = 0.0
                if not offline:
                    logger.error("Studio unavailable: %s", error)
                offline = True
                ready_since = time.monotonic()
            display.render(bridge.buttons(time.monotonic()))
            if bridge.state is not None:
                accepting.set()
            try:
                key, pressed_at = presses.get(timeout=0.25)
            except queue.Empty:
                continue
            accepting.clear()
            if pressed_at < ready_since:
                continue
            buttons = bridge.buttons(time.monotonic())
            display.render([(label, DISABLED) for label, _ in buttons[:3]] + [("WORKING", AMBER), *buttons[4:]])
            try:
                bridge.press(key, pressed_at)
            except (httpx.HTTPError, ValueError) as error:
                bridge.error = str(error)
                logger.error("Command outcome unknown; check Studio before pressing again: %s", error)
            finally:
                ready_since = time.monotonic()
        raise TransportError("Stream Deck disconnected; reconnect it and restart the bridge.")
    finally:
        accepting.clear()
        deck.set_key_callback(None)


@contextlib.contextmanager
def opened_deck() -> Iterator[StreamDeck]:
    """Take exclusive USB ownership only while a workspace has requested the deck."""
    decks = [deck for deck in DeviceManager().enumerate() if deck.is_visual()]
    if len(decks) != 1:
        raise ValueError("Connect exactly one LCD Stream Deck and close other Stream Deck software.")
    deck = decks[0]
    deck.open()
    try:
        deck.reset()
        yield deck
    finally:
        with contextlib.suppress(TransportError):
            if deck.is_open():
                deck.reset()
        deck.close()


class AttachmentChanged(Exception):
    """Release USB and drop queued input when the UI detaches or replaces a target."""


def heartbeat(
    client: httpx.Client,
    helper_id: str,
    connection_id: str | None = None,
    device: str | None = None,
    error: str | None = None,
    dictation_status: str = "idle",
) -> dict[str, Any]:
    """Report availability and receive the UI's current desired connection."""
    response = client.post(
        "/api/streamdeck/heartbeat",
        json={
            "helper_id": helper_id,
            "connection_id": connection_id,
            "device": device,
            "error": error,
            "dictation_status": dictation_status,
        },
    )
    response.raise_for_status()
    return response.json()


class ManagedRecordingBridge(RecordingBridge):
    """Follow the UI's explicit attachment and task without opening a second runtime session."""

    def __init__(self, client: httpx.Client, helper_id: str, target: dict[str, Any], device: str) -> None:
        super().__init__(client, target["session_id"], target["task"])
        self.helper_id = helper_id
        self.connection_id = target["connection_id"]
        self.device = device
        self.command_headers = {"X-StreamDeck-Connection": self.connection_id}
        self.target = target

    def dictation_status(self) -> str:
        """Recording pages never capture microphone input."""
        return "idle"

    def refresh(self) -> None:
        try:
            target = heartbeat(
                self.client,
                self.helper_id,
                self.connection_id,
                self.device,
                self.error[:2000] or None,
                self.dictation_status(),
            )["target"]
            if target is None or target["connection_id"] != self.connection_id:
                raise AttachmentChanged
            self.target = target
            self.task = target["task"]
            super().refresh()
        except (httpx.HTTPError, ValueError, AttachmentChanged):
            self.state = None
            self.discard_until = 0.0
            raise


class ManagedInferenceBridge(ManagedRecordingBridge):
    """Explicit inference start/hold and stopped-only voice drafts on a separate layout."""

    def __init__(self, client: httpx.Client, helper_id: str, target: dict[str, Any], device: str) -> None:
        super().__init__(client, helper_id, target, device)
        self.dictation = LocalDictation()

    def dictation_status(self) -> str:
        return self.dictation.status

    def refresh(self) -> None:
        try:
            super().refresh()
        except (httpx.HTTPError, ValueError, AttachmentChanged):
            self.dictation.cancel()
            raise
        if self.state is None or self.state.get("control") is not None:
            self.dictation.cancel()
            return
        if self.dictation.status == "error":
            self.error = self.dictation.error
            logger.error("Dictation failed: %s", self.error)
            self.dictation.status = "idle"
        elif self.dictation.status == "done" and not self.dictation.cancelled.is_set():
            text = self.dictation.task
            self.dictation.status = "idle"
            # Never retry a result: a failed reply may still have delivered the draft.
            response = self.client.post(
                f"/api/streamdeck/sessions/{quote(self.target['session_id'], safe='')}/dictation",
                json={"task": text},
                headers=self.command_headers,
            )
            response.raise_for_status()
            self.task = text

    def buttons(self, _now: float) -> list[tuple[str, str]]:
        control = (self.state or {}).get("control")
        running = bool(control and control["type"] == "model")
        ready = bool(self.state and self.state.get("environment") and self.state.get("model"))
        stopped = self.state is not None and control is None
        voice = self.dictation.status
        busy = voice in {"preparing", "listening", "transcribing"}
        control_status = "RUNNING" if running else "STOPPED" if stopped else "OTHER\nCONTROL"
        return [
            ("START\nINFERENCE", GREEN if ready and stopped and not busy else DISABLED),
            ("STOP\nHOLD", RED if running else DISABLED),
            (
                "FINISH\nDICTATION" if voice == "listening" else "MIC\nNEW TASK",
                BLUE if stopped and voice in {"idle", "listening"} else DISABLED,
            ),
            (
                control_status if self.state is not None else "OFFLINE",
                GREEN if running else DISABLED,
            ),
            (voice.upper(), AMBER if busy else DISABLED),
            ("ERROR\nSEE LOG" if self.error else "INFERENCE\nPAGE", AMBER if self.error else BLUE),
        ]

    def press(self, key: int, _now: float) -> None:
        if self.state is None:
            return
        control = self.state.get("control")
        voice = self.dictation.status
        if key == 2 and control is None:
            if voice == "listening":
                self.dictation.finish()
            elif voice == "idle":
                self.error = ""
                self.dictation.start()
            return
        if (
            key == 0
            and control is None
            and voice == "idle"
            and self.state.get("environment")
            and self.state.get("model")
        ):
            action = "start"
        elif key == 1 and control and control["type"] == "model":
            action = "stop"
        else:
            return
        self.error = ""
        try:
            response = self.client.post(
                f"/api/streamdeck/sessions/{quote(self.target['session_id'], safe='')}/inference",
                json={"action": action, "request_id": uuid4().hex},
                headers=self.command_headers,
            )
            response.raise_for_status()
            ack = response.json()
            if not ack["ok"]:
                self.error = (ack.get("error") or {}).get("message", "Inference command failed")
                logger.error("%s", self.error)
        finally:
            self.state = None


def run_managed(client: httpx.Client) -> None:
    """Stay available to Studio; idle helpers never enumerate or claim USB devices."""
    helper_id = str(uuid4())
    dictation = LocalDictation()
    error = None
    last_error = None
    logger.info("Helper ready. Use Connect Stream Deck in Studio's recording or inference workspace.")
    while True:
        try:
            target = heartbeat(client, helper_id, error=error)["target"]
            error = None
            if target is None:
                time.sleep(1.0)
                continue
            with opened_deck() as deck:
                logger.info("%s → session %s", deck.deck_type(), target["session_id"])
                last_error = None
                bridge_type = ManagedInferenceBridge if target.get("page") == "inference" else ManagedRecordingBridge
                bridge = bridge_type(client, helper_id, target, deck.deck_type())
                if isinstance(bridge, ManagedInferenceBridge):
                    bridge.dictation = dictation
                try:
                    run_deck(deck, bridge)
                finally:
                    if isinstance(bridge, ManagedInferenceBridge):
                        bridge.dictation.cancel()
        except AttachmentChanged:
            continue
        except (httpx.HTTPError, ProbeError, TransportError, ValueError) as failure:
            error = str(failure)[:2000]
            if error != last_error:
                logger.error("%s", error)
                last_error = error
            time.sleep(1.0)


def main() -> None:
    """Run the UI-managed helper, list sessions, or explicitly attach from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:7860", help="Studio backend URL (not the UI dev server)")
    parser.add_argument("--list-sessions", action="store_true", help="List sessions without opening the USB device")
    parser.add_argument("--session", help="Exact session ID from --list-sessions; never switches automatically")
    parser.add_argument("--task", help="Task text for episodes started by this bridge")
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
        parser.error("--url must be an http:// or https:// backend URL without a query or fragment")
    if (
        not args.list_sessions
        and (args.session or args.task)
        and (not args.session or not args.task or not args.task.strip())
    ):
        parser.error("For manual mode, both --session and a non-empty --task are required")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        with httpx.Client(base_url=args.url.rstrip("/"), timeout=10.0, follow_redirects=False) as client:
            if args.list_sessions:
                response = client.get("/api/runtime/sessions")
                response.raise_for_status()
                for session in response.json()["sessions"]:
                    environment = session["environment_name"] or "(empty)"
                    print(f"{session['session_id']}  {session['project_name']}  {environment}")
                return
            if args.session is None:
                run_managed(client)
            else:
                with opened_deck() as deck:
                    logger.info("%s → session %s; task: %s", deck.deck_type(), args.session, args.task)
                    run_deck(deck, RecordingBridge(client, args.session, args.task))
    except KeyboardInterrupt:
        pass
    except (httpx.HTTPError, ProbeError, TransportError, ValueError) as error:
        parser.exit(1, f"{error}\nSee application/streamdeck/README.md for USB setup and troubleshooting.\n")


if __name__ == "__main__":
    main()
