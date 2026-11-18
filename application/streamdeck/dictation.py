"""Local, bounded microphone capture; only the resulting text leaves this process."""

from __future__ import annotations

import io
import threading
import wave
from typing import Any


class LocalDictation:
    """Keep capture and CPU transcription off the deck's control/heartbeat loop."""

    MAX_SECONDS = 60

    def __init__(self) -> None:
        self.status = "idle"
        self.task = ""
        self.error = ""
        self.stop = threading.Event()
        self.cancelled = threading.Event()
        self.thread: threading.Thread | None = None
        self.model: Any = None

    def start(self) -> None:
        """Use the OS default input device, asking for permission through the host OS."""
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop.clear()
        self.cancelled.clear()
        self.task = self.error = ""
        self.status = "preparing"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def finish(self) -> None:
        """A second explicit press finishes capture; preparation/transcription cannot be queued."""
        if self.status == "listening":
            self.stop.set()

    def cancel(self) -> None:
        """Close capture promptly and discard even an already-running transcription's result."""
        self.cancelled.set()
        self.stop.set()
        if self.thread is None or not self.thread.is_alive():
            self.task = ""
            self.status = "idle"

    def _run(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd

            chunks = []

            def capture(data: Any, _frames: int, _time: Any, status: Any) -> None:
                if status:
                    self.error = f"Microphone capture failed: {status}"
                    self.stop.set()
                if not self.stop.is_set():
                    chunks.append(data[:, 0].copy())

            if self.cancelled.is_set():
                return
            sample_rate = int(sd.query_devices(kind="input")["default_samplerate"])
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", callback=capture):
                self.status = "listening"
                self.stop.wait(self.MAX_SECONDS)
            if self.cancelled.is_set():
                return
            if self.error or not chunks:
                raise ValueError(self.error or "No microphone audio captured. Check the default input and permissions.")
            self.status = "transcribing"
            from faster_whisper import WhisperModel

            if self.model is None:
                # The first use downloads model weights, never microphone audio.
                self.model = WhisperModel("small.en", device="cpu", compute_type="int8")
            if self.cancelled.is_set():
                return
            # Let Whisper resample an in-memory WAV from the microphone's native rate.
            audio = io.BytesIO()
            with wave.open(audio, "wb") as recording:
                recording.setnchannels(1)
                recording.setsampwidth(2)
                recording.setframerate(sample_rate)
                recording.writeframes(np.concatenate(chunks).tobytes())
            audio.seek(0)
            segments, _ = self.model.transcribe(audio, language="en", vad_filter=True, condition_on_previous_text=False)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if self.cancelled.is_set():
                return
            if not text:
                raise ValueError("No speech detected. Try the microphone again.")
            if len(text) > 20_000:
                raise ValueError("Dictated task is too long. Record a shorter prompt.")
            self.task = text
            self.status = "done"
        except Exception as error:  # Hardware, permissions and optional model downloads all fail closed.
            self.error = str(error)
            self.status = "error"
        finally:
            if self.cancelled.is_set():
                self.task = ""
                self.status = "idle"
