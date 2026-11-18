"""No microphone, model weights, network, or audio files are used by these checks."""

import io
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from dictation import LocalDictation


class DictationTests(unittest.TestCase):
    def test_real_whisper_decoder_accepts_native_rate_in_memory_wav(self):
        from faster_whisper.audio import decode_audio

        audio = io.BytesIO()
        with wave.open(audio, "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(48000)
            recording.writeframes(bytes(48000 * 2))
        audio.seek(0)
        samples = decode_audio(audio)
        self.assertEqual(samples.shape, (16000,))
        self.assertEqual(samples.dtype, np.float32)

    def run_capture(self, voice, text="New task", *, capture=True, transcribe=None):
        def enter():
            if capture:
                callback = sound.InputStream.call_args.kwargs["callback"]
                callback(np.ones((480, 1), dtype="int16"), 480, None, False)
            voice.stop.set()

        def decode(audio, **options):
            self.assertEqual(options, {"language": "en", "vad_filter": True, "condition_on_previous_text": False})
            with wave.open(audio, "rb") as recording:
                self.assertEqual(recording.getframerate(), 48000)
                self.assertEqual(recording.getnchannels(), 1)
                self.assertEqual(recording.getnframes(), 480)
            return [SimpleNamespace(text=text)], None

        sound = Mock()
        sound.query_devices.return_value = {"default_samplerate": 48000}
        sound.InputStream.return_value.__enter__ = Mock(side_effect=enter)
        sound.InputStream.return_value.__exit__ = Mock(return_value=False)
        whisper = Mock()
        whisper.WhisperModel.return_value.transcribe.side_effect = transcribe or decode
        with (
            patch.dict("sys.modules", sounddevice=sound, faster_whisper=whisper),
            patch.object(voice.stop, "wait", return_value=True) as wait,
        ):
            voice._run()
            if not voice.cancelled.is_set():
                wait.assert_called_once_with(60)
        return sound, whisper

    def test_native_rate_audio_stays_in_memory_and_becomes_text(self):
        voice = LocalDictation()
        sound, whisper = self.run_capture(voice, " Move the block ")
        self.assertEqual(voice.status, "done")
        self.assertEqual(voice.task, "Move the block")
        self.assertEqual(sound.InputStream.call_args.kwargs["samplerate"], 48000)
        whisper.WhisperModel.assert_called_once_with("small.en", device="cpu", compute_type="int8")
        voice.stop.clear()
        self.run_capture(voice)
        self.assertEqual(whisper.WhisperModel.return_value.transcribe.call_count, 2)
        self.assertEqual(whisper.WhisperModel.call_count, 1)  # cached model is reused

    def test_capture_failure_no_audio_and_invalid_transcripts_fail_closed(self):
        for capture, text in [(False, "task"), (True, " "), (True, "x" * 20_001)]:
            voice = LocalDictation()
            self.run_capture(voice, text, capture=capture)
            self.assertEqual(voice.status, "error")
            self.assertFalse(voice.task)
            self.assertTrue(voice.error)
        sound = Mock()
        sound.query_devices.side_effect = RuntimeError("microphone permission denied")
        voice = LocalDictation()
        with patch.dict("sys.modules", sounddevice=sound):
            voice._run()
        self.assertEqual(voice.error, "microphone permission denied")
        self.assertEqual(voice.status, "error")

    def test_cancel_before_capture_and_during_transcription_discards_audio_and_text(self):
        voice = LocalDictation()
        voice.cancel()
        sound, whisper = self.run_capture(voice)
        sound.InputStream.assert_not_called()
        whisper.WhisperModel.assert_not_called()
        self.assertEqual(voice.status, "idle")

        voice = LocalDictation()

        def late_result(*_args, **_kwargs):
            voice.cancel()
            return [SimpleNamespace(text="Late task")], None

        self.run_capture(voice, transcribe=late_result)
        self.assertEqual(voice.status, "idle")
        self.assertEqual(voice.task, "")

    def test_second_press_only_finishes_listening_and_live_worker_cannot_be_duplicated(self):
        voice = LocalDictation()
        voice.status = "preparing"
        voice.finish()
        self.assertFalse(voice.stop.is_set())
        voice.status = "listening"
        voice.finish()
        self.assertTrue(voice.stop.is_set())
        voice.thread = Mock()
        voice.thread.is_alive.return_value = True
        with patch("dictation.threading.Thread") as thread:
            voice.start()
        thread.assert_not_called()
        voice.thread.is_alive.return_value = False
        voice.status, voice.task = "done", "old"
        voice.cancel()
        self.assertEqual((voice.status, voice.task), ("idle", ""))


if __name__ == "__main__":
    unittest.main()
