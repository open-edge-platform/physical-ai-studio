import logging
from fractions import Fraction

from pydantic import BaseModel, Field

from .dataset import Dataset
from .environment import EnvironmentWithRelations
from .model import Model


class StreamingEncodingSettings(BaseModel):
    streaming_encoding: bool = True
    vcodec: str = "auto"
    encoder_threads: int | None = None
    encoder_queue_maxsize: int = 60

    def with_resolved_vcodec(self) -> "StreamingEncodingSettings":
        # - If vcodec is already explicit (or streaming is disabled),
        #   _resolve_vcodec returns the original value.
        # - If vcodec is "auto", _resolve_vcodec probes candidates and picks
        #   the first usable encoder.
        return self.model_copy(update={"vcodec": self._resolve_vcodec()})

    def _resolve_vcodec(self) -> str:
        if not self.streaming_encoding or self.vcodec != "auto":
            return self.vcodec

        for candidate in self._vcodec_candidates():
            if self._is_vcodec_usable(candidate):
                logging.info(f"Auto-selected vcodec '{candidate}'")
                return candidate

        raise RuntimeError("No usable video encoder found for streaming encoding")

    @staticmethod
    def _vcodec_candidates() -> list[str]:
        return [
            "h264_videotoolbox",
            "hevc_videotoolbox",
            "h264_nvenc",
            "hevc_nvenc",
            "h264_vaapi",
            "h264_qsv",
            "libsvtav1",
            "libx264",
            "h264",
        ]

    @staticmethod
    def _is_vcodec_usable(vcodec: str) -> bool:
        try:
            import av

            encoder = av.CodecContext.create(vcodec, "w")
            setattr(encoder, "width", 320)
            setattr(encoder, "height", 240)
            setattr(encoder, "framerate", Fraction(30, 1))
            setattr(encoder, "time_base", Fraction(1, 30))
            setattr(encoder, "pix_fmt", "yuv420p")
            encoder.open()
            return True
        except Exception as exc:
            logging.warning(f"Skipping unavailable vcodec '{vcodec}': {exc}")
            return False


class TeleoperationConfig(BaseModel):
    task: str
    dataset: Dataset
    environment: EnvironmentWithRelations
    streaming_encoding_settings: StreamingEncodingSettings = Field(default_factory=StreamingEncodingSettings)


class InferenceConfig(BaseModel):
    model: Model
    task_index: int
    environment: EnvironmentWithRelations
    backend: str
