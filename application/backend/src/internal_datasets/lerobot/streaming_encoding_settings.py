import logging
from fractions import Fraction
from functools import cache
from typing import Any

from lerobot.configs import RGBEncoderConfig
from pydantic import BaseModel


class StreamingEncodingSettings(BaseModel):
    streaming_encoding: bool = True
    vcodec: str = "auto"
    encoder_threads: int | None = None
    encoder_queue_maxsize: int = 60

    def with_resolved_vcodec(self) -> "StreamingEncodingSettings":
        # - If vcodec is already explicit (or streaming is disabled),
        #   resolve_vcodec returns the original value.
        # - If vcodec is "auto", resolve_vcodec probes candidates once and
        #   picks the first usable encoder.
        return self.model_copy(update={"vcodec": _resolve_vcodec(self.vcodec, self.streaming_encoding)})

    def to_lerobot_write_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "streaming_encoding": self.streaming_encoding,
            "encoder_threads": self.encoder_threads,
            "encoder_queue_maxsize": self.encoder_queue_maxsize,
        }
        if self.vcodec != "auto":
            kwargs["rgb_encoder"] = RGBEncoderConfig(vcodec=self.vcodec, g=None)
        return kwargs

    @staticmethod
    def _vcodec_candidates() -> list[str]:
        return [
            "h264_videotoolbox",  # macOS
            "hevc_videotoolbox",  # macOS
            "av1_qsv",  # Intel QSV
            "h264_qsv",  # Intel QSV
            "h264_nvenc",  # NVIDIA NVENC
            "hevc_nvenc",  # NVIDIA NVENC
            "h264_vaapi",  # Intel/AMD VA-API
            "libsvtav1",  # SVT-AV1, open source SW AV1
            "libx264",  # open source SW H.264
            "h264",  # Proprietary H.264
        ]

    @staticmethod
    def _is_vcodec_usable(vcodec: str) -> bool:
        try:
            import av

            encoder = av.CodecContext.create(vcodec, "w")
            pix_fmt = "nv12" if vcodec.endswith("qsv") else "yuv420p"
            setattr(encoder, "width", 320)
            setattr(encoder, "height", 240)
            setattr(encoder, "framerate", Fraction(30, 1))
            setattr(encoder, "time_base", Fraction(1, 30))
            setattr(encoder, "pix_fmt", pix_fmt)
            encoder.open()
            return True
        except Exception as exc:
            logging.warning(f"Skipping unavailable vcodec '{vcodec}': {exc}")
            return False


@cache
def _resolve_vcodec(vcodec: str, streaming_encoding: bool) -> str:
    """Probe usable vcodec once per process; result is cached."""
    if not streaming_encoding or vcodec != "auto":
        return vcodec

    for candidate in StreamingEncodingSettings._vcodec_candidates():
        if StreamingEncodingSettings._is_vcodec_usable(candidate):
            logging.info(f"Auto-selected vcodec '{candidate}'")
            return candidate

    raise RuntimeError("No usable video encoder found for streaming encoding")
