import logging
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

from lerobot.configs import RGBEncoderConfig
from lerobot.configs.video import VIDEO_CODECS_ALIASES
from pydantic import BaseModel, Field

_HW_CODEC_PIX_FMTS = dict.fromkeys(
    (
        "av1_qsv",
        "hevc_qsv",
        "h264_qsv",
        "av1_nvenc",
        "hevc_nvenc",
        "h264_nvenc",
        "av1_vaapi",
        "hevc_vaapi",
        "h264_vaapi",
        "h264_videotoolbox",
        "hevc_videotoolbox",
    ),
    "nv12",
)

_VCODEC_CANDIDATES = (
    "av1_qsv",
    "hevc_qsv",
    "h264_qsv",
    "av1_nvenc",
    "hevc_nvenc",
    "h264_nvenc",
    "av1_vaapi",
    "hevc_vaapi",
    "h264_vaapi",
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "libsvtav1",
    "libaom-av1",
    "libx265",
    "libx264",
)


def vcodec_candidates() -> list[str]:
    """Return auto-selection candidates in preference order."""
    return list(_VCODEC_CANDIDATES)


@cache
def _is_vcodec_usable(vcodec: str) -> bool:
    """Probe a codec by encoding and muxing one frame."""
    return _encode_probe(vcodec, _HW_CODEC_PIX_FMTS.get(vcodec, "yuv420p"), {"g": "2"})


def _encode_probe(vcodec: str, pix_fmt: str, options: dict[str, str]) -> bool:
    """Return whether one exact encoder configuration can write a frame."""
    import av
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as probe_file:
        probe_path = Path(probe_file.name)
    try:
        with av.open(str(probe_path), "w") as container:
            stream = container.add_stream(vcodec, 30, options=options)
            if not isinstance(stream, av.VideoStream):
                raise ValueError(f"Codec {vcodec!r} did not create a video stream")
            stream.pix_fmt = pix_fmt
            stream.width, stream.height, stream.time_base = 320, 240, Fraction(1, 30)
            frame = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = 0, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return True
    except Exception as exc:
        logging.warning("Skipping unavailable video encoder '%s': %s", vcodec, exc)
        return False
    finally:
        probe_path.unlink(missing_ok=True)


class StudioRGBEncoderConfig(RGBEncoderConfig):
    """RGB encoder config with Studio's real encode-probe resolution.

    ``lerobot`` ships no type information, so the inherited dataclass fields
    are re-declared here to keep ``mypy`` able to resolve their types.
    """

    vcodec: str = "libsvtav1"
    pix_fmt: str = "yuv420p"
    g: int | None = 2
    crf: int | float | None = 30
    preset: int | str | None = None
    fast_decode: int = 0
    video_backend: str = "pyav"
    extra_options: dict[str, Any] = field(default_factory=dict)

    def resolve_vcodec(self) -> None:
        self.vcodec = VIDEO_CODECS_ALIASES.get(self.vcodec, self.vcodec)
        if self.vcodec != "auto":
            if _is_vcodec_usable(self.vcodec):
                self._apply_default_pix_fmt()
                return
            raise RuntimeError(f"Video codec '{self.vcodec}' is unavailable for encoding")
        for candidate in vcodec_candidates():
            if _candidate_is_viable(candidate):
                if candidate != self.vcodec:
                    self._reset_codec_specific_options(candidate)
                self.vcodec = candidate
                self._apply_default_pix_fmt()
                logging.info("Auto-selected vcodec '%s'", candidate)
                return
        raise RuntimeError("No usable video encoder found for video recording")

    def _reset_codec_specific_options(self, fallback_vcodec: str) -> None:
        """Discard options tied to an unavailable requested encoder.

        Hardware encoders commonly need ``nv12`` and accept string presets such
        as ``veryfast``.  Those values are not necessarily valid for a software
        fallback (for example, libsvtav1 requires yuv420p and a numeric preset).
        Keep codec-independent GOP and quality settings, but allow LeRobot's
        defaults to select a compatible pixel format and preset.
        """
        logging.warning(
            "Falling back from vcodec '%s' to '%s'; resetting pixel format, preset, and extra codec options",
            self.vcodec,
            fallback_vcodec,
        )
        self.pix_fmt = "yuv420p"
        self.preset = None
        self.extra_options = {}

    def _apply_default_pix_fmt(self) -> None:
        if self.pix_fmt == "yuv420p":
            self.pix_fmt = _HW_CODEC_PIX_FMTS.get(self.vcodec, self.pix_fmt)

    def get_codec_options(self, encoder_threads: int | None = None, as_strings: bool = False) -> dict[str, Any]:
        options = super().get_codec_options(encoder_threads, as_strings)
        if "preset" in options:
            preset = _normalise_preset(self.vcodec, options["preset"])
            # PyAV validates numeric AVOptions as numbers, but its encoder API
            # requires every option value to be a string.
            options["preset"] = str(preset) if as_strings else preset
        if self.vcodec.endswith("_qsv") and "crf" in options:
            options["global_quality"] = options.pop("crf")
        elif self.vcodec.endswith("_vaapi") and "crf" in options:
            options["qp"] = options.pop("crf")
        return options


def _candidate_is_viable(vcodec: str) -> bool:
    if not _is_vcodec_usable(vcodec):
        return False
    try:
        StudioRGBEncoderConfig(vcodec=vcodec)
        return True
    except Exception as exc:
        logging.warning("Skipping vcodec '%s': %s", vcodec, exc)
        return False


@cache
def _numeric_preset_choices(vcodec: str) -> dict[str, int | float]:
    """Return named preset values for encoders whose AVOption is numeric."""
    import av

    try:
        codec = av.codec.Codec(vcodec, "w")
    except Exception:
        return {}
    preset = next((option for option in codec.descriptor.options if option.name == "preset"), None)
    if preset is None or preset.type.name not in {"INT", "INT64", "UINT64", "FLOAT", "DOUBLE"}:
        return {}
    choices: Any = getattr(preset, "choices", ())
    return {
        choice.name: choice.value
        for choice in choices or []
        if isinstance(choice.value, int | float) and not isinstance(choice.value, bool)
    }


def _normalise_preset(vcodec: str, preset: Any) -> Any:
    """Convert a named FFmpeg preset to its numeric AVOption value when needed."""
    if not isinstance(preset, str):
        return preset
    return _numeric_preset_choices(vcodec).get(preset, preset)


class VideoEncodingSettings(BaseModel):
    streaming_encoding: bool = True
    vcodec: str = "auto"
    pix_fmt: str | None = None
    g: int | None = 2
    crf: int | float | None = None
    preset: int | str | None = None
    extra_options: dict[str, Any] = Field(default_factory=dict)
    encoder_threads: int | None = None
    encoder_queue_maxsize: int = 60

    def to_lerobot_write_kwargs(self) -> dict[str, Any]:
        return {
            "streaming_encoding": self.streaming_encoding,
            "encoder_threads": self.encoder_threads,
            "encoder_queue_maxsize": self.encoder_queue_maxsize,
            "rgb_encoder": self._build_rgb_encoder_config(),
        }

    def _build_rgb_encoder_config(self) -> StudioRGBEncoderConfig:
        params: dict[str, Any] = {"vcodec": self.vcodec}
        for name in ("pix_fmt", "g", "crf", "preset"):
            value = getattr(self, name)
            if value is not None:
                params[name] = value
        if self.extra_options:
            params["extra_options"] = self.extra_options
        return StudioRGBEncoderConfig(**params)


@dataclass(frozen=True)
class VideoEncodingPreset:
    """A Studio-owned, user-selectable recording configuration."""

    id: str
    name: str
    description: str
    settings: VideoEncodingSettings


_VIDEO_ENCODING_PRESETS = (
    VideoEncodingPreset(
        id="intel-qsv-h264",
        name="Intel GPU QSV",
        description="Hardware H.264 encoding using Intel Quick Sync Video.",
        settings=VideoEncodingSettings(vcodec="h264_qsv", pix_fmt="nv12", crf=23, preset="veryfast"),
    ),
    VideoEncodingPreset(
        id="nvidia-nvenc-h264",
        name="NVIDIA GPU NVENC",
        description="Hardware H.264 encoding using NVIDIA NVENC.",
        settings=VideoEncodingSettings(vcodec="h264_nvenc", pix_fmt="nv12", crf=23, preset="p5"),
    ),
    VideoEncodingPreset(
        id="apple-videotoolbox-h264",
        name="Apple VideoToolbox H.264",
        description="Hardware H.264 encoding using Apple VideoToolbox.",
        settings=VideoEncodingSettings(vcodec="h264_videotoolbox", crf=23),
    ),
    VideoEncodingPreset(
        id="apple-videotoolbox-hevc",
        name="Apple VideoToolbox HEVC",
        description="Hardware HEVC encoding using Apple VideoToolbox.",
        settings=VideoEncodingSettings(vcodec="hevc_videotoolbox", crf=28),
    ),
    VideoEncodingPreset(
        id="cpu-h264",
        name="CPU H.264",
        description="Software H.264 encoding with broad player compatibility.",
        settings=VideoEncodingSettings(vcodec="libx264", crf=23, preset="veryfast"),
    ),
    VideoEncodingPreset(
        id="cpu-av1",
        name="CPU AV1",
        description="Software AV1 encoding with efficient compression.",
        settings=VideoEncodingSettings(vcodec="libsvtav1", crf=30),
    ),
)


def video_encoding_presets() -> tuple[VideoEncodingPreset, ...]:
    """Return all Studio-owned recording presets in display order."""
    return _VIDEO_ENCODING_PRESETS


def get_video_encoding_preset(preset_id: str) -> VideoEncodingPreset | None:
    """Return a preset by stable ID."""
    return next((preset for preset in _VIDEO_ENCODING_PRESETS if preset.id == preset_id), None)


def video_encoding_preset_is_usable(preset: VideoEncodingPreset) -> bool:
    """Return whether a preset's requested encoder and options work here."""
    try:
        config = preset.settings._build_rgb_encoder_config()
        return _encode_probe(config.vcodec, config.pix_fmt, config.get_codec_options(as_strings=True))
    except Exception as exc:
        logging.warning("Video recording preset '%s' is unavailable: %s", preset.id, exc)
        return False


@cache
def available_video_encoding_presets() -> tuple[VideoEncodingPreset, ...]:
    """Discover usable recording presets once for the lifetime of this process."""
    return tuple(preset for preset in video_encoding_presets() if video_encoding_preset_is_usable(preset))


def resolve_video_encoding_settings(preset_id: str | None) -> VideoEncodingSettings:
    """Resolve Default or a verified explicit preset for a recording writer.

    Default retains Studio's historic ``auto`` behavior: probe the ordered
    encoder candidates and use the first codec that can encode a real frame.
    """
    if preset_id is None:
        return VideoEncodingSettings()
    preset = get_video_encoding_preset(preset_id)
    if preset is None:
        raise ValueError(f"Unknown video recording preset '{preset_id}'")
    if not video_encoding_preset_is_usable(preset):
        raise ValueError(f"Video recording preset '{preset.name}' is unavailable on this backend")
    return preset.settings.model_copy(deep=True)
