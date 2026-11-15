"""Display-only, frame-paired detector evidence for Studio; never authorizes motion."""

import base64
import copy
import io
import math
import threading
import time
from collections import OrderedDict

from PIL import Image

MAX_INSPECTION_BODY = 2_000_000
_init_lock = threading.Lock()


def _image(value):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 800_000:
        raise ValueError("Inspection image must be bounded base64 JPEG")
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG" or max(image.size) > 2048:
                raise ValueError("Require JPEG no larger than 2048 pixels")
            image.verify()
    except Exception as exc:
        raise ValueError("Invalid inspection JPEG") from exc
    return value


class InspectionEvidence:
    """Retain latest per-camera evidence with immutable acquisition freshness."""

    def __init__(self):
        self.lock = threading.Lock()
        self.records = {}
        self.seen_frames = OrderedDict()

    def publish(self, body, now=None):
        now = time.time() if now is None else now
        if not isinstance(body, dict):
            raise ValueError("Inspection evidence must be an object")
        camera = body.get("camera")
        if camera not in ("plate", "wrist"):
            raise ValueError("Only plate and wrist inspection are supported")
        for key in ("frame_id", "serial", "model_version", "reason"):
            if not isinstance(body.get(key), str) or not 0 < len(body[key]) <= 500:
                raise ValueError(f"Require bounded {key}")
        captured = body.get("captured_at")
        if type(captured) not in (int, float) or not math.isfinite(captured):
            raise ValueError("Require finite acquisition time")
        if not 0 <= now - captured <= 3:
            raise ValueError("Inspection frame is stale or from the future")
        if body.get("synthetic") is not False:
            raise ValueError("Require explicitly real evidence")
        decision = body.get("decision")
        if decision not in ("UNCALIBRATED", "REINSPECT", "ERROR", "GOOD", "DEFECTIVE"):
            raise ValueError("Invalid inspection decision")
        if decision in ("GOOD", "DEFECTIVE") and not body.get("threshold_version"):
            raise ValueError("Calibrated decisions require threshold provenance")
        distance = body.get("raw_distance")
        if distance is not None and (
            type(distance) not in (int, float) or not math.isfinite(distance) or distance < 0
        ):
            raise ValueError("Raw distance must be finite and nonnegative")
        if decision in ("ERROR", "REINSPECT") and distance is not None:
            raise ValueError("Invalid views cannot publish a trusted distance")
        original = _image(body.get("image_jpeg"))
        overlay = _image(body.get("overlay_jpeg"))
        if overlay and (not original or decision in ("ERROR", "REINSPECT")):
            raise ValueError("Heatmap requires a valid paired source image")
        record = {k: body[k] for k in ("camera", "frame_id", "serial", "model_version", "reason")}
        record.update(
            captured_at=captured,
            decision=decision,
            raw_distance=distance,
            threshold_version=body.get("threshold_version"),
            synthetic=False,
            image_jpeg=original,
            overlay_jpeg=overlay,
            automatic_sorting_allowed=False,
        )
        with self.lock:
            key = (camera, body["serial"], body["frame_id"])
            if key in self.seen_frames:
                raise ValueError("Reject repeated inspection frame")
            prior = self.records.get(camera)
            if prior and (
                captured <= prior["captured_at"] or body["frame_id"] == prior["frame_id"]
            ):
                raise ValueError("Reject repeated or out-of-order inspection frame")
            self.records[camera] = record
            self.seen_frames[key] = captured
            while len(self.seen_frames) > 256:
                self.seen_frames.popitem(last=False)
        return {"ok": True, "camera": camera, "frame_id": record["frame_id"]}

    def snapshot(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            records = copy.deepcopy(self.records)
        for record in records.values():
            record["age_s"] = max(0, now - record["captured_at"])
            record["fresh"] = 0 <= now - record["captured_at"] <= 3
            if not record["fresh"]:
                record.update(
                    decision="STALE",
                    reason="Waiting for a fresh inspection frame",
                    raw_distance=None,
                    image_jpeg=None,
                    overlay_jpeg=None,
                )
        return {"cameras": records, "automatic_sorting_allowed": False}


def inspection_evidence(engine):
    with _init_lock:
        if not hasattr(engine, "_inspection_evidence"):
            engine._inspection_evidence = InspectionEvidence()
        return engine._inspection_evidence
