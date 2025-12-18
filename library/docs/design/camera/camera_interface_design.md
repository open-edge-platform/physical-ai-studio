# Camera Interface Design

## Executive Summary

This document proposes a unified `Camera` interface for `getiaction`, built on top of the existing [FrameSource](https://github.com/ArendJanKramer/FrameSource) library. FrameSource provides solid low-level camera integrations across multiple hardware backends (webcams, RealSense, Basler, GenICam, etc.)—excellent foundational work by our team.

**The challenge**: FrameSource is a fork of a another repository and was developed without strict engineering standards. While the low-level implementations are functional, the codebase lacks:

- Consistent API design and documentation
- Production-level code quality (typing, testing, error handling)
- A user-friendly high-level interface

**Our goal**: Build a clean, production-ready camera abstraction layer on top of FrameSource that:

1. **Differentiates** from the original codebase with a well-designed API
2. **Elevates** to product-level quality (typed, tested, documented)
3. **Provides** an intuitive high-level interface for excellent UX/DX

This design retains FrameSource's low-level strengths while delivering the polish expected of a production library. This would, overall, be our unique contribution, and novel product within the Geti ecosystem.

---

## Overview

A unified `Camera` interface for frame acquisition from live cameras, video files, and image folders. Video files and image folders are recorded camera output—the data originally came from a camera, we're just replaying it. One abstraction covers all cases.

**Design Principles:**

- **Hparams-first**: Explicit constructor args with IDE autocomplete, plus `from_config()` for configs
- **Context manager**: Safe resource management with `with` statement
- **Single ABC**: One `Camera` interface for all sources

---

## Packaging Strategy

This camera interface is needed across the Geti ecosystem (`geti-action`, `geti-prompt`, `geti-inspect`, `geti-tune`) and by external users. We therefore have **two approaches**:

### Option A: Subpackage

We could start inside `getiaction` for rapid iteration:

```text
library/src/getiaction/cameras/
├── __init__.py         # Public API + aliases
├── base.py             # Camera ABC
├── async_mixin.py      # Background capture (optional)
├── webcam.py           # Webcam
├── realsense.py        # RealSense
├── basler.py           # Basler
├── genicam.py          # Genicam
├── ipcam.py            # IPCam
├── screen.py           # Screen
├── video.py            # VideoFile
├── folder.py           # ImageFolder
└── lerobot.py          # LeRobot
```

```python
from getiaction.cameras import Webcam, RealSense
```

### Option B: Standalone Package (In long term, this is preferred)

We could extract to a standalone package for ecosystem-wide reuse. FrameSource is already a standalone repo.
However, we want a unique identity separate from the original codebase, so we would create a new package, e.g., `geticam`:

```text
geticam/
├── src/geticam/
│   ├── __init__.py
│   ├── base.py
│   ├── webcam.py
│   ├── ...
├── pyproject.toml
└── README.md
```

```python
from geticam import Webcam, RealSense
```

We could start with **Option A** for speed. We could design the API to be extraction-friendly (no internal `getiaction` imports in camera code) so migration to **Option B** is seamless once the interface stabilizes.

See [Open Design Decisions](#1-package-vs-subpackage-critical-decision) for full trade-off analysis.

---

## Background: FrameSource

[FrameSource](https://github.com/ArendJanKramer/FrameSource) is an existing library that handles low-level camera integrations. It supports:

| FrameSource Class     | Hardware/Source                                                                   |
| --------------------- | --------------------------------------------------------------------------------- |
| `WebcamCapture`       | USB webcams (OpenCV backend)                                                      |
| `WebcamCaptureNokhwa` | USB webcams ([nokhwa](https://github.com/l1npengtul/nokhwa) backend, more stable) |
| `BaslerCapture`       | Basler industrial cameras                                                         |
| `GenicamCapture`      | Generic GenICam devices                                                           |
| `RealsenseCapture`    | Intel RealSense depth                                                             |
| `IPCameraCapture`     | RTSP/HTTP network cameras                                                         |
| `ScreenCapture`       | Desktop screen capture                                                            |
| `VideoFileCapture`    | Video file playback                                                               |
| `FolderCapture`       | Image sequence playback                                                           |

**What FrameSource does well**: Hardware abstraction, threading, buffer management

**What this design adds**:

- Consistent, typed API with IDE autocomplete
- Context manager pattern for safe resource management
- Config-driven instantiation (`from_config()`)
- Optional async capture mixin (opt-in, not required)
- Production-level error handling and documentation

---

## Class Hierarchy

```text
Camera (ABC)
├── Webcam              # Webcam, USB cameras
├── RealSense           # Intel depth cameras
├── Basler              # Industrial (pypylon)
├── Genicam             # Generic industrial (harvesters)
├── IPCam               # Network cameras (RTSP/HTTP)
├── Screen              # Desktop capture
├── VideoFile           # Recorded: video files
├── ImageFolder         # Recorded: image sequences
└── LeRobot             # Wrapper for LeRobot cameras
```

### Coverage vs. FrameSource

| FrameSource               | This Design   | Notes        |
| ------------------------- | ------------- | ------------ |
| `WebcamCapture`           | `Webcam`      | Yes          |
| `BaslerCapture`           | `Basler`      | Yes          |
| `GenicamCapture`          | `Genicam`     | Yes          |
| `RealsenseCapture`        | `RealSense`   | Yes          |
| `IPCameraCapture`         | `IPCam`       | Yes          |
| `ScreenCapture`           | `Screen`      | Yes          |
| `VideoFileCapture`        | `VideoFile`   | Yes          |
| `FolderCapture`           | `ImageFolder` | Yes          |
| `AudioSpectrogramCapture` | No            | Not a camera |

---

## Dependencies

OpenCV is a base dependency. Optional extras for specialized hardware:

```bash
pip install getiaction[realsense]  # Intel RealSense
pip install getiaction[basler]     # Basler (pypylon)
pip install getiaction[genicam]    # GenICam (harvesters)
pip install getiaction[cameras]    # All camera dependencies
```

---

## Core Interface

### Camera ABC

```python
from abc import ABC, abstractmethod
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Self, TypeVar

import numpy as np
from numpy.typing import NDArray


class ColorMode(str, Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"


class Camera(ABC):
    """Abstract interface for frame acquisition."""

    def __init__(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        self.width = width
        self.height = height
        self.color_mode = color_mode

    @classmethod
    def from_config(cls, config: str | Path | dict | Any) -> Self:
        """Create from YAML file, dict, dataclass, or Pydantic model."""
        import yaml

        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        elif hasattr(config, "model_dump"):
            config = config.model_dump()
        elif hasattr(config, "__dataclass_fields__"):
            config = asdict(config)

        return cls(**config)

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read(self) -> NDArray[np.uint8]: ...

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> NDArray[np.uint8]:
        try:
            return self.read()
        except RuntimeError as e:
            if "end of" in str(e).lower():
                raise StopIteration from e
            raise
```

### Async Capture Mixin (Optional)

**This is optional, not required.** Cameras work perfectly with synchronous `read()` only.

For live cameras that _choose_ to support background capture, a mixin provides:

```python
class AsyncCaptureMixin:
    """Mixin for background capture. Subclass must implement _capture_frame()."""

    def start_async(self) -> None: ...
    def stop_async(self) -> None: ...
    def async_read(self, timeout_ms: float = 200.0) -> NDArray[np.uint8]: ...
```

---

## Proposed Implementations

The following subclasses implement the `Camera` ABC. Details are illustrative—final implementations will be determined after interface agreement.

### Live Cameras

```python
class Webcam(Camera):
    """USB cameras, built-in webcams, V4L2 devices.

    Backend: Can use OpenCV or nokhwa (via omnicamera). nokhwa provides
    better stability for USB cameras on some platforms.
    """

    def __init__(
        self, *,
        index: int = 0,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
        rotation: Rotation = Rotation.NONE,
        warmup_s: float = 1.0,
        backend: str = "opencv",  # or "nokhwa"
    ) -> None: ...


class RealSense(Camera):
    """Intel RealSense with optional depth stream."""

    def __init__(
        self, *,
        serial_number: str | None = None,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        color_mode: ColorMode = ColorMode.RGB,
        use_depth: bool = False,
    ) -> None: ...

    def read_depth(self) -> NDArray[np.uint16]: ...


class Basler(Camera):
    """Basler industrial cameras via pypylon."""

    def __init__(
        self, *,
        serial_number: str | None = None,
        fps: int = 30,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None: ...


class Genicam(Camera):
    """Generic GenICam devices via harvesters."""

    def __init__(
        self, *,
        cti_file: str | Path,
        device_id: int = 0,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None: ...


class IPCam(Camera):
    """Network cameras via RTSP/HTTP."""

    def __init__(
        self, *,
        url: str,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None: ...


class Screen(Camera):
    """Desktop screen capture."""

    def __init__(
        self, *,
        monitor: int = 0,
        region: tuple[int, int, int, int] | None = None,
        fps: int = 30,
    ) -> None: ...
```

### Recorded Sources

```python
class VideoFile(Camera):
    """Playback from video file."""

    def __init__(
        self, *,
        path: str | Path,
        loop: bool = False,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None: ...


class ImageFolder(Camera):
    """Playback from image sequence."""

    def __init__(
        self, *,
        path: str | Path,
        pattern: str = "*.png",
        loop: bool = False,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None: ...
```

### Interop

```python
class LeRobot(Camera):
    """Wrapper for LeRobot camera instances."""

    def __init__(self, lerobot_camera) -> None: ...

    @classmethod
    def from_lerobot_config(cls, config) -> "LeRobot": ...
```

---

## Usage

### Basic

```python
from getiaction.cameras import Webcam, VideoFile, ImageFolder
# or from geticam import Webcam, VideoFile, ImageFolder

# Live camera
with Webcam(index=0, fps=30, width=640, height=480) as camera:
    frame = camera.read()

# Video file
with VideoFile(path="recording.mp4") as video:
    for frame in video:
        process(frame)

# Image folder
with ImageFolder(path="dataset/images/") as folder:
    for frame in folder:
        process(frame)
```

### Async Capture (Optional)

If async capture is needed (e.g., for robotics with tight control loops), use the mixin:

```python
class AsyncWebcam(AsyncCaptureMixin, Webcam):
    """Webcam with background capture support."""
    ...

# Usage
camera = AsyncWebcam(index=0)
with camera:
    camera.start_async()
    while running:
        frame = camera.async_read()  # Latest frame, non-blocking
```

### Multi-Camera (Sync)

```python
cameras = {
    "wrist": Webcam(index=0),
    "overhead": RealSense(serial_number="012345"),
}

for cam in cameras.values():
    cam.connect()

# Synchronous reads
frames = {name: cam.read() for name, cam in cameras.items()}
```

### From Config

```python
# From YAML
camera = Webcam.from_config("camera.yaml")

# From dict
camera = Webcam.from_config({"index": 0, "fps": 30})

# From dataclass/Pydantic
camera = Webcam.from_config(my_config)
```

### Robot Integration (Sync)

```python
from getiaction.cameras import RealSense
from getiaction.robots import SO101
from getiaction.inference import InferenceModel

policy = InferenceModel.load("./exports/act_policy")
robot = SO101.from_config("robot.yaml")
camera = RealSense(fps=30)

with robot, camera:
    while True:
        action = policy.select_action({
            "images": {"wrist": camera.read()},
            "state": robot.get_state(),
        })
        robot.send_action(action)
```

---

## Comparison: FrameSource vs. getiaction.cameras/geticam

| Aspect              | FrameSource               | getiaction.cameras / geticam      |
| ------------------- | ------------------------- | --------------------------------- |
| Instantiation       | Factory with string types | Hparams-first constructors        |
| Configuration       | Kwargs dict               | Explicit params + `from_config()` |
| Async threading     | Not implemented           | Optional via `AsyncCaptureMixin`  |
| Resource management | Manual                    | Context manager (`with`)          |
| Error handling      | `(success, frame)` tuples | Exceptions                        |
| Dependencies        | All bundled               | Optional per camera type          |

---

## Open Design Decisions

### 1. Package vs. Subpackage (Critical Decision)

**Context**: This camera interface will be needed across the Geti ecosystem, not just `getiaction`. Our product portfolio includes:

| Product        | Purpose                                    | Needs Camera? |
| -------------- | ------------------------------------------ | ------------- |
| `geti-action`  | Vision-language-action policies (robotics) | Yes           |
| `geti-prompt`  | Prompt-based tasks (SAM3, etc.)            | Yes           |
| `geti-inspect` | Anomaly detection                          | Yes           |
| `geti-tune`    | Classification, detection, segmentation    | Yes           |
| External users | Third-party integrations                   | Yes           |

**Options**:

| Option                    | Package Name        | Import                               | Pros                                      | Cons                                         |
| ------------------------- | ------------------- | ------------------------------------ | ----------------------------------------- | -------------------------------------------- |
| **A: Subpackage**         | (inside getiaction) | `from getiaction.cameras import ...` | Fast to implement, no new repo            | Tight coupling, can't use elsewhere          |
| **B: Standalone package** | `geticam` (new)     | `from geticam import ...`            | Reusable across ecosystem, clean branding | New repo, more maintenance                   |
| **C: Fork + refactor**    | Keep `framesource`  | `from framesource import ...`        | Minimal effort                            | No differentiation, legacy baggage, not ours |

**Branding consideration**: We need a unique identity separate from the original FrameSource codebase.

| Name           | Import                         | Verdict                                             |
| -------------- | ------------------------------ | --------------------------------------------------- |
| **`geticam`**  | `from geticam import ...`      | Recommended — short, memorable, clear purpose       |
| `geti-camera`  | `from geti_camera import ...`  | Good — matches `geti-action` convention, but longer |
| `geti-capture` | `from geti_capture import ...` | Broader scope, could imply screen recording         |

**Recommendation**: We could start with **Option A** (subpackage in `getiaction.cameras`) for rapid development. We could design the API to be extraction-friendly so we can move to **Option B** later if cross-product usage is confirmed.

**Team alignment needed**: This decision affects repo structure, CI/CD, and versioning strategy.

### 2. LeRobot Interop

| Option            | Location                        | Recommendation       |
| ----------------- | ------------------------------- | -------------------- |
| In cameras        | `getiaction/cameras/lerobot.py` | All cameras together |
| In lerobot module | `getiaction/lerobot/cameras.py` | Groups interop code  |

### 3. Frame Transforms

**Question**: Do we need a transforms system, or are built-in hparams enough?

The existing FrameSource has `FrameProcessor` classes:

- `RealsenseDepthProcessor` - depth colorization
- `Equirectangular360Processor` - 360° dewarp
- `HyperspectralProcessor` - band selection

**Options**:

| Option                    | Approach                     | Example                             |
| ------------------------- | ---------------------------- | ----------------------------------- |
| **A: Built-in only**      | Transforms via hparams       | `Webcam(width=640, color_mode=RGB)` |
| **B: Callable hook**      | Accept postprocess function  | `Webcam(postprocess=fn)`            |
| **C: Transform pipeline** | torchvision-style (no torch) | `Compose([Resize(640), ToRGB()])`   |

**Recommendation**: Start with **A** (built-in hparams). Add **B** (callable) if needed. Defer **C** unless clear demand.

Specialized processors (360°, depth colorization) would be separate utilities, not part of Camera.

### 4. Additional Opens

1. **Calibration data**: Add optional `calibration` property?
2. **Recording**: Add `CameraRecorder` for saving frames?
3. **Discovery**: Add `Camera.find_all()` for listing available devices?

---

## References

- [Robot Interface Design](robot_interface_design.md)
- [LeRobot Cameras](https://github.com/huggingface/lerobot/tree/main/src/lerobot/cameras)
