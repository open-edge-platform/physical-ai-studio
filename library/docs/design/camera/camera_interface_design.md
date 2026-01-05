# Camera Interface Design

## Table of Contents

- [Camera Interface Design](#camera-interface-design)
  - [Table of Contents](#table-of-contents)
  - [Executive Summary](#executive-summary)
  - [Overview](#overview)
  - [Packaging Strategy](#packaging-strategy)
    - [Option A: Subpackage](#option-a-subpackage)
    - [Option B: Standalone Package (In long term, this is preferred)](#option-b-standalone-package-in-long-term-this-is-preferred)
  - [Background: FrameSource](#background-framesource)
  - [Class Hierarchy](#class-hierarchy)
    - [Coverage vs. FrameSource](#coverage-vs-framesource)
  - [Dependencies](#dependencies)
  - [Multi-Consumer \& Lifecycle Management](#multi-consumer--lifecycle-management)
    - [Problem Statement](#problem-statement)
    - [Design Choice: Invisible Sharing](#design-choice-invisible-sharing)
    - [Alternative Considered: CameraManager](#alternative-considered-cameramanager)
    - [Other Alternatives](#other-alternatives)
  - [Core Interface](#core-interface)
    - [Camera ABC](#camera-abc)
    - [Callback System (PyTorch Lightning-inspired)](#callback-system-pytorch-lightning-inspired)
    - [Capability Mixins](#capability-mixins)
  - [Proposed Implementations](#proposed-implementations)
    - [Live Cameras](#live-cameras)
    - [Recorded Sources](#recorded-sources)
    - [Interop (Outside cameras package)](#interop-outside-cameras-package)
  - [Usage](#usage)
    - [Basic](#basic)
    - [Push-Based Frame Delivery (Callbacks)](#push-based-frame-delivery-callbacks)
    - [Multi-Camera Setup](#multi-camera-setup)
    - [Multi-Consumer (Automatic Sharing)](#multi-consumer-automatic-sharing)
    - [From Config](#from-config)
    - [Robot Integration](#robot-integration)
    - [IPCam with PTZ Control](#ipcam-with-ptz-control)
  - [Comparison: FrameSource vs. getiaction.cameras/geticam](#comparison-framesource-vs-getiactioncamerasgeticam)
  - [Open Design Decisions](#open-design-decisions)
    - [1. Package vs. Subpackage (Critical Decision)](#1-package-vs-subpackage-critical-decision)
    - [2. Frame Transforms](#2-frame-transforms)
    - [3. Additional Opens](#3-additional-opens)
  - [References](#references)

---

## Executive Summary

This document proposes a unified `Camera` interface for `getiaction`, built on top of the existing [FrameSource](https://github.com/ArendJanKramer/FrameSource) library. FrameSource provides solid low-level camera integrations across multiple hardware backends (webcams, RealSense, Basler, GenICam, etc.)—excellent foundational work by our team.

**The challenge**: FrameSource is a fork of a another repository and was developed without strict engineering standards. While the low-level implementations are functional, the codebase has limitations:

- Consistent API design and documentation
- Production-level code quality (typing, testing, error handling)
- A user-friendly high-level interface
- PyPI compatibility: FrameSource has GitHub-based dependencies that could prevent PyPI publication—it cannot be installed via `pip install framesource`

**Our goal**: Build a clean, production-ready camera abstraction layer that:

1. **Differentiates** from the original codebase with a well-designed API
2. **Elevates** to product-level quality (typed, tested, documented)
3. **Provides** an intuitive high-level interface for excellent UX/DX
4. **Is pip-installable** — deployable as a standard Python package on PyPI

This design retains FrameSource's low-level strengths while delivering the polish expected of a production library. This would, overall, be our unique contribution, and novel product within the Geti ecosystem.

---

## Overview

A unified `Camera` interface for frame acquisition from live cameras, video files, and image folders. Video files and image folders are recorded camera output—the data originally came from a camera, we're just replaying it. One abstraction covers all cases.

**Design Principles:**

- **Hparams-first**: Explicit constructor args with IDE autocomplete, plus `from_config()` for configs
- **Context manager**: Safe resource management with `with` statement
- **Single ABC**: One `Camera` interface for all sources
- **Invisible sharing**: Multiple Camera instances for the same device share automatically
- **Callback-driven**: PyTorch Lightning-style callbacks for reactive/event-driven use cases
- **Capability mixins**: Optional features (Async, PTZ, color control, resolution discovery) via composable mixins

---

## Packaging Strategy

This camera interface is needed across the Geti ecosystem (`geti-action`, `geti-prompt`, `geti-inspect`, `geti-tune`) and by external users. We therefore have **two approaches**:

### Option A: Subpackage

We could start inside `getiaction` for rapid iteration:

```text
library/src/getiaction/cameras/
├── __init__.py         # Public API + aliases
├── base.py             # Camera ABC, _Capture, ColorMode
├── callbacks.py        # Callback base class and hooks
├── mixins.py           # PTZMixin, ColorControlMixin, ResolutionDiscoveryMixin
├── webcam.py           # Webcam (with nokhwa/opencv backends)
├── realsense.py        # RealSense
├── basler.py           # Basler
├── genicam.py          # Genicam
├── ipcam.py            # IPCam (with PTZMixin)
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
- Invisible sharing: multiple Camera instances share the same physical device automatically
- Callback system for push-based frame delivery
- Capability mixins for optional features (PTZ, color control, resolution discovery)
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
└── ImageFolder         # Recorded: image sequences
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

## Multi-Consumer & Lifecycle Management

### Problem Statement

Real-world camera usage involves multiple consumers accessing the same physical device:

- **UI display** needs camera feed for live preview
- **Recording** needs same camera feed to save to disk
- **Teleoperation** needs feed for remote control
- **WebSocket/WebRTC** streams need feed for network transmission

**The challenge**: A physical camera can only be opened once. Multiple opens fail or cause undefined behavior.

**Requirements**:

| Requirement                    | Description                                        |
| ------------------------------ | -------------------------------------------------- |
| **R1: Single device access**   | Only one process opens the physical camera         |
| **R2: Multi-consumer support** | Multiple code paths read from same device          |
| **R3: Automatic lifecycle**    | Device opens on first use, closes when unused      |
| **R4: Simple API**             | Basic usage should remain simple                   |
| **R5: Thread safety**          | Safe access from multiple threads                  |
| **R6: Camera as main concept** | Avoid introducing many new abstractions            |
| **R7: Extensibility**          | Support callbacks and capability mixins            |
| **R8: Push & Pull delivery**   | Both polling (`read()`) and callbacks (`on_frame`) |

### Design Choice: Invisible Sharing

**Concept**: `Camera` remains the only public concept. Sharing happens automatically based on device identity. Internal reference counting manages lifecycle.

```python
# Simple usage
with Webcam(index=0) as cam:
    frame = cam.read()

# Multi-consumer - automatic sharing
cam1 = Webcam(index=0)
cam2 = Webcam(index=0)  # Same device

cam1.connect()  # Opens device, ref_count = 1
cam2.connect()  # Reuses capture, ref_count = 2

frame1 = cam1.read()  # Both get frames
frame2 = cam2.read()  # from same source

cam1.disconnect()  # ref_count = 1, device stays open
cam2.disconnect()  # ref_count = 0, device closes
```

**How it works**:

```text
┌─────────────────────────────────────────────┐
│              User Code                      │
│  cam1 = Webcam(0)     cam2 = Webcam(0)      │
│        ↑                    ↑               │
│   cam1.read()          callback.on_frame()  │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│           Camera Interface                  │
│  - Clean public API                         │
│  - Delegates to internal shared state       │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│     Internal: _Capture (ref counted)        │
│  "webcam:0" → _Capture                      │
│     └─ ref_count: 2                         │
│     └─ callbacks: [Logging, Recording]      │
│     └─ Background capture thread            │
│     └─ Invokes on_frame() for each frame    │
└─────────────────────────────────────────────┘
```

**Why this approach?**

The best abstractions **hide complexity**, not expose it. Similar patterns:

- `logging.getLogger("app")` — Multiple calls return same logger
- Python reference counting — Automatic, invisible
- ORM connection pooling — You just query, pool is hidden

| Pros                                | Cons                                |
| ----------------------------------- | ----------------------------------- |
| Camera stays the only concept       | Implicit sharing may surprise users |
| Zero API change for consumers       | Hidden global state                 |
| Automatic resource management       | Testing requires care               |
| Simple single-camera case unchanged | Need clear docs on sharing behavior |
| Supports callbacks for push-based   |                                     |

### Alternative Considered: CameraManager

A separate manager object could handle sharing explicitly:

```python
manager = CameraManager()
cam1 = manager.get_camera("webcam:0", fps=30)
cam2 = manager.get_camera("webcam:0", fps=30)  # Shared
```

| Pros                           | Cons                            |
| ------------------------------ | ------------------------------- |
| Explicit resource coordination | Requires managing the manager   |
| Clear ownership model          | Extra object to pass around     |
| Easy to test                   | Less ergonomic for simple cases |

**Not recommended** because it introduces a second concept users must learn.

### Other Alternatives

**Camera + Explicit Feed**: Separate "device ownership" from "frame reading" via a `Feed` object. May not be ideal because callbacks (`on_frame`) provide the same push-based delivery more elegantly.

**Broadcast Mode**: Add `start_broadcast()` / `subscribe()` methods. May not be ideal because two modes of operation is confusing, and `_Capture` already runs a background thread.

---

## Core Interface

### Camera ABC

The `Camera` ABC defines the interface all camera implementations must follow.

**Key design elements**:

- **Abstract methods** (`device_key`, `_connect_device`, `_disconnect_device`, `_read_frame`) — Subclasses implement these
- **Public API** (`connect`, `disconnect`, `read`, `add_callback`) — Users call these
- **Invisible sharing** — Handled automatically via class-level `_captures` dict
- **Capability flags** — `supports_ptz`, `supports_color_control`, etc. (set by mixins)

```python
class ColorMode(str, Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"


class Camera(ABC):
    """Abstract interface for frame acquisition with automatic sharing."""

    def __init__(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
        callbacks: list[Callback] | None = None,
    ) -> None: ...

    # === Abstract: Subclasses must implement ===

    @property
    @abstractmethod
    def device_key(self) -> str:
        """Unique identifier for the physical device (e.g., 'webcam:0')."""
        ...

    @abstractmethod
    def _connect_device(self) -> None:
        """Open the physical device (called only for first user)."""
        ...

    @abstractmethod
    def _disconnect_device(self) -> None:
        """Close the physical device (called only when last user disconnects)."""
        ...

    @abstractmethod
    def _read_frame(self) -> NDArray[np.uint8] | None:
        """Read a single frame from the device."""
        ...

    # === Public API ===

    @property
    def is_connected(self) -> bool: ...

    @property
    def is_shared(self) -> bool:
        """Whether other Camera instances are using this device."""
        ...

    def connect(self) -> None:
        """Connect to the device (shares if already open)."""
        ...

    def disconnect(self) -> None:
        """Disconnect (device closes only when last user disconnects)."""
        ...

    def read(self) -> NDArray[np.uint8]:
        """Read the latest frame (pull-based)."""
        ...

    def add_callback(self, callback: Callback) -> None:
        """Add a callback for push-based frame delivery."""
        ...

    def remove_callback(self, callback: Callback) -> None:
        """Remove a callback."""
        ...

    @classmethod
    def from_config(cls, config: str | Path | dict) -> Self:
        """Create from YAML file, dict, dataclass, or Pydantic model."""
        ...

    @classmethod
    def active_devices(cls) -> list[str]:
        """List device keys of all currently open captures."""
        ...

    # Context manager and iterator support
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> NDArray[np.uint8]: ...
```

### Callback System (PyTorch Lightning-inspired)

Callbacks provide push-based frame delivery and lifecycle hooks. Override only the hooks you need.

```python
class Callback:
    """Base callback class. Override hooks as needed."""

    def on_connect(self, camera: Camera) -> None:
        """Called after camera connects."""
        pass

    def on_disconnect(self, camera: Camera) -> None:
        """Called before camera disconnects."""
        pass

    def on_frame(self, camera: Camera, frame: NDArray[np.uint8]) -> None:
        """Called when a new frame is captured (push-based)."""
        pass

    def on_error(self, camera: Camera, error: Exception) -> None:
        """Called when a capture error occurs."""
        pass
```

**Example callbacks:**

```python
class LoggingCallback(Callback):
    def on_connect(self, camera):
        logger.info(f"Connected: {camera.device_key}")

    def on_frame(self, camera, frame):
        logger.debug(f"Frame: {frame.shape}")


class RecordingCallback(Callback):
    def __init__(self, path: Path):
        self.writer = VideoWriter(path)

    def on_frame(self, camera, frame):
        self.writer.write(frame)

    def on_disconnect(self, camera):
        self.writer.close()
```

### Capability Mixins

Optional capabilities are added via mixins. Each mixin sets a `ClassVar` flag automatically.

```python
class PTZMixin:
    """Adds pan-tilt-zoom controls."""
    supports_ptz: ClassVar[bool] = True

    @abstractmethod
    def pan(self, degrees: float) -> None: ...

    @abstractmethod
    def tilt(self, degrees: float) -> None: ...

    @abstractmethod
    def zoom(self, level: float) -> None: ...

    def move_to(self, pan: float, tilt: float, zoom: float) -> None:
        """Move to absolute position (default implementation)."""
        self.pan(pan)
        self.tilt(tilt)
        self.zoom(zoom)


class ColorControlMixin:
    """Adds camera color/exposure controls."""
    supports_color_control: ClassVar[bool] = True

    @abstractmethod
    def get_brightness(self) -> float: ...
    @abstractmethod
    def set_brightness(self, value: float) -> None: ...
    @abstractmethod
    def get_exposure(self) -> float: ...
    @abstractmethod
    def set_exposure(self, value: float) -> None: ...


class DepthMixin:
    """Adds depth capture capability.

    For cameras that support depth sensing (RealSense, stereo cameras,
    ToF sensors, etc.). Depth is returned as uint16 in millimeters.

    Example:
        cam = RealSense(serial_number="12345678")
        cam.connect()

        if cam.supports_depth:
            rgb = cam.read()
            depth = cam.read_depth()  # (H, W) uint16 in mm
            rgbd = cam.read_rgbd()    # (H, W, 4) with depth as 4th channel
    """
    supports_depth: ClassVar[bool] = True

    @abstractmethod
    def read_depth(self) -> NDArray[np.uint16]:
        """Read depth frame.

        Returns:
            Depth map as (H, W) uint16 array in millimeters.
        """
        ...

    def read_rgbd(self) -> NDArray[np.uint16]:
        """Read aligned RGB-D frame.

        Returns:
            RGBD as (H, W, 4) array. RGB channels are uint8, depth is uint16.
            Default implementation stacks read() and read_depth().
        """
        rgb = self.read()
        depth = self.read_depth()
        # Stack RGB (H,W,3) with depth (H,W,1)
        return np.dstack([rgb, depth[..., np.newaxis]])


class ResolutionDiscoveryMixin:
    """Adds format discovery and selection."""
    supports_resolution_discovery: ClassVar[bool] = True

    @abstractmethod
    def get_supported_formats(self) -> list[Format]: ...

    @abstractmethod
    def set_format(self, format: Format) -> None: ...


@dataclass
class Format:
    """Represents a supported camera format."""
    width: int
    height: int
    fps: float
    pixel_format: str = "RGB"


class AsyncContextMixin:
    """Adds async context manager support for async/await usage.

    Enables `async with` syntax for cameras in async code:

        async with Webcam(index=0) as cam:
            frame = cam.read()

    Note: This wraps synchronous connect/disconnect. For fully async I/O,
    subclasses can override with thread pool executors.
    """
    supports_async_context: ClassVar[bool] = True

    async def __aenter__(self) -> Self:
        """Async context entry - connects the camera."""
        self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context exit - disconnects the camera."""
        self.disconnect()
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


class RealSense(DepthMixin, Camera):
    """Intel RealSense with depth sensing capability.

    Inherits from DepthMixin to provide read_depth() and read_rgbd().
    """

    def __init__(
        self, *,
        serial_number: str | None = None,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None: ...

    # read_depth() inherited from DepthMixin, implemented here
    def read_depth(self) -> NDArray[np.uint16]:
        """Read depth frame from RealSense depth sensor."""
        ...


class StereoCamera(DepthMixin, Camera):
    """Stereo camera pair with computed depth.

    Example of another camera type using DepthMixin.
    Depth is computed from stereo disparity.
    """

    def __init__(
        self, *,
        left_index: int = 0,
        right_index: int = 1,
        baseline_mm: float = 60.0,
        # ... other stereo params
    ) -> None: ...

    def read_depth(self) -> NDArray[np.uint16]:
        """Compute depth from stereo disparity."""
        ...


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

### Interop (Outside cameras package)

LeRobot interop lives in `getiaction.cameras.lerobot`, not in the cameras package:

```python
# getiaction/lerobot/cameras.py
class LeRobotCamera(Camera):
    """Adapter wrapping LeRobot camera instances."""

    def __init__(self, lerobot_camera) -> None: ...

    @classmethod
    def from_lerobot_config(cls, config) -> "LeRobotCamera": ...
```

```python
from getiaction.lerobot import LeRobotCamera
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

### Push-Based Frame Delivery (Callbacks)

For reactive/event-driven systems, use callbacks instead of polling:

```python
class FrameProcessor(Callback):
    def on_frame(self, camera, frame):
        # Called automatically when new frame arrives
        result = model.predict(frame)
        display(result)

# Register callback
camera = Webcam(index=0, callbacks=[FrameProcessor()])
with camera:
    # Frames are pushed to callback automatically
    time.sleep(10)  # Just wait, frames are processed in background

# Or add callback later
camera = Webcam(index=0)
camera.add_callback(FrameProcessor())
camera.connect()
```

### Multi-Camera Setup

```python
cameras = {
    "wrist": Webcam(index=0),
    "overhead": RealSense(serial_number="012345"),
}

for cam in cameras.values():
    cam.connect()

# Pull-based: synchronous reads
frames = {name: cam.read() for name, cam in cameras.items()}

for cam in cameras.values():
    cam.disconnect()
```

### Multi-Consumer (Automatic Sharing)

Multiple Camera instances for the same device share automatically:

```python
# UI display thread
cam_ui = Webcam(index=0)
cam_ui.connect()

# Recording thread (same physical camera, shared automatically)
cam_record = Webcam(index=0)
cam_record.connect()

print(cam_ui.is_shared)  # True

# Both read from the same device
while running:
    display(cam_ui.read())
    save_to_disk(cam_record.read())

cam_ui.disconnect()     # Device stays open
cam_record.disconnect() # Device closes (last user)
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

### Robot Integration

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

### IPCam with PTZ Control

```python
class IPCam(Camera, PTZMixin):
    """Network camera with PTZ support."""
    ...

cam = IPCam(url="rtsp://192.168.1.100/stream")
with cam:
    print(cam.supports_ptz)  # True (from PTZMixin)

    cam.pan(45)   # Pan 45 degrees
    cam.tilt(-10) # Tilt down 10 degrees
    cam.zoom(2.0) # Zoom level 2x

    frame = cam.read()
```

---

## Comparison: FrameSource vs. getiaction.cameras/geticam

| Aspect              | FrameSource               | getiaction.cameras / geticam           |
| ------------------- | ------------------------- | -------------------------------------- |
| Instantiation       | Factory with string types | Hparams-first constructors             |
| Configuration       | Kwargs dict               | Explicit params + `from_config()`      |
| Multi-consumer      | Manual threading          | Invisible sharing (automatic)          |
| Frame delivery      | Pull only                 | Pull (`read()`) + push (callbacks)     |
| Resource management | Manual                    | Context manager (`with`) + ref-counted |
| Error handling      | `(success, frame)` tuples | Exceptions + `on_error` callback       |
| Capabilities        | All-or-nothing            | Composable mixins (PTZ, color, etc.)   |
| Dependencies        | All bundled               | Optional per camera type               |

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

### 2. Frame Transforms

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

**Recommendation**: We could start with **A** (built-in hparams). Add **B** (callable) if needed. Defer **C** unless clear demand.

Specialized processors (360°, depth colorization) would be separate utilities, not part of Camera.

### 3. Additional Opens

1. **Config conflicts**: What if `Webcam(index=0, fps=30)` and `Webcam(index=0, fps=60)` try to share? Options: first config wins, error on mismatch, or warn on mismatch.
2. **Frame freshness**: Should consumers get "latest frame" (may skip) or queue-based delivery (no skipping)?

---

## References

- [Robot Interface Design](../robot/robot_interface_design.md)
- [FrameSource Repository](https://github.com/ArendJanKramer/FrameSource)
- [LeRobot Cameras](https://github.com/huggingface/lerobot/tree/main/src/lerobot/cameras)
