# Camera Interface Design

## Table of Contents

- [Camera Interface Design](#camera-interface-design)
  - [Table of Contents](#table-of-contents)
  - [Executive Summary](#executive-summary)
  - [Overview](#overview)
  - [Packaging Strategy](#packaging-strategy)
  - [Architecture](#architecture)
    - [Class Hierarchy](#class-hierarchy)
    - [Package Structure](#package-structure)
    - [Backend Strategy](#backend-strategy)
  - [Dependencies](#dependencies)
  - [Core Interface](#core-interface)
    - [Frame](#frame)
    - [Sensor ABC](#sensor-abc)
    - [Camera ABC](#camera-abc)
    - [DeviceInfo](#deviceinfo)
    - [Capability Mixins](#capability-mixins)
  - [Read Semantics](#read-semantics)
  - [Proposed Implementations](#proposed-implementations)
    - [OpenCVCamera](#opencvcamera)
    - [RealSenseCamera](#realsensecamera)
    - [BaslerCamera](#baslercamera)
    - [GenicamCamera](#genicamcamera)
    - [IPCamera](#ipcamera)
  - [Recorded Sources (Future)](#recorded-sources-future)
  - [Factory Function](#factory-function)
  - [Sharing Model](#sharing-model)
  - [Usage](#usage)
    - [Basic](#basic)
    - [Multi-Camera Setup](#multi-camera-setup)
    - [Device Discovery](#device-discovery)
    - [Config-Driven](#config-driven)
    - [Robot Integration](#robot-integration)
    - [Async (FastAPI)](#async-fastapi)
  - [Migration from FrameSource](#migration-from-framesource)
    - [Feature Parity Checklist](#feature-parity-checklist)
    - [API Migration Map](#api-migration-map)
    - [Migration Plan](#migration-plan)
  - [Comparison with LeRobot](#comparison-with-lerobot)
  - [Open Design Decisions](#open-design-decisions)
  - [References](#references)

---

## Executive Summary

This document defines the camera/capture interface for the physical-AI ecosystem, packaged as `physicalai.capture` inside `physicalai`.

**Key decisions:**

- **Package**: `physicalai.capture` — lives in `physicalai`, zero coupling to other subpackages, designed for future extraction as a standalone repo
- **Backends**: Low-level capture code absorbed selectively from our team's FrameSource fork, rewritten under `physicalai.capture` naming — no FrameSource branding
- **Primary API**: Dedicated camera classes (`OpenCVCamera`, `RealSenseCamera`, etc.) with explicit constructor parameters
- **Convenience API**: Thin `create_camera()` factory for config-driven workflows
- **Read model**: Three-tier — `read()` (blocking sequential), `read_latest()` (non-blocking latest), `async_read()` (async/await)
- **Frame type**: `Frame` dataclass carrying image data + timestamp + sequence number
- **Sharing**: Explicit only — no invisible global state, no hidden reference counting
- **Callbacks**: Removed from base class — application layer concern

**Goal**: The best camera framework for vision and robotics applications. Clean API, production-grade quality, hardware-agnostic.

---

## Overview

**Design Principles:**

- **Hparams-first**: Explicit constructor args with IDE autocomplete — `OpenCVCamera(index=0, fps=30, width=640)`
- **Context manager**: Safe resource management via `with` statement
- **Async context manager**: `async with` supported for event-loop integration
- **Dedicated classes**: Each camera type is a concrete class, not a factory string — `RealSenseCamera(serial="...")` not `create("realsense", serial="...")`
- **Explicit sharing**: No hidden global state. Multi-consumer access is the application's responsibility.
- **Three-tier reads**: `read()`, `read_latest()`, `async_read()` cover sequential, real-time, and async use cases
- **Timestamped frames**: Every frame carries `timestamp` and `sequence` — no ambiguity about when data was captured
- **Capability mixins**: Optional features (depth, PTZ, format discovery) via composable mixins
- **Zero coupling**: No imports from other `physical_ai` subpackages

---

## Packaging Strategy

`physicalai.capture` lives inside the `physical-ai` repo as a subpackage with **zero coupling** to other subpackages. It is designed as if it were standalone — no internal cross-imports — so it can be extracted into its own repository once mature.

```text
physical-ai/
└── src/physicalai/
    └── capture/
        ├── __init__.py          # Public API: re-exports cameras, Frame, discover_all
        ├── _frame.py            # Frame dataclass
        ├── _sensor.py           # Sensor ABC
        ├── _camera.py           # Camera ABC (extends Sensor)
        ├── _discovery.py        # DeviceInfo, discover_all()
        ├── cameras/
        │   ├── __init__.py
        │   ├── opencv.py        # OpenCVCamera
        │   ├── realsense.py     # RealSenseCamera
        │   ├── basler.py        # BaslerCamera
        │   ├── genicam.py       # GenicamCamera
        │   └── ip.py            # IPCamera
        ├── sources/             # Future: non-live sources
        │   ├── __init__.py
        │   ├── video.py         # VideoSource
        │   └── directory.py     # ImageDirectorySource
        └── mixins/
            ├── __init__.py
            ├── depth.py         # DepthMixin
            ├── ptz.py           # PTZMixin
            └── formats.py       # FormatDiscoveryMixin
```

```python
from physicalai.capture import OpenCVCamera, RealSenseCamera, Frame
from physicalai.capture import create_camera, discover_all
```

**Why subpackage now, standalone later?**

- Rapid iteration without repo/CI overhead
- Zero coupling means extraction is a `mv` + `pyproject.toml` change
- Once other repos (`geti-prompt`, `geti-inspect`) need cameras, extract

---

## Architecture

### Class Hierarchy

```text
Sensor (ABC)                       # Base: connect/disconnect/read/read_latest/async_read
├── Camera (ABC)                   # Adds: device discovery, hardware config (width/height/fps)
│   ├── OpenCVCamera               # USB webcams via OpenCV
│   ├── RealSenseCamera            # Intel RealSense (+ DepthMixin)
│   ├── BaslerCamera               # Basler industrial cameras (pypylon)
│   ├── GenicamCamera              # Generic GenICam devices (harvesters)
│   └── IPCamera                   # RTSP/HTTP network cameras
└── [Future] RecordedSource        # Non-live playback
    ├── VideoSource                # Video file playback
    └── ImageDirectorySource       # Image sequence playback
```

`Sensor` is the universal ABC — anything that produces frames. `Camera` extends it with hardware-specific concepts (device discovery, resolution, FPS). `RecordedSource` is future work for replaying recorded data.

### Package Structure

Internal modules use underscore prefix (`_frame.py`, `_sensor.py`) to signal they are not public API. Users import from `physicalai.capture` directly.

Each camera backend is a separate module under `cameras/`. This keeps dependencies isolated — importing `OpenCVCamera` doesn't pull in `pypylon` or `pyrealsense2`.

Optional SDK imports must be **lazy**: camera modules should import their SDKs only when instantiated or when `connect()` is called, and raise `MissingDependencyError` with an install hint if the extra is not installed. `physicalai.capture.__init__` should avoid eager imports that force optional dependencies.

### Backend Strategy

We selectively absorb low-level capture code from our team's FrameSource fork. The fork is maintained by our team, but the FrameSource brand belongs to the original author. We:

1. **Cherry-pick** proven capture logic (device enumeration, buffer management, format negotiation)
2. **Rewrite** under `physicalai.capture` naming and conventions
3. **Improve** with typed APIs, proper error handling, and timestamped frames
4. **Own** the code — no external dependency on FrameSource at runtime

The fork maintainer continues adding features. We absorb selectively as needed, not wholesale.

---

## Dependencies

OpenCV is the only required dependency. Hardware-specific SDKs are optional extras:

```bash
pip install physicalai                    # Core + OpenCVCamera only
pip install physicalai[realsense]         # + Intel RealSense (pyrealsense2)
pip install physicalai[basler]            # + Basler (pypylon)
pip install physicalai[genicam]           # + GenICam (harvesters)
pip install physicalai[capture]           # All camera dependencies
```

| Camera            | Required Package | Optional Extra |
| ----------------- | ---------------- | -------------- |
| `OpenCVCamera`    | `opencv-python`  | (base)         |
| `RealSenseCamera` | `pyrealsense2`   | `[realsense]`  |
| `BaslerCamera`    | `pypylon`        | `[basler]`     |
| `GenicamCamera`   | `harvesters`     | `[genicam]`    |
| `IPCamera`        | `opencv-python`  | (base)         |

---

## Core Interface

### Frame

Every read operation returns a `Frame` — never a raw numpy array.

```python
@dataclass(frozen=True, slots=True)
class Frame:
    """A captured image with metadata."""

    data: NDArray[np.uint8]    # (H, W, C) or (H, W) image array
    timestamp: float           # time.monotonic() at capture moment
    sequence: int              # Monotonic counter per source (0, 1, 2, ...)
```

**Why not just `ndarray`?**

- `timestamp` answers "when was this frame captured?" — critical for multi-camera sync, latency measurement, and replay. The timestamp uses a monotonic clock at capture time; if the device provides hardware timestamps, implementations should map them to monotonic time where possible.
- `sequence` answers "did I miss any frames?" — enables drop detection
- Frozen dataclass prevents accidental mutation of metadata (the underlying `data` buffer is still mutable)

### Sensor ABC

The base abstraction for anything that produces frames.

```python
class Sensor(ABC):
    """Base interface for frame acquisition."""

    # === Lifecycle ===

    @abstractmethod
    def connect(self) -> None:
        """Open the source. Must be called before reading."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the source and release resources."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the source is currently open."""
        ...

    # === Reading ===

    @abstractmethod
    def read(self) -> Frame:
        """Read the next frame. Blocks until available.

        Frames are returned in sequence — no frames are skipped.
        Use for recording, sequential processing, or any case where
        every frame matters.

        Raises:
            NotConnectedError: If not connected.
            CaptureError: If frame acquisition fails.
        """
        ...

    @abstractmethod
    def read_latest(self) -> Frame:
        """Read the most recent frame. Non-blocking.

        Returns immediately with the latest captured frame. May skip
        intermediate frames. Use for real-time control, teleoperation,
        or any case where freshness matters more than completeness.

        Raises:
            NotConnectedError: If not connected.
            NoFrameError: If no frame is available.
        """
        ...

    async def async_read(self) -> Frame:
        """Read the next frame, yielding to the event loop while waiting.

        Default implementation wraps read() in a thread pool executor.
        Subclasses with native async support can override.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.read)

    # === Async context manager ===

    async def __aenter__(self) -> Self:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.connect)
        return self

    async def __aexit__(self, *args) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.disconnect)

    # === Context manager ===

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    # === Iterator ===

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Frame:
        try:
            return self.read()
        except CaptureError:
            raise StopIteration
```

### Camera ABC

Extends `Sensor` with hardware-specific concepts.

````python
class ColorMode(str, Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"


class Camera(Sensor):
    """Abstract interface for live camera hardware."""

    def __init__(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._color_mode = color_mode

    # Implementations must honor color_mode by converting output as needed.
    # For example, OpenCV provides BGR by default and should convert to RGB when color_mode=RGB.

    @property
    @abstractmethod
    def device_id(self) -> str:
        """Stable identifier for the physical device.

        Examples: "/dev/video0", "serial:12345678", "rtsp://192.168.1.100/stream"
        """
        ...

    @classmethod
    def discover(cls) -> list[DeviceInfo]:
        """List available devices of this camera type.

        Returns empty list if discovery is not supported for this camera type.
        """
        return []

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Create from a configuration dictionary."""
        return cls(**config)

### Errors

`physicalai.capture` defines explicit error types for predictable handling:

```python
class CaptureError(RuntimeError):
    """Base error for capture failures."""


class NotConnectedError(CaptureError):
    """Raised when read methods are called before connect()."""


class NoFrameError(CaptureError):
    """Raised when no frame is available for read_latest()."""


class MissingDependencyError(CaptureError):
    """Raised when a camera SDK extra is not installed."""
````

````

### DeviceInfo

Metadata about a discovered camera, returned by `Camera.discover()`.

```python
@dataclass
class DeviceInfo:
    """Metadata about a discovered camera device."""

    device_id: str              # Stable identifier (serial number, path, URL)
    name: str                   # Human-readable name ("Logitech C920", "D435")
    manufacturer: str = ""      # "Intel", "Basler", etc.
    model: str = ""             # "D435", "acA1920-40gc", etc.
````

### Capability Mixins

Optional capabilities added via mixins. Each mixin adds a `ClassVar` flag.

**DepthMixin** — for cameras with depth sensing:

```python
class DepthMixin:
    """Adds depth capture capability."""

    supports_depth: ClassVar[bool] = True

    @abstractmethod
    def read_depth(self) -> Frame:
        """Read depth frame.

        Returns:
            Frame with data as (H, W) uint16 array in millimeters.
        """
        ...

    def read_rgbd(self) -> tuple[Frame, Frame]:
        """Read aligned RGB and depth frames.

        Returns:
            Tuple of (rgb_frame, depth_frame). RGB is uint8, depth is uint16.
            Default implementation calls read() and read_depth() sequentially.
        """
        return self.read(), self.read_depth()
```

**Note**: `read_rgbd()` returns a tuple, not a single mixed-type array. RGB data is `uint8`, depth data is `uint16` — stacking them into one array would require type coercion and lose information.

**PTZMixin** — for cameras with pan-tilt-zoom:

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
        """Move to absolute position."""
        self.pan(pan)
        self.tilt(tilt)
        self.zoom(zoom)
```

**FormatDiscoveryMixin** — for cameras that support format enumeration:

```python
@dataclass
class Format:
    """A supported camera format."""
    width: int
    height: int
    fps: float
    pixel_format: str = "RGB"


class FormatDiscoveryMixin:
    """Adds resolution/format discovery and selection."""

    supports_format_discovery: ClassVar[bool] = True

    @abstractmethod
    def get_supported_formats(self) -> list[Format]: ...

    @abstractmethod
    def set_format(self, fmt: Format) -> None: ...
```

---

## Read Semantics

Three read methods cover all production use cases:

| Method          | Blocking | Skips Frames | Use Case                                       |
| --------------- | -------- | ------------ | ---------------------------------------------- |
| `read()`        | Yes      | No           | Recording, sequential processing               |
| `read_latest()` | No       | Yes          | Teleoperation, real-time control, live preview |
| `async_read()`  | Yields   | No           | FastAPI endpoints, asyncio event loops         |

**`read()`** — blocks until the next frame is available. Every frame is returned in order. Use when every frame matters (recording, data collection).

```python
with OpenCVCamera(index=0) as cam:
    for i in range(100):
        frame = cam.read()
        save_frame(frame.data, frame.timestamp)
```

**`read_latest()`** — returns the most recent frame immediately. Intermediate frames may be skipped. Use when freshness matters more than completeness (teleoperation, inference). If no frame has been captured yet, it raises `NoFrameError`.

**Capture loop and buffer** — `connect()` starts a background capture loop that fills a small ring buffer (default 2–4 frames). `read()` consumes in order; `read_latest()` returns the newest frame and may drop older ones when the buffer is full. No explicit `start/stop` is required.

```python
with OpenCVCamera(index=0) as cam:
    while running:
        frame = cam.read_latest()
        action = policy.predict(frame.data)
        robot.send_action(action)
```

**`async_read()`** — awaitable version of `read()`. Yields control to the event loop while waiting for the next frame. Implementations may override for native async SDKs.

```python
async def stream_frames(cam: Camera):
    async with cam:
        while True:
            frame = await cam.async_read()
            yield frame
```

This three-tier model is inspired by [LeRobot's camera interface](https://github.com/huggingface/lerobot/tree/main/src/lerobot/cameras), battle-tested in robotics applications.

**Thread safety**: `Sensor` instances are safe to share across threads for read-only access, but concurrent reads are serialized internally. If multiple consumers need strict per-consumer ordering guarantees, create a dedicated `CameraPool` in the application layer.

---

## Proposed Implementations

### OpenCVCamera

USB webcams, built-in cameras, V4L2 devices.

```python
class OpenCVCamera(Camera):
    """USB cameras and built-in webcams via OpenCV.

    ``index`` is convenient for quick prototyping but not stable across
    reboots. For production, prefer ``device_path`` (Linux) or serial
    number-based identification.
    """

    def __init__(
        self,
        *,
        index: int = 0,
        device_path: str | None = None,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
        rotation: int = 0,
        warmup_s: float = 1.0,
    ) -> None: ...
```

### RealSenseCamera

Intel RealSense with depth sensing.

```python
class RealSenseCamera(DepthMixin, Camera):
    """Intel RealSense depth cameras.

    Provides RGB via read() and depth via read_depth() from DepthMixin.
    read_rgbd() returns (rgb_frame, depth_frame) tuple.
    """

    def __init__(
        self,
        *,
        serial_number: str | None = None,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        color_mode: ColorMode = ColorMode.RGB,
        depth_width: int | None = None,
        depth_height: int | None = None,
    ) -> None: ...

    def read_depth(self) -> Frame:
        """Read depth frame from RealSense depth sensor.

        Returns:
            Frame with data as (H, W) uint16 in millimeters.
        """
        ...
```

### BaslerCamera

Basler industrial cameras via pypylon.

```python
class BaslerCamera(FormatDiscoveryMixin, Camera):
    """Basler industrial cameras.

    Achievable framerate depends on ROI/binning, connection bandwidth
    (GigE vs USB3), and exposure time. The ``fps`` parameter sets a
    requested rate; actual rate is capped by hardware limits.
    """

    def __init__(
        self,
        *,
        serial_number: str | None = None,
        fps: int = 30,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None: ...
```

### GenicamCamera

Generic GenICam devices via harvesters.

```python
class GenicamCamera(Camera):
    """Generic GenICam-compliant cameras via harvesters."""

    def __init__(
        self,
        *,
        cti_file: str | Path,
        device_id: str = "",
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None: ...
```

### IPCamera

Network cameras via RTSP/HTTP.

```python
class IPCamera(Camera):
    """Network cameras via RTSP or HTTP streams."""

    def __init__(
        self,
        *,
        url: str,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None: ...
```

---

## Recorded Sources (Future)

Non-live sources for replaying recorded data. Useful when the package is separated from `physicalai` and used for offline development and testing.

**These are future work — not part of the initial implementation.**

```python
class VideoSource(Sensor):
    """Playback from video file."""

    def __init__(
        self,
        *,
        path: str | Path,
        loop: bool = False,
    ) -> None: ...


class ImageDirectorySource(Sensor):
    """Playback from a directory of images."""

    def __init__(
        self,
        *,
        path: str | Path,
        pattern: str = "*.png",
        loop: bool = False,
    ) -> None: ...
```

These extend `Sensor` directly (not `Camera`) since they don't represent live hardware.

---

## Factory Function

Dedicated camera classes are the **primary API**. The factory is a convenience for config-driven workflows (YAML files, database configs, UI dropdowns).

```python
def create_camera(driver: str, **kwargs) -> Camera:
    """Create a camera by driver name.

    Convenience function for config-driven instantiation. Prefer
    dedicated classes (OpenCVCamera, RealSenseCamera, etc.) for
    direct usage.

    Args:
        driver: Camera type — "opencv", "realsense", "basler", "genicam", "ip". Driver names are lowercase and case-insensitive; unknown drivers raise `ValueError`.
        **kwargs: Forwarded to the camera constructor.

    Returns:
        Camera instance.

    Raises:
        ValueError: If the driver name is unknown.
        MissingDependencyError: If the driver requires an optional SDK that is not installed.

    Examples:
        cam = create_camera("opencv", index=0, fps=30)
        cam = create_camera("realsense", serial_number="12345678")
    """
    ...


def discover_all() -> dict[str, list[DeviceInfo]]:
    """Discover available cameras across all supported types.

    Returns:
        Dict mapping driver name to list of discovered devices. All known drivers are included; drivers with missing optional dependencies return an empty list.

    Examples:
        devices = discover_all()
        # {"opencv": [DeviceInfo(...)], "realsense": [DeviceInfo(...)]}
    """
    ...
```

---

## Sharing Model

**Sharing is explicit.** There is no invisible reference counting, no global `_captures` dict, no hidden shared state.

If you need the same camera in two places, you have two options:

1. **Pass the same instance** — the simplest and most explicit approach
2. **Application-level pool** — if your app needs managed multi-consumer access, build a `CameraPool` at the application layer

```python
# Option 1: Pass the instance (recommended)
cam = OpenCVCamera(index=0)
cam.connect()

# Pass to multiple consumers explicitly
display_thread = Thread(target=display_loop, args=(cam,))
record_thread = Thread(target=record_loop, args=(cam,))
```

**Why not invisible sharing?**

- Hidden global state makes testing unreliable — tests that run in parallel interfere with each other
- Implicit behavior surprises users when two `Camera` objects silently share state
- Config conflicts (same device, different FPS) have no clean resolution
- Explicit passing is simple and debuggable

---

## Usage

### Basic

```python
from physicalai.capture import OpenCVCamera, RealSenseCamera

# Single camera, context manager
with OpenCVCamera(index=0, fps=30, width=640, height=480) as cam:
    frame = cam.read()
    print(f"Got {frame.data.shape} at t={frame.timestamp:.3f}")

# Depth camera
with RealSenseCamera(serial_number="12345678") as cam:
    rgb, depth = cam.read_rgbd()
    print(f"RGB: {rgb.data.shape}, Depth: {depth.data.shape}")
```

### Multi-Camera Setup

```python
from physicalai.capture import OpenCVCamera, RealSenseCamera

cameras = {
    "wrist": OpenCVCamera(index=0, fps=30),
    "overhead": RealSenseCamera(serial_number="12345678"),
}

for cam in cameras.values():
    cam.connect()

try:
    frames = {name: cam.read() for name, cam in cameras.items()}
finally:
    for cam in cameras.values():
        cam.disconnect()
```

### Device Discovery

```python
from physicalai.capture import OpenCVCamera, RealSenseCamera, discover_all

# Discover specific type
realsense_devices = RealSenseCamera.discover()
for dev in realsense_devices:
    print(f"{dev.name} (serial: {dev.device_id})")

# Discover all camera types
all_devices = discover_all()
for driver, devices in all_devices.items():
    print(f"{driver}: {len(devices)} device(s)")
```

### Config-Driven

```python
from physicalai.capture import create_camera, OpenCVCamera

# From dict (e.g., loaded from YAML or database)
config = {"index": 0, "fps": 30, "width": 640}
cam = OpenCVCamera.from_config(config)

# Factory for driver-string configs (UI dropdowns, YAML)
cam = create_camera("realsense", serial_number="12345678", fps=30)
```

### Robot Integration

```python
from physicalai.capture import RealSenseCamera

camera = RealSenseCamera(fps=30)
robot = SO101.from_config("robot.yaml")
policy = InferenceModel.load("./exports/act_policy")

with robot, camera:
    while True:
        frame = camera.read_latest()
        action = policy.select_action({
            "images": {"wrist": frame.data},
            "state": robot.get_state(),
        })
        robot.send_action(action)
```

### Async (FastAPI)

```python
from physicalai.capture import OpenCVCamera

camera = OpenCVCamera(index=0, fps=30)
camera.connect()

@app.get("/frame")
async def get_frame():
    frame = await camera.async_read()
    return {"timestamp": frame.timestamp, "shape": frame.data.shape}

@app.on_event("shutdown")
async def shutdown():
    camera.disconnect()
```

---

## Migration from FrameSource

The application backend currently uses FrameSource in 6 files. Migration swaps FrameSource for `physicalai.capture` in a single PR once feature parity is reached.

### Feature Parity Checklist

| FrameSource API                               | physicalai.capture Equivalent                       | Status                                   |
| --------------------------------------------- | --------------------------------------------------- | ---------------------------------------- |
| `FrameSourceFactory.create(driver, **params)` | `OpenCVCamera(...)` or `create_camera(driver, ...)` | Direct replacement                       |
| `.connect()`                                  | `.connect()`                                        | Same                                     |
| `.read()` → `(success, frame)`                | `.read()` → `Frame`                                 | Returns `Frame` (raises on failure)      |
| `.start_async()` + `.get_latest_frame()`      | `.read_latest()`                                    | Simplified to one call                   |
| `.stop()`                                     | (not needed)                                        | `read_latest()` is stateless             |
| `.disconnect()`                               | `.disconnect()`                                     | Same                                     |
| `FrameSourceFactory.discover_devices(driver)` | `discover_all()` or `Camera.discover()`             | Direct replacement                       |
| `.get_supported_formats()`                    | `FormatDiscoveryMixin.get_supported_formats()`      | Via mixin                                |
| `.attach_processor()`                         | Removed                                             | Was broken in production (commented out) |

### API Migration Map

**camera_worker.py** — sync read pattern:

```python
# Before (FrameSource)
source = FrameSourceFactory.create(driver, **params)
source.connect()
success, frame = source.read()
source.disconnect()

# After (physicalai.capture)
camera = create_camera(driver, **params)
camera.connect()
frame = camera.read()  # frame.data for the image
camera.disconnect()
```

**teleoperate_worker.py / inference_worker.py** — async latest-frame pattern:

```python
# Before (FrameSource)
source = FrameSourceFactory.create(driver, **params)
source.connect()
source.start_async()
frame = source.get_latest_frame()
source.stop()
source.disconnect()

# After (physicalai.capture)
camera = create_camera(driver, **params)
camera.connect()
frame = camera.read_latest()  # frame.data for the image
camera.disconnect()
```

**hardware.py** — device discovery:

```python
# Before (FrameSource)
devices = FrameSourceFactory.discover_devices(driver)

# After (physicalai.capture)
devices = discover_all()
# or: devices = RealSenseCamera.discover()
```

**camera.py** — format query:

```python
# Before (FrameSource)
source = FrameSourceFactory.create(driver, **params)
formats = source.get_supported_formats()

# After (physicalai.capture)
camera = BaslerCamera(serial_number="12345678")
camera.connect()
formats = camera.get_supported_formats()  # via FormatDiscoveryMixin
```

### Migration Plan

1. **Build** `physicalai.capture` in parallel — no changes to existing application code
2. **Validate** feature parity against the checklist above
3. **Swap** in a single PR: replace FrameSource imports with `physicalai.capture` imports in all 6 backend files
4. **Remove** FrameSource dependency from `application/backend/pyproject.toml`

The application's existing retry logic (`CameraConnectionManager` with `tenacity`) stays in the application layer — error recovery is not the camera library's responsibility.

---

## Comparison with LeRobot

| Aspect           | physicalai.capture                                 | LeRobot cameras                                   |
| ---------------- | -------------------------------------------------- | ------------------------------------------------- |
| Base class       | `Sensor` → `Camera`                                | `Camera` ABC                                      |
| Read model       | 3-tier: `read()`, `read_latest()`, `async_read()`  | 3-tier: `read()`, `read_latest()`, `async_read()` |
| Frame type       | `Frame(data, timestamp, sequence)`                 | Raw `ndarray` (no metadata)                       |
| Lifecycle        | `connect()` / `disconnect()`                       | `connect()` / `disconnect()`                      |
| Hardware support | OpenCV, RealSense, Basler, GenICam, IP cameras     | OpenCV, RealSense, (fewer industrial)             |
| Depth            | `DepthMixin` with `read_depth()` → `Frame(uint16)` | Not built-in                                      |
| PTZ              | `PTZMixin`                                         | Not built-in                                      |
| Config           | `from_config()` + dataclass configs                | Pydantic `CameraConfig`                           |
| Discovery        | `Camera.discover()` + `discover_all()`             | Not built-in                                      |
| Factory          | Optional `create_camera()` convenience             | Not applicable                                    |
| Sharing          | Explicit (pass instance)                           | Explicit                                          |

We adopted LeRobot's three-tier read model and explicit `connect/disconnect` lifecycle. We add timestamped frames, depth/PTZ mixins, industrial camera support, and device discovery.

---

## Open Design Decisions

**1. Buffer Policy**
What ring buffer size? What happens when the buffer is full — drop oldest or block? This affects `read()` vs `read_latest()` behavior under load. Likely: small ring buffer (2-4 frames), drop oldest.

**2. Multi-Consumer Access**
If multiple consumers need the same camera, should the library provide a `CameraPool`? Or is passing the same instance sufficient? Current position: application layer concern. Revisit if multiple teams hit this need.

**3. Thread Model**
One background capture thread per camera, or a shared thread pool? Per-camera threads are simpler but scale poorly with many cameras. Likely: per-camera threads initially, optimize later.

**4. Error Recovery**
Retry on transient failures (USB disconnect/reconnect) in the library, or leave to the application? The application backend already has `tenacity`-based retry in `CameraConnectionManager`. Current position: library raises, application retries.

---

## References

- [Strategy](../strategy.md) — Architecture vision and key decisions
- [Robot Interface Design](./robot-interface.md) — Robot interface specification
- [FrameSource Repository](https://github.com/ArendJanKramer/FrameSource) — Original camera library (reference only)
- [LeRobot Cameras](https://github.com/huggingface/lerobot/tree/main/src/lerobot/cameras) — LeRobot's camera interface (design inspiration)

---

_Last Updated: 2026-02-18_
