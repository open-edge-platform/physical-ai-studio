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
    - [Camera ABC](#camera-abc)
    - [Errors](#errors)
    - [DeviceInfo](#deviceinfo)
    - [Capability Mixins](#capability-mixins)
  - [Read Semantics](#read-semantics)
    - [Why No Iterator Protocol](#why-no-iterator-protocol)
  - [Proposed Implementations](#proposed-implementations)
    - [UVCCamera](#uvccamera)
    - [RealSenseCamera](#realsensecamera)
    - [BaslerCamera](#baslercamera)
    - [GenicamCamera](#genicamcamera)
    - [IPCamera](#ipcamera)
  - [Recorded Sources (Future)](#recorded-sources-future)
  - [Multi-Camera Synchronization](#multi-camera-synchronization)
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
  - [Logging](#logging)
  - [Testing Strategy](#testing-strategy)
  - [Configuration Validation](#configuration-validation)
  - [Open Design Decisions](#open-design-decisions)
  - [References](#references)

---

## Executive Summary

This document defines the camera/capture interface for the physical-AI ecosystem, packaged as `physicalai.capture` inside `physicalai`.

**Key decisions:**

- **Package**: `physicalai.capture` — lives in `physicalai`, zero coupling to other subpackages, designed for future extraction as a standalone repo
- **Backends**: Low-level capture code absorbed selectively from our team's FrameSource fork, rewritten under `physicalai.capture` naming — no FrameSource branding
- **Primary API**: Dedicated camera classes (`UVCCamera`, `RealSenseCamera`, etc.) with explicit constructor parameters
- **Convenience API**: Thin `create_camera()` factory for config-driven workflows
- **Read model**: Three-tier — `read()` (blocking sequential), `read_latest()` (non-blocking latest), `async_read()` (async/await)
- **Multi-camera**: `read_cameras()` / `async_read_cameras()` for temporally aligned multi-camera reads
- **Frame type**: `Frame` frozen slotted dataclass carrying image data + timestamp + sequence number
- **Sharing**: Explicit only — no invisible global state, no hidden reference counting
- **Callbacks**: Removed from base class. Application layer concern
- **No iterator protocol**: `Camera` does not implement `__iter__`. Explicit `read()` calls are clearer and avoid silent error swallowing

**Goal**: The best camera framework for vision and robotics applications. Clean API, production-grade quality, hardware-agnostic.

---

## Overview

**Design Principles:**

- **Hparams-first**: Explicit constructor args with IDE autocomplete — `UVCCamera(device=0, fps=30, width=640)`
- **Context manager**: Safe resource management via `with` statement
- **Async context manager**: `async with` supported for event-loop integration
- **Dedicated classes**: Each camera type is a concrete class, not a factory string — `RealSenseCamera(serial="...")` not `create("realsense", serial="...")`
- **Explicit sharing**: No hidden global state. Multi-consumer access is the application's responsibility.
- **Three-tier reads**: `read()`, `read_latest()`, `async_read()` cover sequential, real-time, and async use cases
- **Multi-camera sync**: `read_cameras()` / `async_read_cameras()` for temporally aligned multi-camera reads
- **Timestamped frames**: Every frame carries `timestamp` and `sequence`. No ambiguity about when data was captured
- **Capability mixins**: Optional features (depth, PTZ, format discovery) via composable mixins
- **Zero coupling**: No imports from other `physical_ai` subpackages

---

## Packaging Strategy

`physicalai.capture` lives inside the `physical-ai` repo as a subpackage with **zero coupling** to other subpackages. It is designed as if it were standalone (no internal cross-imports) so it can be extracted into its own repository once mature.

```text
physical-ai/
└── src/physicalai/
    └── capture/
        ├── __init__.py          # Public API: re-exports cameras, Frame, discover_all, read_cameras
        ├── frame.py             # Frame dataclass
        ├── camera.py            # Camera ABC
        ├── discovery.py         # DeviceInfo, discover_all()
        ├── sync.py              # read_cameras(), async_read_cameras()
        ├── errors.py            # Error hierarchy
        ├── cameras/
        │   ├── __init__.py
        │   ├── uvccamera.py     # USB Video Class cameras
        │   ├── realsense.py     # RealSenseCamera
        │   ├── basler.py        # BaslerCamera
        │   ├── genicam.py       # GenicamCamera
        │   └── ip.py            # IPCamera
        └── mixins/
            ├── __init__.py
            ├── depth.py         # DepthMixin
            ├── ptz.py           # PTZMixin
            └── formats.py       # FormatDiscoveryMixin
```

```python
from physicalai.capture import UVCCamera, RealSenseCamera, Frame
from physicalai.capture import create_camera, discover_all, read_cameras
```

**Why subpackage now, standalone later?**

- Rapid iteration without repo/CI overhead
- Zero coupling means extraction is a `mv` + `pyproject.toml` change
- Once other repos (`geti-prompt`, `geti-inspect`) need cameras, extract

---

## Architecture

### Class Hierarchy

```text
Camera (ABC)                       # Base: connect/disconnect/read/read_latest/async_read
├── UVCCamera                      # USB webcams/UVC cameras
├── RealSenseCamera                # RealSense (+ DepthMixin)
├── BaslerCamera                   # Basler industrial cameras (pypylon)
├── GenicamCamera                  # Generic GenICam devices (harvesters)
└── IPCamera                       # RTSP/HTTP network cameras
```

`Camera` is the single ABC for all live hardware.

**Future extensibility:** When non-live sources (video file playback, image directories)
are added, they should _not_ inherit from `Camera`. The semantics diverge (e.g.,
`read_latest()` is meaningless for recorded data). Instead, a lightweight `typing.Protocol`
can be introduced at that point to define the shared surface (`read()`, `connect()`,
`disconnect()`, context manager) without forcing a common base class.

### Package Structure

Users import from `physicalai.capture` directly; `__all__` in `__init__.py` defines the public API surface. Internal module names use plain names (no underscore prefix), consistent with the `physicalai.robot` package.

Each camera backend is a separate module under `cameras/`. This keeps dependencies isolated: importing `UVCCamera` doesn't pull in `pypylon` or `pyrealsense2`.

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

OmniCamera is the required UVC dependency for non-Linux platforms (macOS/Windows). It is installed from git until PyPI wheels are available. Hardware-specific SDKs are optional extras:

```bash
pip install physicalai                    # Core + UVCCamera (UVC support)
pip install physicalai[realsense]         # + RealSense (pyrealsense2)
pip install physicalai[basler]            # + Basler (pypylon)
pip install physicalai[genicam]           # + GenICam (harvesters)
pip install physicalai[capture]           # All camera dependencies
```

OmniCamera is installed from git until PyPI wheels land:

```toml
# pyproject.toml
omni_camera = {git = "https://github.com/ArendJanKramer/OmniCamera.git", rev = "master"}
```

> **Note:** The `opencv` backend has been removed. `backend="opencv"` raises
> `ValueError("The 'opencv' backend has been removed. Use backend='omnicamera' or backend='auto' instead.")`.
> OpenCV (`opencv-python`) is no longer a required dependency.

| Camera            | Required Package                           | Optional Extra | Platform                   |
| ----------------- | ------------------------------------------ | -------------- | -------------------------- |
| `UVCCamera`       | `omni_camera` (non-Linux) / kernel (Linux) | (base)         | All (auto-selects backend) |
| `RealSenseCamera` | `pyrealsense2`                             | `[realsense]`  | All                        |
| `BaslerCamera`    | `pypylon`                                  | `[basler]`     | All                        |
| `GenicamCamera`   | `harvesters`                               | `[genicam]`    | All                        |
| `IPCamera`        | —                                          | (base)         | All                        |

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

- `timestamp` answers "when was this frame captured?". Critical for multi-camera sync, latency measurement, and replay. The timestamp uses a monotonic clock at capture time; if the device provides hardware timestamps, implementations should map them to monotonic time where possible.
- `sequence` answers "did I miss any frames?". Enables drop detection
- Frozen dataclass prevents accidental mutation of metadata (the underlying `data` buffer is still mutable)

**Why uint8 only?** Some industrial cameras (Basler, GenICam) natively produce 10/12/16-bit images. `physicalai.capture` normalizes all color images to `uint8` at capture time. This is a deliberate simplification: every robotics inference model in the target domain (ACT, Diffusion Policy, VLAs) expects `uint8` RGB input. Supporting mixed dtypes would complicate the `Frame` type, every consumer, and every preprocessor for a use case that doesn't exist yet. Depth data (`DepthMixin`) uses `uint16` because millimeter-precision depth inherently requires it. If full bit-depth color capture becomes necessary for industrial vision use cases, a `raw_read()` escape hatch can be added without changing the `Frame` contract.

### Camera ABC

The single ABC for all live camera hardware. Combines lifecycle, reading, hardware config, and discovery.

```python
class ColorMode(str, Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"


class Camera(ABC):
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
        self.__executor = None  # async read executor

    # Implementations must honor color_mode by converting output as needed.
    # OmniCamera outputs RGB natively. BGR is obtained via [:, :, ::-1]; GRAY via
    # weighted luma (0.299R + 0.587G + 0.114B) — no cv2 dependency required.

    # === Lifecycle ===

    @abstractmethod
    def connect(self, timeout: float = 5.0) -> None:
        """Open the camera and verify it produces frames.

        Blocks until the first frame is successfully captured, confirming
        the hardware is operational. If no frame arrives within ``timeout``
        seconds, raises ``CaptureTimeoutError``.

        After connect() returns, read() and read_latest() are guaranteed
        to succeed (barring subsequent hardware failures).

        Args:
            timeout: Maximum seconds to wait for the first frame.
                Covers both hardware initialization and first-frame capture.

        Raises:
            CaptureTimeoutError: Camera opened but no frame within timeout.
            CaptureError: Hardware-level connection failure.
        """
        ...

    @abstractmethod
    def _do_disconnect(self) -> None:
        """Release hardware resources. Called by disconnect().

        Subclasses implement this to release SDK handles, close devices,
        and stop background capture loops. Do not override disconnect()
        directly — the base class handles executor cleanup.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from camera hardware and release all resources.

        Calls _do_disconnect() to release hardware, then shuts down the
        async executor if it was created. Subclasses override
        _do_disconnect(), not this method.

        If an async_read() is in flight when disconnect() is called,
        unstarted futures are cancelled. Callers should ensure no reads
        are being awaited after disconnecting.
        """
        self._do_disconnect()
        if self.__executor is not None:
            self.__executor.shutdown(wait=False, cancel_futures=True)
            self.__executor = None

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the camera is currently open."""
        ...

    @property
    @abstractmethod
    def device_id(self) -> str:
        """Identifier for the physical device this instance targets.

        Stable for the lifetime of the connection. May change across
        reconnects for OS-assigned paths (e.g., /dev/video0).

        Should match the corresponding DeviceInfo.device_id returned
        by discover() for the same device.

        Examples: "/dev/video0", "serial:12345678", "rtsp://192.168.1.100/stream"
        """
        ...

    @property
    def _executor(self) -> ThreadPoolExecutor:
        """Lazy-initialized per-camera executor for async reads."""
        if self.__executor is None:
            self.__executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"capture-{self.device_id}",
            )
        return self.__executor

    # === Reading ===

    @abstractmethod
    def read(self, timeout: float | None = None) -> Frame:
        """Read the next frame. Blocks until available.

        Frames are returned in sequence; no frames are skipped.
        Use for recording, sequential processing, or any case where
        every frame matters.

        Args:
            timeout: Maximum seconds to wait for a frame. None means
                wait indefinitely. Defaults to None.

        Raises:
            NotConnectedError: If not connected.
            CaptureTimeoutError: If no frame arrives within timeout.
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
            CaptureError: If frame acquisition fails.
        """
        ...

    async def async_read(self, timeout: float | None = None) -> Frame:
        """Read the next frame, yielding to the event loop while waiting.

        Default implementation offloads read() to a dedicated per-camera
        ThreadPoolExecutor(max_workers=1), lazily created on first async
        call and cleaned up by disconnect(). Sync-only usage incurs no
        thread overhead. Subclasses with native async support can override.

        Args:
            timeout: Maximum seconds to wait for a frame. None means
                wait indefinitely.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.read, timeout)

    # === Discovery ===

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

    # === Context managers ===

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    async def __aenter__(self) -> Self:
        loop = asyncio.get_running_loop()
        # Uses default executor. Per-camera executor doesn't exist until first async_read()
        await loop.run_in_executor(None, self.connect)
        return self

    async def __aexit__(self, *args) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.disconnect)
```

**`color_mode` output shapes:**

| `color_mode` | `read()` shape | dtype   | Notes                                                    |
| ------------ | -------------- | ------- | -------------------------------------------------------- |
| `RGB`        | `(H, W, 3)`    | `uint8` | Default. OmniCamera outputs RGB natively (no conversion) |
| `BGR`        | `(H, W, 3)`    | `uint8` | Slice conversion: `frame[:, :, ::-1]`. No cv2 needed     |
| `GRAY`       | `(H, W)`       | `uint8` | Weighted luma: `0.299R + 0.587G + 0.114B`. No cv2        |

`color_mode` applies only to color image reads (`read()`, `async_read()`, and the RGB
portion of `read_rgbd()`). `read_depth()` always returns `(H, W)` `uint16` regardless
of `color_mode`.

### Errors

`physicalai.capture` defines explicit error types for predictable handling:

```python
class CaptureError(RuntimeError):
    """Base error for capture failures."""


class NotConnectedError(CaptureError):
    """Raised when read methods are called before connect()."""


class CaptureTimeoutError(CaptureError):
    """Raised when a read or connect operation exceeds its timeout."""


class MissingDependencyError(CaptureError):
    """Raised when a camera SDK extra is not installed."""


class DeviceInUseError(CaptureError):
    """Raised when connecting to a device already held by another instance.

    Note: Not yet enforced. Reserved for future device exclusivity checks.
    """
```

### DeviceInfo

Metadata about a discovered camera, returned by `Camera.discover()`.

```python
@dataclass
class DeviceInfo:
    """Metadata about a discovered camera device."""

    device_id: str              # Backend-specific identifier (e.g., "/dev/video0", index, IP)
    name: str = ""              # Human-readable name ("Logitech C920", "D435")
    driver: str = ""            # Backend that found it: "uvc", "realsense", "basler", "genicam"
    hardware_id: str = ""       # Stable cross-backend ID: serial number or USB bus path
    manufacturer: str = ""      # "Intel", "Basler", etc.
    model: str = ""             # "D435", "acA1920-40gc", etc.
    metadata: dict = field(default_factory=dict)  # Backend-specific extras
```

`hardware_id` enables deduplication when the same physical device is discovered by
multiple backends (see [Device Discovery](#device-discovery)).

### Capability Mixins

Optional capabilities added via mixins. Each mixin adds a `ClassVar` flag.

**DepthMixin**: for cameras with depth sensing:

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

**Note**: `read_rgbd()` returns a tuple, not a single mixed-type array. RGB data is `uint8`, depth data is `uint16`; stacking them into one array would require type coercion and lose information.

**PTZMixin**: for cameras with pan-tilt-zoom:

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

**FormatDiscoveryMixin**: for cameras that support format enumeration:

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

**`read()`**: blocks until the next frame is available. Every frame is returned in order. Use when every frame matters (recording, data collection). Accepts an optional `timeout` parameter; raises `CaptureTimeoutError` if no frame arrives in time.

```python
with UVCCamera(device=0) as cam:
    for i in range(100):
        frame = cam.read(timeout=5.0)
        save_frame(frame.data, frame.timestamp)
```

**`read_latest()`**: returns the most recent frame immediately. Intermediate frames may be skipped. Use when freshness matters more than completeness (teleoperation, inference). Since `connect()` guarantees that at least one frame is available before returning, `read_latest()` will always succeed on a connected camera.

**`read_latest()` contract**: `read_latest()` must return without blocking for the next hardware frame. How this is achieved depends on the backend — OmniCamera exposes `poll_frame_np()` (non-blocking single poll), so `read_latest()` returns the cached last-good frame if the poll returns `None`. RealSense and pypylon maintain internal frame buffers at the OS or driver level, making the latest frame available without a dedicated capture thread. Backends whose SDKs only offer blocking reads may need an internal capture loop, but this is a subclass implementation detail, not an ABC requirement. May return the same frame as a previous call if no new frame has been captured since; use `frame.sequence` to detect duplicates.

```python
with UVCCamera(device=0) as cam:
    while running:
        frame = cam.read_latest()
        action = model({"images": {"wrist": frame.data}})
        robot.send_action(action)
```

**`async_read()`**: awaitable version of `read()`. Yields control to the event loop while waiting for the next frame.

Camera SDKs (OmniCamera, pyrealsense2, pypylon) are blocking at the application level: a single `read()` call holds
the calling thread until a frame arrives from hardware. In an async application, calling
`read()` directly from a coroutine would freeze the entire event loop for the duration of
the capture (up to 33ms at 30fps), starving all other coroutines.

The default implementation offloads `read()` to a dedicated per-camera
`ThreadPoolExecutor(max_workers=1)`, lazily created on the first `async_read()` call and
cleaned up by `disconnect()`. Sync-only usage incurs no thread overhead. A per-camera
executor, rather than the shared default pool, ensures that camera reads cannot contend
with each other or with unrelated I/O tasks in the application. Subclasses with native
async support (e.g., an RTSP backend using `asyncio` sockets) may override `async_read()`
to avoid the thread indirection.

```python
async def stream_frames(cam: Camera):
    async with cam:
        while True:
            frame = await cam.async_read()
            yield frame
```

This three-tier model is inspired by [LeRobot's camera interface](https://github.com/huggingface/lerobot/tree/main/src/lerobot/cameras), battle-tested in robotics applications.

**Thread safety**: `Camera` instances are safe to share across threads for read-only access, but concurrent reads are serialized internally. If multiple consumers need strict per-consumer ordering guarantees, create a dedicated `CameraPool` in the application layer.

### Why No Iterator Protocol

`Camera` deliberately does not implement `__iter__` / `__next__`. A natural implementation
would convert `CaptureError` to `StopIteration`, meaning any transient hardware error
(USB glitch, corrupt frame) silently terminates the loop, a data-loss footgun in
recording workflows. The explicit `read()` call in a standard `for` / `while` loop is one
line longer and gives the caller full control over error handling.

---

## Proposed Implementations

### UVCCamera

USB webcams, built-in cameras, and UVC devices on **all platforms**.
`UVCCamera` is a facade that delegates to a platform-specific backend:

- **macOS/Windows/Linux**: backend via [OmniCamera](https://github.com/ArendJanKramer/OmniCamera)
- **Linux** (optional): native V4L2 backend via V4L2 ioctls

The `backend` parameter selects the backend explicitly (`"v4l2"` or `"omnicamera"`); the default is `"omnicamera"`. On Linux, pass `backend="v4l2"` to use the native backend.

**Discovery** delegates to the active backend. On non-Linux, uses `omni_camera.query(only_usable=True)` which returns a `list[CameraInfo]`. If the `omni_camera` package is not installed, discovery gracefully returns `[]`.

**Format negotiation** (OmniCamera backend) follows a pipeline:

1. `camera.get_format_options()` — enumerate supported formats
2. `prefer_*` chainable filters (`.prefer_width_range()`, `.prefer_height_range()`, `.prefer_fps_range()`, `.prefer_frame_format()`)
3. `.resolve()` — best-effort match; relaxes filters and falls back to `.resolve_default()` if needed
4. `camera.open(fmt)` — opens the device at the resolved format

**Read semantics** (OmniCamera backend):

- `read()`: blocking poll-loop with 1 ms sleep (`_POLL_INTERVAL_S = 0.001`); raises `CaptureTimeoutError` if no frame arrives within timeout
- `read_latest()`: single `poll_frame_np()` call; returns cached last-good frame if poll returns `None`; raises `NotConnectedError` if not connected

**Color output**: OmniCamera outputs **RGB natively** (unlike OpenCV's BGR-native output). Color mode conversions:

- `RGB`: no-op (pass-through)
- `BGR`: `frame[:, :, ::-1]` (array slice — no `cv2` dependency)
- `GRAY`: weighted luma `0.299R + 0.587G + 0.114B` (no `cv2` dependency)

```python
class UVCCamera(Camera):
    """Camera facade for UVC devices (USB Video Class).

    Delegates to OmniCamera (macOS/Windows) or V4L2Camera (Linux)
    based on the ``backend`` parameter.

    ``device`` is a unified selector: integer index (0, 1, ...) or
    device path string ("/dev/video0" on Linux).
    """

    def __init__(
        self,
        *,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        color_mode: ColorMode = ColorMode.RGB,
        backend: Literal["v4l2", "omnicamera"] = "omnicamera",
        backend_options: dict[str, Any] | None = None,
    ) -> None: ...

    @classmethod
    def discover(cls) -> list[DeviceInfo]:
        """List available UVC cameras.

        Delegates to the platform backend. Returns empty list if the
        required SDK is not installed (graceful degradation).
        """
        ...
```

Valid backend strings are: `"v4l2"` (Linux), `"omnicamera"` (macOS/Windows).

### RealSenseCamera

RealSense with depth sensing.

```python
class RealSenseCamera(DepthMixin, Camera):
    """RealSense depth cameras.

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

**These are future work, not part of the initial implementation.**

Recorded sources will **not** inherit from `Camera`. The semantics diverge (e.g.,
`read_latest()` is meaningless for a video file). Instead, they will be standalone classes
that satisfy a shared `typing.Protocol` (see [Class Hierarchy](#class-hierarchy)).

```python
class VideoSource:
    """Playback from video file."""

    def __init__(
        self,
        *,
        path: str | Path,
        loop: bool = False,
    ) -> None: ...


class ImageDirectorySource:
    """Playback from a directory of images."""

    def __init__(
        self,
        *,
        path: str | Path,
        pattern: str = "*.png",
        loop: bool = False,
    ) -> None: ...
```

---

## Multi-Camera Synchronization

Reading from multiple cameras sequentially introduces inter-camera skew (~33ms per
camera at 30fps). For multi-view inference policies (ACT, Diffusion Policy) that expect
temporally aligned observations, this skew can degrade performance during fast motions.

`read_cameras()` and `async_read_cameras()` read from all cameras in parallel,
minimizing skew to ~1–3ms (limited by OS thread scheduling).

```python
@dataclass(frozen=True)
class SyncedFrames:
    """Temporally aligned frames from multiple cameras."""
    frames: dict[str, Frame]     # camera name → frame
    max_skew_ms: float           # worst-case temporal skew: max(timestamps) - min(timestamps),
                                 # converted to milliseconds. Measures how far apart in time
                                 # the earliest and latest frames were captured. Lower is
                                 # better; typical values are 1–3ms with thread-based parallel
                                 # reads. Applications can log this to monitor sync quality
                                 # or assert thresholds for safety-critical control loops.


def read_cameras(
    cameras: dict[str, Camera],
    timeout: float = 1.0,
    latest: bool = True,
) -> SyncedFrames:
    """Read one frame from each camera in parallel, minimizing temporal skew.

    Spawns one thread per camera and reads simultaneously.
    Returns when all threads complete or timeout is reached.

    Args:
        cameras: Mapping of name to connected Camera instance.
        timeout: Max seconds to wait for all cameras to respond.
        latest: If True (default), uses read_latest() for freshest frame.
            If False, uses read() for sequential capture. Use False for
            recording workflows where every frame matters.

    Raises:
        CaptureTimeoutError: One or more cameras didn't respond in time.
        CaptureError: A camera failed during read.
    """
    ...


async def async_read_cameras(
    cameras: dict[str, Camera],
    timeout: float = 1.0,
    latest: bool = True,
) -> SyncedFrames:
    """Async version of read_cameras() using asyncio.gather.

    Uses asyncio.create_task to schedule reads on all cameras
    concurrently, then gathers the results.
    """
    ...
```

**Usage:**

```python
from physicalai.capture import read_cameras

cameras = {"wrist": wrist_cam, "overhead": overhead_cam}
synced = read_cameras(cameras)
print(f"Skew: {synced.max_skew_ms:.1f}ms")  # ~1–3ms vs ~33ms sequential
```

**Hardware sync:** For sub-millisecond synchronization requirements, hardware trigger
(genlock) is needed. This is backend-specific (e.g., RealSense GPIO, Basler trigger
lines) and would live in specialized subclass methods, not in the general API.

---

## Factory Function

Dedicated camera classes are the **primary API**. The factory is a convenience for config-driven workflows (YAML files, database configs, UI dropdowns).

```python
def create_camera(driver: str, **kwargs) -> Camera:
    """Create a camera by driver name.

    Convenience function for config-driven instantiation. Prefer
    dedicated classes (UVCCamera, RealSenseCamera, etc.) for
    direct usage.

    Args:
        driver: Camera type, one of "uvc", "realsense", "basler", "genicam", "ip".
            Driver names are lowercase and case-insensitive; unknown drivers raise `ValueError`.
            Note: "opencv" is no longer valid and raises ValueError.
        **kwargs: Forwarded to the camera constructor.

    Returns:
        Camera instance.

    Raises:
        ValueError: If the driver name is unknown or is the removed "opencv" driver.
        MissingDependencyError: If the driver requires an optional SDK that is not installed.

    Examples:
        cam = create_camera("uvc", device=0, fps=30)
        cam = create_camera("realsense", serial_number="12345678")
    """
    ...


def discover_all() -> dict[str, list[DeviceInfo]]:
    """Discover available cameras across all supported types.

    Returns:
        Dict mapping driver name to list of discovered devices. All known
        drivers are included; drivers with missing optional dependencies
        return an empty list.

    Note:
        The same physical device may appear under multiple drivers (e.g., a
        USB camera found by both UVC and GenICam). Use ``hardware_id`` to
        deduplicate across backends when needed::

            all_devices = discover_all()
            seen: set[str] = set()
            unique: list[DeviceInfo] = []
            for devices in all_devices.values():
                for dev in devices:
                    key = dev.hardware_id or f"{dev.driver}:{dev.device_id}"
                    if key not in seen:
                        seen.add(key)
                        unique.append(dev)

    Examples:
        devices = discover_all()
        # {"uvc": [DeviceInfo(...)], "realsense": [DeviceInfo(...)]}
    """
    ...
```

---

## Sharing Model

**Sharing is explicit.** There is no invisible reference counting, no global `_captures` dict, no hidden shared state.

If you need the same camera in two places, you have two options:

1. **Pass the same instance**: the simplest and most explicit approach
2. **Application-level pool**: if your app needs managed multi-consumer access, build a `CameraPool` at the application layer

```python
# Option 1: Pass the instance (recommended)
cam = UVCCamera(device=0)
cam.connect()

# Pass to multiple consumers explicitly
display_thread = Thread(target=display_loop, args=(cam,))
record_thread = Thread(target=record_loop, args=(cam,))
```

**Why not invisible sharing?**

- Hidden global state makes testing unreliable: tests that run in parallel interfere with each other
- Implicit behavior surprises users when two `Camera` objects silently share state
- Config conflicts (same device, different FPS) have no clean resolution
- Explicit passing is simple and debuggable

### Device Exclusivity

Creating multiple `Camera` instances targeting the same physical device is **undefined
behavior**. The outcome depends on the backend and OS:

| Backend                     | Typical behavior with duplicate open                 |
| --------------------------- | ---------------------------------------------------- |
| UVCCamera (V4L2/OmniCamera) | Second `connect()` fails or produces corrupt frames  |
| RealSense                   | Both pipelines connect but compete for USB bandwidth |
| Basler / GenICam            | SDK rejects the second open with an access error     |
| IP Camera                   | Both instances connect (read-only RTSP allows it)    |

**Guidance:** Do not create multiple connected `Camera` instances for the same device.
If you need multiple consumers, read from a single `Camera` and distribute frames in
application code.

**Future work:** Enforce one-connected-instance-per-device via a per-subclass registry
in `connect()`, raising `DeviceInUseError` on conflicts. Each subclass would identify
its device via a `_device_key()` method (device index, serial number, URL, etc.), and
weak references would auto-release forgotten instances on garbage collection.

---

## Usage

### Basic

```python
from physicalai.capture import UVCCamera, RealSenseCamera

# Single camera, context manager
with UVCCamera(device=0, fps=30, width=640, height=480) as cam:
    frame = cam.read()
    print(f"Got {frame.data.shape} at t={frame.timestamp:.3f}")

# Depth camera
with RealSenseCamera(serial_number="12345678") as cam:
    rgb, depth = cam.read_rgbd()
    print(f"RGB: {rgb.data.shape}, Depth: {depth.data.shape}")
```

### Multi-Camera Setup

```python
from physicalai.capture import UVCCamera, RealSenseCamera, read_cameras

cameras = {
    "wrist": UVCCamera(device=0, fps=30),
    "overhead": RealSenseCamera(serial_number="12345678"),
}

for cam in cameras.values():
    cam.connect()

try:
    # Parallel read with temporal alignment (~1–3ms skew)
    synced = read_cameras(cameras)
    wrist_frame = synced.frames["wrist"]
    overhead_frame = synced.frames["overhead"]
finally:
    for cam in cameras.values():
        cam.disconnect()
```

### Device Discovery

```python
from physicalai.capture import UVCCamera, RealSenseCamera, discover_all

# Discover UVC cameras
uvc_devices = UVCCamera.discover()
for dev in uvc_devices:
    print(f"{dev.name} (id: {dev.device_id})")

# Discover all camera types
all_devices = discover_all()
for driver, devices in all_devices.items():
    print(f"{driver}: {len(devices)} device(s)")
```

### Config-Driven

```python
from physicalai.capture import create_camera, UVCCamera

# From dict (e.g., loaded from YAML or database)
config = {"device": 0, "fps": 30, "width": 640}
cam = UVCCamera.from_config(config)

# Factory for driver-string configs (UI dropdowns, YAML)
cam = create_camera("realsense", serial_number="12345678", fps=30)
```

### Robot Integration

```python
from physicalai.capture import RealSenseCamera

camera = RealSenseCamera(fps=30)
robot = SO101.from_config("robot.yaml")
model = InferenceModel.load("./exports/act_policy")

with robot, camera:
    while True:
        frame = camera.read_latest()
        robot_obs = robot.get_observation()
        obs = {
            "images": {"wrist": frame.data},
            "state": robot_obs["state"],
        }
        action = model(obs)
        robot.send_action(action)
```

**Inference pacing:** `read_latest()` may return the same frame if called faster than
the camera framerate. Use `frame.sequence` to detect duplicates. Whether to skip
redundant frames or run inference on every call (e.g., for action interpolation or
temporal ensembling) is an application-layer decision.

**Temporal alignment:** Both `Frame.timestamp` and robot `get_observation()["timestamp"]`
use `time.monotonic()`, enabling applications to measure observation-to-action latency or
detect excessive camera-robot skew. Formal multi-modal temporal alignment (e.g.,
interpolating 200Hz robot state to the camera frame timestamp) is a runtime observation
pipeline concern, not the camera library's responsibility. See
[architecture.md](../../architecture/architecture.md) for the planned observation pipeline.

### Async (FastAPI)

```python
from physicalai.capture import UVCCamera

camera = UVCCamera(device=0, fps=30)
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

| FrameSource API                               | physicalai.capture Equivalent                    | Status                                                |
| --------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------- |
| `FrameSourceFactory.create(driver, **params)` | `UVCCamera(...)` or `create_camera(driver, ...)` | Direct replacement                                    |
| `.connect()`                                  | `.connect()`                                     | Same                                                  |
| `.read()` → `(success, frame)`                | `.read()` → `Frame`                              | Returns `Frame` (raises on failure)                   |
| `.start_async()` + `.get_latest_frame()`      | `.read_latest()`                                 | Simplified to one call                                |
| `.stop()`                                     | (not needed)                                     | No separate start/stop; managed by connect/disconnect |
| `.disconnect()`                               | `.disconnect()`                                  | Same                                                  |
| `FrameSourceFactory.discover_devices(driver)` | `discover_all()` or `Camera.discover()`          | Direct replacement                                    |
| `.get_supported_formats()`                    | `FormatDiscoveryMixin.get_supported_formats()`   | Via mixin                                             |
| `.attach_processor()`                         | Removed                                          | Was broken in production (commented out)              |

### API Migration Map

**camera_worker.py**: sync read pattern:

```python
# Before (FrameSource)
source = FrameSourceFactory.create(driver, **params)
source.connect()
success, frame = source.read()
source.disconnect()

# After (physicalai.capture)
camera = create_camera(driver, **params)  # driver: "omnicamera", "v4l2", "realsense", etc.
camera.connect()
frame = camera.read()  # frame.data for the image
camera.disconnect()
```

**teleoperate_worker.py / inference_worker.py**: async latest-frame pattern:

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

**hardware.py**: device discovery:

```python
# Before (FrameSource)
devices = FrameSourceFactory.discover_devices(driver)

# After (physicalai.capture)
devices = discover_all()
# or: devices = RealSenseCamera.discover()
```

**camera.py**: format query:

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

1. **Build** `physicalai.capture` in parallel. No changes to existing application code
2. **Validate** feature parity against the checklist above
3. **Swap** in a single PR: replace FrameSource imports with `physicalai.capture` imports in all 6 backend files
4. **Remove** FrameSource dependency from `application/backend/pyproject.toml`

The application's existing retry logic (`CameraConnectionManager` with `tenacity`) stays in the application layer: error recovery is not the camera library's responsibility.

---

## Comparison with LeRobot

| Aspect           | physicalai.capture                                        | LeRobot cameras                                   |
| ---------------- | --------------------------------------------------------- | ------------------------------------------------- |
| Base class       | `Camera` ABC (flat)                                       | `Camera` ABC                                      |
| Read model       | 3-tier: `read()`, `read_latest()`, `async_read()`         | 3-tier: `read()`, `read_latest()`, `async_read()` |
| Frame type       | `Frame(data, timestamp, sequence)`                        | Raw `ndarray` (no metadata)                       |
| Lifecycle        | `connect(timeout)` / `disconnect()`                       | `connect()` / `disconnect()`                      |
| Multi-camera     | `read_cameras()` / `async_read_cameras()`                 | Manual sequential reads                           |
| Hardware support | UVCCamera (all platforms), RealSense, Basler, GenICam, IP | OpenCV, RealSense, (fewer industrial)             |
| Depth            | `DepthMixin` with `read_depth()` → `Frame(uint16)`        | Not built-in                                      |
| PTZ              | `PTZMixin`                                                | Not built-in                                      |
| Config           | `from_config()` + dataclass configs                       | Pydantic `CameraConfig`                           |
| Discovery        | `Camera.discover()` + `discover_all()`                    | Not built-in                                      |
| Factory          | Optional `create_camera()` convenience                    | Not applicable                                    |
| Sharing          | Explicit (pass instance)                                  | Explicit                                          |
| Iterator         | Not implemented (explicit `read()` preferred)             | `__iter__` / `__next__`                           |

We adopted LeRobot's three-tier read model and explicit `connect/disconnect` lifecycle. We add timestamped frames, depth/PTZ mixins, industrial camera support, device discovery, and multi-camera synchronization via `read_cameras()`.

---

## Logging

`physicalai.capture` uses `loguru` for logging. The library disables its output by
default via `logger.disable("physicalai.capture")`. Applications opt in with
`logger.enable("physicalai.capture")`.

| Level   | What gets logged                              |
| ------- | --------------------------------------------- |
| DEBUG   | Every frame captured (sequence, timestamp)    |
| INFO    | Connect/disconnect, camera parameters applied |
| WARNING | Frame drops (sequence gaps)                   |
| ERROR   | Capture failures, SDK exceptions, timeouts    |

The library **never** configures handlers or sets levels; that is the application's
responsibility. By default, nothing is printed.

---

## Testing Strategy

- **Unit tests**: A `FakeCamera(Camera)` subclass that returns pre-built frames is
  provided for testing application code without hardware. All coordination logic
  (`read_cameras()`, `discover_all()`, error paths, timeout behavior) is tested
  against `FakeCamera`.

- **Integration tests** (future): Each backend will have hardware-gated tests behind
  pytest markers (`@pytest.mark.uvc`, `@pytest.mark.realsense`, `@pytest.mark.basler`).
  These will verify connect/disconnect lifecycle, frame format correctness, and timeout
  behavior against real devices. Requires dedicated hardware runners: to be set up when
  hardware is available.

---

## Configuration Validation

Camera parameters (`fps`, `width`, `height`, `color_mode`) are validated at
`connect()` time, when hardware capabilities are known. If a requested parameter is
not supported by the hardware:

- The camera applies the **nearest supported value** (e.g., requested 60fps,
  hardware supports 30fps → runs at 30fps)
- A **warning is logged** with the requested vs. actual value
- The actual applied values are available via read-only properties after `connect()`:

```python
cam = UVCCamera(device=0, fps=60, width=1920, height=1080)
cam.connect()
print(cam.fps)     # 30  (hardware maximum)
print(cam.width)   # 1920
print(cam.height)  # 1080
# loguru WARNING: "Requested fps=60, device supports max 30. Using 30."
```

This best-effort approach avoids hard failures in config-driven workflows where
parameters may come from a YAML file written for different hardware. Applications
that require exact parameters should assert after `connect()`.

---

## Open Design Decisions

**1. Multi-Consumer Access**
If multiple consumers need the same camera, should the library provide a `CameraPool`? Or is passing the same instance sufficient? Current position: application layer concern. Revisit if multiple teams hit this need.

**2. Error Recovery**
Retry on transient failures (USB disconnect/reconnect) in the library, or leave to the application? The application backend already has `tenacity`-based retry in `CameraConnectionManager`. Current position: library raises, application retries.

**3. Device Exclusivity Enforcement**
Currently documented as undefined behavior (see [Device Exclusivity](#device-exclusivity)). Future work to enforce via per-subclass registry with `DeviceInUseError`.

**4. Performance Budgets**
Concrete benchmarks for `Frame` wrapper overhead and `read_cameras()` skew will be established after initial implementation. Target: zero-copy where possible, <5ms multi-camera skew on commodity hardware.

---

## References

- [Strategy](../../architecture/strategy.md) — Architecture vision and key decisions
- [Robot Interface Design](./robot-interface.md) — Robot interface specification
- [FrameSource Repository](https://github.com/ArendJanKramer/FrameSource) — Original camera library (reference only)
- [LeRobot Cameras](https://github.com/huggingface/lerobot/tree/main/src/lerobot/cameras) — LeRobot's camera interface (design inspiration)
