---
marp: true
theme: default
paginate: true
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 40px;
  }
  h2 {
    font-size: 32px;
  }
  table {
    font-size: 20px;
  }
  pre {
    font-size: 18px;
  }
---

# `physicalai.capture`

### Camera Interface Design

---

## Why `physicalai.capture`?

We need a camera interface that fits our ecosystem:

- **Typed, metadata-rich frames** — timestamp + sequence number on every read
- **Three read modes** — blocking, latest-frame, and async for different workloads
- **Multi-camera sync** — parallel reads with skew reporting
- **Hardware-agnostic** — OpenCV, RealSense, Basler, GenICam, IP cameras
- **Depth and PTZ** — via composable mixins
- **Device discovery** — enumerate cameras across all backends
- **Zero coupling** — standalone subpackage, extractable later

Built on capture logic from FrameSource, rewritten under our naming and conventions.

---

## Architecture Overview

```text
physicalai/
└── capture/                    # Zero coupling to other subpackages
    ├── frame.py                # Frame dataclass
    ├── camera.py               # Camera ABC
    ├── sync.py                 # read_cameras(), async_read_cameras()
    ├── discovery.py            # DeviceInfo, discover_all()
    ├── errors.py               # Error hierarchy
    └── cameras/
        ├── opencv.py           # OpenCVCamera
        ├── realsense.py        # RealSenseCamera (+ DepthMixin)
        ├── basler.py           # BaslerCamera
        ├── genicam.py          # GenicamCamera
        └── ip.py               # IPCamera
```

Designed as standalone — zero coupling, extractable to its own repo later.

---

## Camera ABC

```python
class Camera(ABC):
    def __init__(self, *, width=None, height=None, fps=None,
                 color_mode=ColorMode.RGB): ...

    # Lifecycle
    def connect(self, timeout: float = 5.0) -> None: ...
    def disconnect(self) -> None: ...

    # Reading
    def read(self, timeout: float | None = None) -> Frame: ...
    def read_latest(self) -> Frame: ...
    async def async_read(self, timeout: float | None = None) -> Frame: ...

    # Context managers
    def __enter__(self) -> Self: ...
    async def __aenter__(self) -> Self: ...
```

**Key guarantee:** `connect()` blocks until the first frame arrives. After it returns, reads are guaranteed to succeed (barring hardware failure).

**Why?** Without this, every consumer must handle "no frame yet" (race condition) on the first `read_latest()` call. 

---

## Why ABC, Not Protocol?

The robot interface uses **Protocol** (structural typing). Why does the camera use **ABC**?

Because `Camera` carries **real shared implementation**:

```python
# These are not just signatures — they contain shared logic
connect(timeout)     # Timeout handling, first-frame guarantee
disconnect()         # Executor cleanup, registry cleanup (future)
async_read()         # run_in_executor with per-camera executor
__enter__ / __exit__       # Calls connect/disconnect
__aenter__ / __aexit__     # Async connect via default executor
```

A Protocol can only define signatures. Every backend would duplicate this logic.

**Rule of thumb:** Protocol for consumers (type hints). ABC for implementors (shared logic).

---

## Camera Capability Mixins

Optional features are added via **composable mixins**, not by bloating the base `Camera` ABC.

| Mixin | Adds | Used by |
|---|---|---|
| `DepthMixin` | `read_depth()`, `read_rgbd()` | `RealSenseCamera` |
| `PTZMixin` | `pan()`, `tilt()`, `zoom()` | `IPCamera` |
| `FormatDiscoveryMixin` | `get_supported_formats()` | `BaslerCamera` |

```python
class RealSenseCamera(DepthMixin, Camera):
    def read_depth(self) -> Frame:    # uint16, millimeter depth
        ...
    def read_rgbd(self) -> tuple[Frame, Frame]:
        return self.read(), self.read_depth()
```

**Why mixins?** Not every camera has depth or PTZ. Adding these to the ABC would force every backend to raise `NotImplementedError`. Mixins keep the base interface clean and let `isinstance` checks discover capabilities.

---

## Why Frame Dataclass?

Every read returns a `Frame`, never a raw ndarray.

```python
@dataclass(frozen=True, slots=True)
class Frame:
    data: NDArray[np.uint8]    # (H, W, C) or (H, W)
    timestamp: float           # time.monotonic() at capture
    sequence: int              # Monotonic counter (0, 1, 2, ...)
```

**Why not raw `ndarray`?**
- `timestamp`: critical for multi-camera sync, latency measurement
- `sequence`: enables frame drop detection

**Why frozen dataclass, not TypedDict?**
- Frozen prevents accidental mutation of metadata

**Why `uint8` only?** Every robotics inference model (ACT, Diffusion Policy, VLAs) expects `uint8` RGB.

---

## Three-Tier Read Model

| Method          | Blocking | Skips Frames | Use Case                        |
| --------------- | -------- | ------------ | ------------------------------- |
| `read()`        | Yes      | No           | Recording, data collection      |
| `read_latest()` | No       | Yes          | Inference, teleoperation        |
| `async_read()`  | Yields   | No           | FastAPI, asyncio event loops    |

```python
# Recording: every frame matters
with OpenCVCamera(index=0) as cam:
    for i in range(100):
        frame = cam.read(timeout=5.0)
        save_frame(frame.data, frame.timestamp)

# Inference: freshness matters
with OpenCVCamera(index=0) as cam:
    while running:
        frame = cam.read_latest()
        action = model({"images": {"wrist": frame.data}})
```

Inspired by LeRobot's camera interface

---

## Why Lazy Executor for async_read()?

Camera SDKs are blocking. Calling `read()` from a coroutine freezes the event loop.

**Solution:** Per-camera `ThreadPoolExecutor(max_workers=1)`

```python
async def async_read(self, timeout=None) -> Frame:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(self._executor, self.read, timeout)
```

**Why lazy (created on first `async_read()`, not in `__init__`)?**
Sync-only usage is the common case. Creating a thread pool in `__init__` penalizes every user — even those who never call `async_read()`.

**Why per-camera, not a shared pool?**
A shared pool lets camera reads contend with each other and with unrelated I/O. One slow camera blocks reads from all others.

---

## Multi-Camera Sync

Sequential reads introduce ~33ms skew per camera. Parallel reads: ~1-3ms.

```python
synced = read_cameras(
    {"wrist": cam1, "overhead": cam2},
    latest=True,    # True for inference, False for recording
)

print(synced.max_skew_ms)                # 1.2
wrist_img = synced.frames["wrist"].data
overhead_img = synced.frames["overhead"].data
```

```python
@dataclass(frozen=True)
class SyncedFrames:
    frames: dict[str, Frame]
    max_skew_ms: float         # max(timestamps) - min(timestamps) in ms
```

`max_skew_ms` makes sync quality **observable**, not assumed. Applications decide their tolerance (10ms for inference, <1ms needs hardware sync).

---

## Cameras ≠ Robot

Cameras are managed **separately** from the robot interface.

**Why?**
- A camera may be **shared** across robots (overhead cam in ALOHA setup)
- Robot state and images run at **different frequencies** (joints at 200Hz, images at 30Hz)
- Robot drivers stay simple — no camera logic in `connect()` / `disconnect()`
- You can read joints **without** cameras (calibration, debugging)

```python
with robot, camera:
    frame = camera.read_latest()
    robot_obs = robot.get_observation()
    obs = {
        "images": {"wrist": frame.data},
        "state": robot_obs["state"],
    }
    action = model(obs)
    robot.send_action(action)
```

The user assembles the observation. Both timestamps use `time.monotonic()`.

---

## Multimodality

Multimodal sensors don't need a shared base class. Each sensor type has:
- Different read semantics (`read_latest()` for cameras, blocking `read()` for force/torque)
- Different return types (`Frame`, `Wrench`, `IMUReading`)
- Different configuration (resolution/fps vs sample rate vs calibration)

The model always needs to know which sensor is which:

```python
obs = {
    "images": {"wrist": frame.data, "overhead": overhead.data},
    "force_torque": wrench,
    "imu": imu_data,
    "state": robot_obs["state"],
}
action = model(obs)
```

**Design each sensor type independently. Extract shared patterns from real code, not speculation.**

---

## Future Extensibility

**Playback sources** (`VideoSource`, `ImageDirectorySource`) will **not** inherit from `Camera`.

`read_latest()` is meaningless for a video file. Forcing playback under `Camera` creates a leaky abstraction.

**Plan:** A lightweight `typing.Protocol` for the shared surface (`read()`, `connect()`, `disconnect()`) — introduced when playback sources actually ship. Non-breaking change.

**Lifecycle consistency across device types:** If needed, a `Device(Protocol)` can enforce `connect`/`disconnect`/`is_connected` naming across cameras, force/torque sensors, etc. Also a non-breaking addition — no hierarchy required.

---

## Migration from FrameSource

| FrameSource                               | physicalai.capture                         |
| ----------------------------------------- | ------------------------------------------ |
| `FrameSourceFactory.create(driver, ...)`  | `OpenCVCamera(...)` or `create_camera()`   |
| `.read()` → `(success, frame)`           | `.read()` → `Frame` (raises on failure)  |
| `.start_async()` + `.get_latest_frame()`  | `.read_latest()`                           |
| `.stop()`                                 | Not needed (managed by connect/disconnect) |
| `FrameSourceFactory.discover_devices()`   | `discover_all()` or `Camera.discover()`    |

**Plan:**
1. Build `physicalai.capture` in parallel (no changes to existing code)
2. Validate feature parity
3. Swap in single PR (6 backend files)
4. Remove FrameSource dependency

---

## Open Decisions

| Topic | Current Position | Revisit When |
|-------|-----------------|--------------|
| **Error recovery** | Library raises, app retries | Never |
| **Device exclusivity** | Undefined behavior | Implement with DeviceInUseError |
| **Performance budgets** | TBD | After initial implementation |

---

## Summary

| Decision | Choice | Why |
|---|---|---|
| Interface | ABC | Carries shared implementation (lifecycle, executor, context managers) |
| Multimodality | No shared `Sensor` base | Each sensor has different read semantics and return types; no polymorphic reads needed |
| Frame type | Frozen slotted dataclass | Immutable metadata, slots for memory, prints nicely |
| Read model | 3-tier: `read`, `read_latest`, `async_read` | Covers recording, real-time, and async use cases |
| Async executor | Lazy, per-camera | Sync users pay nothing; per-camera avoids contention |
| `disconnect()` | Template method (`_do_disconnect()`) | Base class owns executor cleanup; backends can't forget it |
| Cameras vs robot | Separate lifecycle | Different frequencies, shared cameras, simpler drivers |

---

## Discussion

Questions?
