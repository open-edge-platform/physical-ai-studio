---
marp: true
theme: default
paginate: true
footer: "Camera Interface Design | getiaction.cameras"
---

# Camera Interface Design

## Unified frame acquisition for getiaction

Live cameras, video files, and image folders — one interface

---

## Background: FrameSource

**Existing work**: [FrameSource](https://github.com/olkham/FrameSource) — low-level camera integrations

✅ **What it does well**:

- Hardware abstraction (webcams, RealSense, Basler, GenICam, IP cameras)
- Threading and buffer management
- Solid foundational implementation

⚠️ **The challenge**:

- Fork of a previous employee's repo
- Developed without strict engineering standards
- Lacks consistent API, typing, tests, documentation

---

## What This Design Adds

| FrameSource (Low-level)     | This Design (High-level)           |
| --------------------------- | ---------------------------------- |
| Functional but inconsistent | Clean, typed API with IDE support  |
| Minimal error handling      | Production-level error management  |
| Limited documentation       | Fully documented with examples     |
| No config support           | `from_config()` for YAML/dataclass |
| Manual resource management  | Context manager pattern            |

**Goal**: Keep FrameSource's strengths, add the polish for production

---

## Agenda

1. **Design** — Principles, Class Hierarchy, Camera ABC
2. **Core** — AsyncCaptureMixin, Webcam
3. **Implementations** — RealSense, VideoFile, ImageFolder
4. **Usage** — Basic, Async, Multi-Camera, Robot Integration
5. **Decisions** — Package Structure, Dependencies, Open Questions

---

<!-- _header: "1. Design" -->

## Design Principles

| Principle           | Description                                     |
| ------------------- | ----------------------------------------------- |
| **Hparams-first**   | Explicit constructor args with IDE autocomplete |
| **Context manager** | Safe resource management with `with` statement  |
| **Single ABC**      | One `Camera` interface for all sources          |

```python
with Webcam(index=0, fps=30) as camera:
    frame = camera.read()
```

---

<!-- _header: "1. Design" -->

## Class Hierarchy

```text
Camera (ABC)
├── Webcam              # Webcam, USB cameras
├── RealSense           # Intel depth cameras
├── Basler              # Industrial (pypylon)
├── Genicam             # Generic industrial
├── IPCam               # Network cameras (RTSP/HTTP)
├── Screen              # Desktop capture
├── VideoFile           # Recorded: video files
├── ImageFolder         # Recorded: image sequences
└── LeRobot             # Wrapper for LeRobot
```

---

<!-- _header: "1. Design" -->

## Camera ABC — Interface

```python
class Camera(ABC):
    def __init__(self, *, width, height, color_mode): ...

    @classmethod
    def from_config(cls, config) -> Self: ...  # YAML/dict/dataclass

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read(self) -> NDArray[np.uint8]: ...

    def __enter__(self) -> Self: ...  # Context manager
    def __iter__(self) -> Self: ...   # Iterator support
```

---

<!-- _header: "2. Core" -->

## AsyncCaptureMixin (Optional)

Background capture for live cameras — **opt-in, not required**

```python
class AsyncCaptureMixin:
    def start_async(self) -> None: ...
    def stop_async(self) -> None: ...
    def async_read(self, timeout_ms=200) -> NDArray: ...
```

> ⚠️ FrameSource intentionally does not implement async. Validate need before adding.

**Usage (if needed):**

```python
class AsyncWebcam(AsyncCaptureMixin, Webcam): ...

with AsyncWebcam(index=0) as camera:
    camera.start_async()
    frame = camera.async_read()
```

---

<!-- _header: "2. Core" -->

## Webcam

USB cameras, built-in webcams, V4L2 devices

```python
class Webcam(Camera):  # Sync only by default
    def __init__(
        self, *,
        index: int = 0,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
        rotation: Rotation = Rotation.NONE,
        warmup_s: float = 1.0,
    ): ...
```

---

<!-- _header: "3. Implementations" -->

## RealSense

Intel depth cameras with optional depth stream

```python
class RealSense(Camera):  # Sync only by default
    def __init__(
        self, *,
        serial_number: str | None = None,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        use_depth: bool = False,
    ): ...

    def read(self) -> NDArray[np.uint8]: ...
    def read_depth(self) -> NDArray[np.uint16]: ...
```

---

<!-- _header: "3. Implementations" -->

## VideoFile & ImageFolder

Recorded camera data — same interface as live cameras

```python
# Video file
with VideoFile(path="recording.mp4", loop=True) as video:
    for frame in video:
        process(frame)

# Image folder
with ImageFolder(path="dataset/images/", pattern="*.png") as folder:
    for frame in folder:
        process(frame)
```

Video files and image folders are just recorded camera output

---

<!-- _header: "4. Usage" -->

## Multi-Camera Setup

```python
cameras = {
    "wrist": Webcam(index=0),
    "overhead": RealSense(serial_number="012345"),
}

for cam in cameras.values():
    cam.connect()
    cam.start_async()

# Capture from all cameras
frames = {name: cam.async_read() for name, cam in cameras.items()}
```

---

<!-- _header: "4. Usage" -->

## Robot Integration

```python
from getiaction.cameras import RealSense
from getiaction.robots import SO101
from getiaction.inference import InferenceModel

policy = InferenceModel.load("./exports/act_policy")
robot = SO101.from_config("robot.yaml")
camera = RealSense(fps=30)

with robot, camera:
    camera.start_async()
    while True:
        action = policy.select_action({
            "images": {"wrist": camera.async_read()},
            "state": robot.get_state(),
        })
        robot.send_action(action)
```

---

<!-- _header: "5. Decisions" -->

## Package Structure

```text
library/src/getiaction/cameras/
├── __init__.py         # Public API + aliases
├── base.py             # Camera ABC
├── async_mixin.py      # Background capture
├── webcam.py           # Webcam
├── realsense.py        # RealSense
├── basler.py           # Basler
├── video.py            # VideoFile
├── folder.py           # ImageFolder
└── lerobot.py          # LeRobot
```

---

<!-- _header: "5. Decisions" -->

## Dependencies

OpenCV is a base dependency. Optional extras for specialized hardware:

```bash
pip install getiaction[realsense]  # Intel RealSense
pip install getiaction[basler]     # Basler (pypylon)
pip install getiaction[genicam]    # GenICam (harvesters)
pip install getiaction[cameras]    # All camera deps
```

---

<!-- _header: "5. Decisions" -->

## Critical: Package vs. Subpackage

**This camera interface is needed across the Geti ecosystem:**

| Product        | Purpose                                 |
| -------------- | --------------------------------------- |
| `geti-action`  | Robotics / VLA policies                 |
| `geti-prompt`  | SAM3, prompt-based tasks                |
| `geti-inspect` | Anomaly detection                       |
| `geti-tune`    | Classification, detection, segmentation |

⚠️ **Team alignment needed on this decision**

---

<!-- _header: "5. Decisions" -->

## Packaging Options

| Option            | Import                             | Trade-off                    |
| ----------------- | ---------------------------------- | ---------------------------- |
| **A: Subpackage** | `from getiaction.cameras import …` | Fast, but tied to getiaction |
| **B: Standalone** | `from geti_camera import …`        | Reusable, but new repo       |
| **C: Keep fork**  | `from framesource import …`        | Minimal effort, no branding  |

**Recommendation**: Start with **A**, design for extraction to **B**

**Branding**: Need unique identity (`geti-camera`? `geti-vision`? `geticam`?)

---

<!-- _header: "5. Decisions" -->

## Other Open Decisions

| Decision            | Options                                 | Recommendation    |
| ------------------- | --------------------------------------- | ----------------- |
| **LeRobot interop** | In cameras vs. lerobot module           | ✅ In cameras     |
| **Transforms**      | Built-in vs. callable hook vs. pipeline | ✅ Built-in first |

### Additional Questions

- Calibration data property?
- `CameraRecorder` for saving frames?
- `Camera.find_all()` for device discovery?

---

<!-- _header: "5. Decisions" -->

## Implementation Plan

| Phase | Deliverables                                |
| ----- | ------------------------------------------- |
| **1** | `Camera` ABC, `AsyncCaptureMixin`, `Webcam` |
| **2** | `VideoFile`, `ImageFolder`                  |
| **3** | `RealSense`, `Basler`, `LeRobot`            |
| **4** | YAML config, documentation                  |

---

<!-- _class: lead -->
<!-- _header: "" -->
<!-- _footer: "" -->

## Questions?

📄 Full reference: [camera_interface_design.md](camera_interface_design.md)
