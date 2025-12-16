---
marp: true
theme: default
paginate: true
footer: "Camera Interface Design | getiaction.cameras"
---

# Camera Interface Design

**Unified frame acquisition for getiaction**

Live cameras, video files, and image folders — one interface

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

```
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

## AsyncCaptureMixin

Background capture for live cameras — no more dropped frames

```python
class AsyncCaptureMixin:
    def start_async(self) -> None: ...
    def stop_async(self) -> None: ...
    def async_read(self, timeout_ms=200) -> NDArray: ...
```

**Usage:**

```python
with Webcam(index=0) as camera:
    camera.start_async()
    while running:
        frame = camera.async_read()  # Latest frame
```

---

<!-- _header: "2. Core" -->

## Webcam

USB cameras, built-in webcams, V4L2 devices

```python
class Webcam(AsyncCaptureMixin, Camera):
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

_Alias: `OpenCVCamera` for LeRobot compatibility_

---

<!-- _header: "3. Implementations" -->

## RealSense

Intel depth cameras with optional depth stream

```python
class RealSense(AsyncCaptureMixin, Camera):
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

_Video files and image folders are just recorded camera output_

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

```
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

## Open Design Decisions

| Decision            | Options                                 | Recommendation           |
| ------------------- | --------------------------------------- | ------------------------ |
| **Location**        | Inside getiaction vs. separate package  | ✅ Inside, extract later |
| **LeRobot interop** | In cameras vs. lerobot module           | ✅ In cameras            |
| **Transforms**      | Built-in vs. callable hook vs. pipeline | ✅ Built-in first        |

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

# Questions?

📄 Full reference: [camera_interface_design.md](camera_interface_design.md)
