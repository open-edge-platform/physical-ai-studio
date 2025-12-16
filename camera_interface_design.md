# Camera Interface Design

## Overview

A unified `Camera` interface for frame acquisition from live cameras, video files, and image folders. Video files and image folders are recorded camera output—the data originally came from a camera, we're just replaying it. One abstraction covers all cases.

**Module**: `getiaction.cameras`

**Design Principles:**

- **Hparams-first**: Explicit constructor args with IDE autocomplete, plus `from_config()` for configs
- **Context manager**: Safe resource management with `with` statement
- **Single ABC**: One `Camera` interface for all sources

---

## Class Hierarchy

```
Camera (ABC)
├── Webcam              # Webcam, USB cameras (alias: OpenCVCamera)
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
| `WebcamCapture`           | `Webcam`      | ✅           |
| `BaslerCapture`           | `Basler`      | ✅           |
| `GenicamCapture`          | `Genicam`     | ✅           |
| `RealsenseCapture`        | `RealSense`   | ✅           |
| `IPCameraCapture`         | `IPCam`       | ✅           |
| `ScreenCapture`           | `Screen`      | ✅           |
| `VideoFileCapture`        | `VideoFile`   | ✅           |
| `FolderCapture`           | `ImageFolder` | ✅           |
| `AudioSpectrogramCapture` | ❌            | Not a camera |

---

## Package Structure

```
library/src/getiaction/cameras/
├── __init__.py         # Public API + aliases
├── base.py             # Camera ABC
├── async_mixin.py      # Background capture
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

**Dependencies**: OpenCV is a base dependency. Optional extras for specialized hardware:

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

### Async Capture Mixin

For live cameras that benefit from background capture:

```python
from threading import Event, Lock, Thread
import numpy as np
from numpy.typing import NDArray


class AsyncCaptureMixin:
    """Mixin for background capture. Subclass must implement _capture_frame()."""

    def __init__(self) -> None:
        self._frame_lock = Lock()
        self._new_frame_event = Event()
        self._thread: Thread | None = None
        self._stop_event: Event | None = None
        self._latest_frame: NDArray[np.uint8] | None = None

    def _capture_frame(self) -> NDArray[np.uint8]:
        raise NotImplementedError

    def start_async(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event = Event()
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop_async(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _capture_loop(self) -> None:
        while self._stop_event and not self._stop_event.is_set():
            try:
                frame = self._capture_frame()
                with self._frame_lock:
                    self._latest_frame = frame
                self._new_frame_event.set()
            except Exception:
                pass

    def async_read(self, timeout_ms: float = 200.0) -> NDArray[np.uint8]:
        if not self._new_frame_event.wait(timeout=timeout_ms / 1000):
            raise TimeoutError(f"No frame within {timeout_ms}ms")
        self._new_frame_event.clear()
        with self._frame_lock:
            if self._latest_frame is None:
                raise RuntimeError("No frame available")
            return self._latest_frame.copy()
```

---

## Implementations

### Webcam

```python
from enum import Enum
from pathlib import Path
import cv2
import numpy as np
from numpy.typing import NDArray


class Rotation(int, Enum):
    NONE = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270


class Webcam(AsyncCaptureMixin, Camera):
    """Webcam or USB camera via OpenCV.

    Works with built-in laptop cameras, USB cameras, and any V4L2 device.
    Alias: OpenCVCamera (for LeRobot compatibility)
    """

    def __init__(
        self,
        *,
        index: int = 0,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
        rotation: Rotation = Rotation.NONE,
        warmup_s: float = 1.0,
    ) -> None:
        super().__init__(width=width, height=height, color_mode=color_mode)
        AsyncCaptureMixin.__init__(self)
        self.index = index
        self.fps = fps
        self.rotation = rotation
        self.warmup_s = warmup_s
        self._cap: cv2.VideoCapture | None = None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def connect(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to connect camera: {self.index}")

        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Warmup
        if self.warmup_s > 0:
            import time
            end = time.time() + self.warmup_s
            while time.time() < end:
                self._cap.read()

    def disconnect(self) -> None:
        self.stop_async()
        if self._cap:
            self._cap.release()
            self._cap = None

    def read(self) -> NDArray[np.uint8]:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return self._postprocess(frame)

    def _capture_frame(self) -> NDArray[np.uint8]:
        return self.read()

    def _postprocess(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if self.rotation != Rotation.NONE:
            rotate_map = {
                Rotation.ROTATE_90: cv2.ROTATE_90_CLOCKWISE,
                Rotation.ROTATE_180: cv2.ROTATE_180,
                Rotation.ROTATE_270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            frame = cv2.rotate(frame, rotate_map[self.rotation])

        if self.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.color_mode == ColorMode.GRAY:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame
```

### RealSense

```python
class RealSense(AsyncCaptureMixin, Camera):
    """Intel RealSense with optional depth."""

    def __init__(
        self,
        *,
        serial_number: str | None = None,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        color_mode: ColorMode = ColorMode.RGB,
        use_depth: bool = False,
    ) -> None:
        super().__init__(width=width, height=height, color_mode=color_mode)
        AsyncCaptureMixin.__init__(self)
        self.serial_number = serial_number
        self.fps = fps
        self.use_depth = use_depth
        self._pipeline = None

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    def connect(self) -> None:
        import pyrealsense2 as rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if self.serial_number:
            cfg.enable_device(self.serial_number)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        if self.use_depth:
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._pipeline.start(cfg)

    def disconnect(self) -> None:
        self.stop_async()
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

    def read(self) -> NDArray[np.uint8]:
        frames = self._pipeline.wait_for_frames()
        return np.asanyarray(frames.get_color_frame().get_data())

    def read_depth(self) -> NDArray[np.uint16]:
        frames = self._pipeline.wait_for_frames()
        return np.asanyarray(frames.get_depth_frame().get_data())

    def _capture_frame(self) -> NDArray[np.uint8]:
        return self.read()
```

### VideoFile (Recorded)

```python
class VideoFile(Camera):
    """Recorded camera data from video file."""

    def __init__(
        self,
        *,
        path: str | Path,
        loop: bool = False,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        super().__init__(width=width, height=height, color_mode=color_mode)
        self.path = Path(path)
        self.loop = loop
        self._cap: cv2.VideoCapture | None = None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def connect(self) -> None:
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open: {self.path}")

    def disconnect(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def read(self) -> NDArray[np.uint8]:
        ret, frame = self._cap.read()
        if not ret:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError("End of video")
        return self._postprocess(frame)

    def _postprocess(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if self.width and self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        if self.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
```

### ImageFolder (Recorded)

```python
class ImageFolder(Camera):
    """Recorded camera data from image folder."""

    def __init__(
        self,
        *,
        path: str | Path,
        pattern: str = "*.png",
        loop: bool = False,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        super().__init__(width=width, height=height, color_mode=color_mode)
        self.path = Path(path)
        self.pattern = pattern
        self.loop = loop
        self._files: list[Path] = []
        self._index = 0

    @property
    def is_connected(self) -> bool:
        return len(self._files) > 0

    def connect(self) -> None:
        self._files = sorted(self.path.glob(self.pattern))
        self._index = 0
        if not self._files:
            raise RuntimeError(f"No images in {self.path}")

    def disconnect(self) -> None:
        self._files = []
        self._index = 0

    def read(self) -> NDArray[np.uint8]:
        if self._index >= len(self._files):
            if self.loop:
                self._index = 0
            else:
                raise RuntimeError("End of images")

        frame = cv2.imread(str(self._files[self._index]))
        self._index += 1
        return self._postprocess(frame)

    def _postprocess(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if self.width and self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        if self.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
```

### LeRobot (Wrapper)

```python
class LeRobot(Camera):
    """Wrapper for LeRobot camera instances."""

    def __init__(self, lerobot_camera) -> None:
        super().__init__()
        self._camera = lerobot_camera

    @classmethod
    def from_lerobot_config(cls, config) -> "LeRobot":
        from lerobot.cameras.utils import make_cameras_from_configs
        cameras = make_cameras_from_configs({"cam": config})
        return cls(cameras["cam"])

    @property
    def is_connected(self) -> bool:
        return self._camera.is_connected

    def connect(self) -> None:
        self._camera.connect()

    def disconnect(self) -> None:
        self._camera.disconnect()

    def read(self):
        return self._camera.read()
```

---

## Usage

### Basic

```python
from getiaction.cameras import Webcam, VideoFile, ImageFolder

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

### Async Capture

```python
camera = Webcam(index=0)
with camera:
    camera.start_async()
    while running:
        frame = camera.async_read()  # Latest frame, non-blocking
```

### Multi-Camera

```python
cameras = {
    "wrist": Webcam(index=0),
    "overhead": RealSense(serial_number="012345"),
}

for cam in cameras.values():
    cam.connect()
    cam.start_async()

frames = {name: cam.async_read() for name, cam in cameras.items()}
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

### Aliases

```python
# In __init__.py - for LeRobot compatibility
from .webcam import Webcam
OpenCVCamera = Webcam  # Alias
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
    camera.start_async()
    while True:
        action = policy.select_action({
            "images": {"wrist": camera.async_read()},
            "state": robot.get_state(),
        })
        robot.send_action(action)
```

---

## Comparison: FrameSource vs. getiaction.cameras

| Aspect              | FrameSource (Old)         | getiaction.cameras (New)          |
| ------------------- | ------------------------- | --------------------------------- |
| Instantiation       | Factory with string types | Hparams-first constructors        |
| Configuration       | Kwargs dict               | Explicit params + `from_config()` |
| Async threading     | Duplicated per subclass   | `AsyncCaptureMixin`               |
| Resource management | Manual                    | Context manager (`with`)          |
| Error handling      | `(success, frame)` tuples | Exceptions                        |
| Dependencies        | All bundled               | Optional per camera type          |

---

## Open Design Decisions

### 1. Repository Location

| Option            | Location                          | Recommendation |
| ----------------- | --------------------------------- | -------------- |
| Inside getiaction | `library/src/getiaction/cameras/` | ✅ Start here  |
| Separate package  | Extract later if needed           | Defer          |

### 2. LeRobot Interop

| Option            | Location                        | Recommendation          |
| ----------------- | ------------------------------- | ----------------------- |
| In cameras        | `getiaction/cameras/lerobot.py` | ✅ All cameras together |
| In lerobot module | `getiaction/lerobot/cameras.py` | Groups interop code     |

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

### 4. Additional Questions

1. **Calibration data**: Add optional `calibration` property?
2. **Recording**: Add `CameraRecorder` for saving frames?
3. **Discovery**: Add `Camera.find_all()` for listing available devices?

---

## Implementation Plan

1. **Phase 1**: `Camera` ABC, `AsyncCaptureMixin`, `Webcam`
2. **Phase 2**: `VideoFile`, `ImageFolder`
3. **Phase 3**: `RealSense`, `Basler`, `LeRobot`
4. **Phase 4**: YAML config, documentation

---

## References

- [Robot Interface Design](robot_interface_design.md)
- [LeRobot Cameras](https://github.com/huggingface/lerobot/tree/main/src/lerobot/cameras)
