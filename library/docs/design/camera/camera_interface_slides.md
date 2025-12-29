---
marp: true
theme: default
paginate: true
footer: "Camera Interface Design | getiaction.cameras"
---

# Camera Interface Design

## Unified frame acquisition for getiaction

Live cameras, video files, and image folders — one interface

**Key features:** Invisible sharing, callbacks, capability mixins

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
| Manual multi-consumer       | Invisible sharing (automatic)      |
| Pull-only frame access      | Pull (`read()`) + push (callbacks) |
| No config support           | `from_config()` for YAML/dataclass |
| Manual resource management  | Context manager + ref-counting     |
| All-or-nothing features     | Composable capability mixins       |

**Goal**: Keep FrameSource's strengths, add the polish for production

---

## Agenda

1. **Design** — Principles, Class Hierarchy, Camera ABC
2. **Core** — Invisible Sharing, Callbacks, Mixins
3. **Implementations** — Webcam, RealSense, IPCam (with PTZ)
4. **Usage** — Basic, Multi-Consumer, Callbacks, Robot Integration
5. **Decisions** — Package Structure, Dependencies, Open Questions

---

<!-- _header: "1. Design" -->

## Design Principles

| Principle             | Description                                        |
| --------------------- | -------------------------------------------------- |
| **Hparams-first**     | Explicit constructor args with IDE autocomplete    |
| **Context manager**   | Safe resource management with `with` statement     |
| **Single ABC**        | One `Camera` interface for all sources             |
| **Invisible sharing** | Multiple instances share same device automatically |
| **Callback-driven**   | Push-based frame delivery (Lightning-inspired)     |
| **Capability mixins** | Optional features via composable mixins            |

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
└── ImageFolder         # Recorded: image sequences
```

---

<!-- _header: "1. Design" -->

## Camera ABC — Interface

```python
class Camera(ABC):
    def __init__(self, *, width, height, fps, color_mode, callbacks): ...

    @property
    def device_key(self) -> str: ...  # Unique ID for sharing
    @property
    def is_shared(self) -> bool: ...  # Other instances using this?

    def connect(self) -> None: ...    # Shares if device already open
    def disconnect(self) -> None: ... # Closes only when last user
    def read(self) -> NDArray: ...    # Pull-based

    def add_callback(self, cb: Callback) -> None: ...  # Push-based
```

---

<!-- _header: "2. Core" -->

## Callback System (Lightning-inspired)

Push-based frame delivery — override only hooks you need

```python
class Callback:
    def on_connect(self, camera: Camera) -> None: ...
    def on_disconnect(self, camera: Camera) -> None: ...
    def on_frame(self, camera: Camera, frame: NDArray) -> None: ...
    def on_error(self, camera: Camera, error: Exception) -> None: ...
```

**Usage:**

```python
class RecordingCallback(Callback):
    def on_frame(self, camera, frame):
        self.writer.write(frame)

camera = Webcam(index=0, callbacks=[RecordingCallback("out.mp4")])
```

---

<!-- _header: "2. Core" -->

## Multi-Consumer Challenge

Real-world usage requires multiple consumers on same camera:

- **UI display** — Live preview
- **Recording** — Save to disk
- **Teleoperation** — Remote control
- **WebSocket/WebRTC** — Network streaming

**Problem**: Physical camera can only be opened once!

**Solution**: Invisible sharing with reference counting

---

<!-- _header: "2. Core" -->

## Invisible Sharing

Multiple Camera instances for same device share automatically:

```python
cam_ui = Webcam(index=0)
cam_record = Webcam(index=0)  # Same device

cam_ui.connect()     # Opens device
cam_record.connect() # Reuses existing capture

print(cam_ui.is_shared)  # True

cam_ui.disconnect()     # Device stays open
cam_record.disconnect() # Device closes (last user)
```

Similar to `logging.getLogger()` — same name returns shared instance

---

<!-- _header: "3. Implementations" -->

## Capability Mixins

Optional features via composable mixins:

```python
class PTZMixin:
    supports_ptz: ClassVar[bool] = True  # Auto-set
    def pan(self, degrees: float) -> None: ...
    def tilt(self, degrees: float) -> None: ...
    def zoom(self, level: float) -> None: ...

class ColorControlMixin:
    supports_color_control: ClassVar[bool] = True
    def get_brightness(self) -> float: ...
    def set_brightness(self, value: float) -> None: ...

class ResolutionDiscoveryMixin:
    supports_resolution_discovery: ClassVar[bool] = True
    def get_supported_formats(self) -> list[Format]: ...
```

---

<!-- _header: "3. Implementations" -->

## IPCam with PTZ (Mixin Example)

```python
class IPCam(Camera, PTZMixin):
    """Network camera with PTZ support."""
    ...

cam = IPCam(url="rtsp://192.168.1.100/stream")
with cam:
    print(cam.supports_ptz)  # True (from PTZMixin)

    cam.pan(45)   # Pan 45 degrees
    cam.tilt(-10) # Tilt down
    cam.zoom(2.0) # Zoom level 2x

    frame = cam.read()
```

Capability flags are set automatically by mixin inheritance

---

<!-- _header: "4. Usage" -->

## Callback-Based Processing

```python
class FrameProcessor(Callback):
    def on_frame(self, camera, frame):
        result = model.predict(frame)
        display(result)

class MetricsCallback(Callback):
    def on_frame(self, camera, frame):
        self.frame_count += 1
        self.log_fps()

# Multiple callbacks, composable
camera = Webcam(
    index=0,
    callbacks=[FrameProcessor(), MetricsCallback()]
)
with camera:
    time.sleep(10)  # Frames pushed to callbacks automatically
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
    while True:
        action = policy.select_action({
            "images": {"wrist": camera.read()},
            "state": robot.get_state(),
        })
        robot.send_action(action)
```

Pull-based `read()` for control loops, callbacks for streaming

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
| **B: Standalone** | `from geticam import …`            | Reusable, but new repo       |
| **C: Keep fork**  | `from framesource import …`        | Minimal effort, no branding  |

**Recommendation**: Start with **A**, design for extraction to **B**

**Branding**: Need unique identity (`geti-camera`? `geti-vision`? `geticam`?)

---

<!-- _header: "5. Decisions" -->

## Other Open Decisions

| Decision            | Options                                 | Recommendation   |
| ------------------- | --------------------------------------- | ---------------- |
| **LeRobot interop** | In cameras vs. lerobot module           | ✅ In cameras     |
| **Transforms**      | Built-in vs. callable hook vs. pipeline | ✅ Built-in first |

### Additional Questions

- Calibration data property?
- `CameraRecorder` for saving frames?
- `Camera.find_all()` for device discovery?

---

<!-- _header: "5. Decisions" -->

## Implementation Plan

| Phase | Deliverables                                      |
| ----- | ------------------------------------------------- |
| **1** | `Camera` ABC, `_Capture`, `Callback`, `Webcam`    |
| **2** | Capability mixins (PTZ, ColorControl, Resolution) |
| **3** | `VideoFile`, `ImageFolder`, `RealSense`           |
| **4** | `IPCam` (with PTZ), `Basler`, `LeRobot`           |
| **5** | YAML config, documentation, tests                 |

---

<!-- _class: lead -->
<!-- _header: "" -->
<!-- _footer: "" -->

## Questions?

📄 Full reference: [camera_interface_design.md](camera_interface_design.md)
📄 Lifecycle details: [camera_lifecycle_design.md](camera_lifecycle_design.md)
