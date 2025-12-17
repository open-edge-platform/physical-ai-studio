# Robot Interface Design

## Executive Summary

This document proposes adding a `Robot` interface to the Geti Action library to enable programmatic robot control for policy deployment. Currently, robot interaction is only available through the Application (Studio) backend. Users wanting to run trained policies on real robots must either run the full Application stack or bypass Geti Action entirely by using LeRobot directly.

The proposed design:

- Adds a `Robot` abstract base class to the library
- Wraps existing SDKs (LeRobot, UR, ABB) with thin adapters
- Follows the same patterns as our policy wrappers (universal + specific)
- Shares the interface between Library and Application

This enables a simple deployment workflow: `pip install getiaction[robot]`, then run inference on real robots with ~10 lines of Python.

---

## Background

### Framework Landscape

**LeRobot** dominates robotics learning research with hardware drivers (SO-101, Aloha, Koch), teleoperation, and training. It owns the full pipeline from hardware to trained model.

**OpenPI** from Physical Intelligence provides foundation models (Pi0) with inference serving. It assumes you have your own robot stack—the brain, not the body.

**Isaac GR00T** from NVIDIA targets humanoid deployment with TensorRT optimization and Jetson support.

**Geti Action** provides multi-policy training (ACT, Diffusion, Pi0, SmolVLA, GR00T), Lightning integration, export pipeline (OpenVINO, ONNX), and a GUI Application. What's missing: robot hardware interface in the library.

### Current Architecture

Geti Action has two packages:

| Package                                | Purpose                             | Target Users                                                  |
| -------------------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| **Library** (`pip install getiaction`) | Training, inference, export         | ML researchers, robotics engineers                            |
| **Application** (Studio)               | Data collection, teleoperation, GUI | Subject matter experts such as Lab operators, non-programmers |

The library handles model development. The application handles human interaction. Robot control currently exists only in the Application, tightly coupled to its backend.

### The Gap

A robotics engineer who trains a policy and exports to ONNX/OpenVINO cannot easily deploy it:

| Current Options         | Problem                                 |
| ----------------------- | --------------------------------------- |
| Run Application backend | Requires web server                     |
| Use LeRobot directly    | Bypasses Geti Action inference pipeline |
| Write custom glue code  | Duplicates effort                       |

---

## Proposed Design

We design a `Robot` interface in the library, following the same patterns as our policy interface, where we could have both first party robot wrappers and third party robot integrations via LeRobot.

### Target Workflow

```bash
pip install getiaction[lerobot]
```

```python
from getiaction.inference import InferenceModel
from getiaction.robots import SO101

policy = InferenceModel.load("./exports/act_policy")
robot = SO101.from_config("robot.yaml")

with robot:
    policy.reset()
    while True:
        obs = robot.get_observation()
        action = policy.select_action(obs)
        robot.send_action(action)
```

Library-as-building-blocks. No web server required.

### Robot ABC

```python
# getiaction/robots/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from getiaction.data import Observation, Action


class Robot(ABC):
    """Abstract interface for robot hardware.

    Follows hparams-first design: explicit constructor args,
    with `.from_config()` classmethod for config-based instantiation.
    """

    @classmethod
    def from_config(cls, config: str | Path | dict) -> Self:
        """Create robot from config file path or config dict."""
        ...

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to robot hardware."""

    @abstractmethod
    def disconnect(self) -> None:
        """Safely disconnect from robot."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Read current state: images, joint positions."""

    @abstractmethod
    def send_action(self, action: Action) -> None:
        """Execute action on robot."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Connection status."""

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()
```

### Wrapper Architecture

The design mirrors our policy wrappers:

| Layer             | Policies           | Robots                      |
| ----------------- | ------------------ | --------------------------- |
| Universal wrapper | `LeRobotPolicy`    | `LeRobotRobot`              |
| Specific wrappers | `ACT`, `Diffusion` | `SO101`, `Aloha`            |
| External SDKs     | `lerobot.policies` | `lerobot.robots`, `ur_rtde` |

**Universal wrapper** provides flexibility:

```python
# getiaction/robots/lerobot/universal.py
class LeRobotRobot(Robot):
    """Universal wrapper for any LeRobot robot.

    Similar to LeRobotPolicy, accepts robot_type + explicit kwargs.
    """

    def __init__(
        self,
        robot_type: str,
        *,
        id: str = "robot",
        **kwargs,
    ) -> None:
        self._robot_type = robot_type
        self._id = id
        self._kwargs = kwargs
        self._robot = None

    def connect(self) -> None:
        config = make_robot_config(self._robot_type, id=self._id, **self._kwargs)
        self._robot = make_robot_from_config(config)
        self._robot.connect()

    def get_observation(self) -> Observation:
        raw = self._robot.get_observation()
        return Observation.from_dict(raw)

    def send_action(self, action: Action) -> None:
        self._robot.send_action(action.to_dict())
```

**Specific wrappers** provide IDE autocomplete:

```python
# getiaction/robots/lerobot/so101.py
class SO101(LeRobotRobot):
    """SO-101 robot with explicit parameters for IDE support."""

    def __init__(
        self,
        *,
        id: str = "so101",
        port: str = "/dev/ttyUSB0",
        cameras: dict[str, Camera] | None = None,
        disable_torque_on_disconnect: bool = True,
        max_relative_target: float | dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            robot_type="so101_follower",
            id=id,
            port=port,
            cameras=_convert_cameras(cameras),
            disable_torque_on_disconnect=disable_torque_on_disconnect,
            max_relative_target=max_relative_target,
        )
```

### Supported Robots

All implementations wrap pip-installable SDKs where available:

| Vendor                        | SDK              | Installation                 |
| ----------------------------- | ---------------- | ---------------------------- |
| LeRobot (SO-101, Aloha, Koch) | `lerobot`        | `pip install lerobot`        |
| Universal Robots              | `ur_rtde`        | `pip install ur_rtde`        |
| ABB                           | `abb_librws`     | `pip install abb-librws`     |
| Franka (Panda)                | `frankx`         | `pip install frankx`         |
| KUKA                          | `py-openshowvar` | `pip install py-openshowvar` |

**Note**: Trossen/Interbotix robots (ViperX, WidowX) can be supported via LeRobot, which wraps their Dynamixel-based hardware. As a permanent solution, we collaborate with Trossen to add native SDK support in the future.

No vendored code—thin wrappers only.

### Camera Pipeline (Future)

A unified `Camera` interface supporting industrial cameras is planned, potentially as a separate package within the Geti ecosystem:

| Vendor          | SDK             |
| --------------- | --------------- |
| OpenCV          | `opencv-python` |
| Intel RealSense | `pyrealsense2`  |
| Basler          | `pypylon`       |
| FLIR            | `PySpin`        |
| Allied Vision   | `VimbaPython`   |

Robot wrappers will accept this `Camera` type, converting internally to SDK-specific formats (e.g., LeRobot's `CameraConfig`). This decouples camera configuration from robot SDK specifics and enables reuse across the Geti ecosystem.

---

## Usage Patterns

### Pattern 1: Python Script

```python
from getiaction.inference import InferenceModel
from getiaction.robots import SO101

policy = InferenceModel.load("./exports/act_policy")
robot = SO101.from_config("robot.yaml")

with robot:
    policy.reset()
    for _ in range(1000):
        obs = robot.get_observation()
        action = policy.select_action(obs)
        robot.send_action(action)
```

### Pattern 2: CLI

```bash
getiaction infer \
    --model ./exports/openvino \
    --robot so101 \
    --robot-config robot.yaml \
    --episodes 10
```

### Pattern 3: Application

Application imports the same interface, and can use it as is within its pipeline, or extend it with additional functionality:

```python
# application/backend/src/workers/inference_worker.py
from getiaction.inference import InferenceModel
from getiaction.robots import Robot, SO101

class InferenceWorker:
    def __init__(self, robot: Robot, model_path: str):
        self.robot = robot
        self.policy = InferenceModel.load(model_path)
```

One interface, three usage patterns.

---

## File Structure

```
library/src/getiaction/
├── robots/                      # NEW
│   ├── __init__.py              # Public API exports
│   ├── base.py                  # Robot ABC
│   ├── lerobot/                 # LeRobot-wrapped robots
│   │   ├── __init__.py
│   │   ├── universal.py         # LeRobotRobot (universal)
│   │   ├── so101.py             # SO101 (explicit args)
│   │   ├── aloha.py             # Aloha (explicit args)
│   │   └── koch.py              # Koch (explicit args)
│   ├── ur/                      # Universal Robots
│   │   ├── __init__.py
│   │   └── ur5e.py
│   └── abb/                     # ABB
│       ├── __init__.py
│       └── irb.py
└── ...
```

---

## Dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
lerobot = ["lerobot>=0.1.0"]
ur = ["ur_rtde>=1.5.0"]
abb = ["abb-librws>=1.0.0"]
franka = ["frankx>=0.3.0"]
robots = ["lerobot", "ur_rtde", "abb-librws", "frankx"]
```

```bash
pip install getiaction              # Core (no robot support)
pip install getiaction[lerobot]     # LeRobot robots only
pip install getiaction[ur]          # Universal Robots only
pip install getiaction[robots]      # All robots
```

---

## Library vs Application

| Component         | Library | Application |
| ----------------- | :-----: | :---------: |
| Robot ABC         |    ✓    |   imports   |
| LeRobot robots    |    ✓    |   imports   |
| Industrial robots |    ✓    |   imports   |
| Inference loop    |    ✓    |    uses     |
| Teleoperation     |         |      ✓      |
| Recording/upload  |         |      ✓      |
| Calibration       |         |      ✓      |
| GUI               |         |      ✓      |

The library provides building blocks. The application provides workflows. Both share the same robot interface.

---

## Future: Industrial Extensions

Industrial robots have additional safety requirements. Vendor SDKs expose these methods, which can be optional in our interface:

```python
class Robot(ABC):
    # Core (required)
    def get_observation(self) -> Observation: ...
    def send_action(self, action: Action) -> None: ...

    # Safety (optional)
    def set_speed_scale(self, scale: float) -> None:
        """Set speed scaling 0.0-1.0."""
        raise NotImplementedError


class UR5e(Robot):
    def set_speed_scale(self, scale: float) -> None:
        self._rtde.setSpeedSlider(scale)  # Delegates to ur_rtde
```

Core interface stays simple. Industrial features are opt-in. Alternatively, we can define a separate `IndustrialRobot` ABC if needed.
