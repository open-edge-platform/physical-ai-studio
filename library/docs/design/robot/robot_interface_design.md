
# Robot Interface Design

## Executive Summary

This document proposes adding a `Robot` interface to the Geti Action library to enable programmatic robot control for policy deployment. Currently, robot interaction is only available through the Application (Studio) backend. Users wanting to run trained policies on real robots must either run the full Application stack or bypass Geti Action entirely by using LeRobot directly.

The proposed design:

- Adds a **framework-agnostic** `Robot` abstract base class to the library
- Uses standard Python types (`dict`, `np.ndarray`) at the core for maximum portability
- Provides format conversion via `format` parameter (consistent with `data_format` pattern)
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

| Package                                 | Purpose                             | Target Users                                                  |
| --------------------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| **Library** (`pip install getiaction`)  | Training, inference, export         | ML researchers, robotics engineers                            |
| **Application** (Studio)                | Data collection, teleoperation, GUI | Subject matter experts such as Lab operators, non-programmers |

The library handles model development. The application handles human interaction. Robot control currently exists only in the Application, tightly coupled to its backend.

### The Gap

A robotics engineer who trains a policy and exports to ONNX/OpenVINO cannot easily deploy it:

| Current Options         | Problem                                 |
| ----------------------- | --------------------------------------- |
| Run Application backend | Requires web server                     |
| Use LeRobot directly    | Bypasses Geti Action inference pipeline |
| Write custom glue code  | Duplicates effort                       |

---

## Design Principles

### Framework Agnosticism

The Robot interface must work seamlessly with:

- **getiaction** - Native `Observation` objects
- **LeRobot** - Flattened dict format with `observation.` prefixes
- **ROS/ROS2** - Standard message types
- **Custom pipelines** - Plain Python dicts and numpy arrays

This is achieved through:

1. **Core interface uses standard types** (`dict`, `np.ndarray`) - no framework imports required
2. **Format parameter** for output conversion - consistent with `data_format` pattern in DataModule
3. **Lazy imports** - getiaction types only imported when requested

### Decoupled Camera Handling

The Robot ABC does **not** depend on Camera types. Instead:

- Robot ABC accepts camera configurations as `dict[str, dict[str, Any]]`
- Robot implementations (e.g., SO101) accept **both** config dicts and Camera objects
- Internal normalization converts Camera objects to config dicts

This ensures the base interface remains portable while providing convenience for getiaction users.

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
        obs = robot.get_observation()  # Returns dict by default
        action = policy.select_action(obs)
        robot.send_action(action)
```

Library-as-building-blocks. No web server required.

### Robot ABC

```python
# getiaction/robots/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np

# Type alias for observation format
ObsFormat = Literal["dict", "getiaction", "lerobot"]


class Robot(ABC):
    """Abstract interface for robot hardware.

    Framework-agnostic by default, with optional format conversion.

    Design Principles:
        - Core methods use standard Python types (dict, np.ndarray)
        - Format parameter enables framework-specific output
        - Follows hparams-first design with from_config() classmethod
        - Context manager for safe resource management
    """

    # === Configuration ===

    @classmethod
    @abstractmethod
    def from_config(cls, config: str | Path | dict[str, Any]) -> Self:
        """Create robot from configuration.

        Args:
            config: Path to YAML/JSON file, or config dict.

        Returns:
            Configured Robot instance (not yet connected).
        """

    # === Connection Lifecycle ===

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to robot hardware."""

    @abstractmethod
    def disconnect(self) -> None:
        """Safely disconnect from robot."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Connection status."""

    # === Observation ===

    def get_observation(
        self,
        format: ObsFormat = "dict",  # noqa: A002
    ) -> dict[str, Any]:
        """Read current robot state.

        Args:
            format: Output format.
                - "dict" (default): Framework-agnostic dict
                - "getiaction": getiaction.data.Observation object
                - "lerobot": LeRobot-style flattened dict

        Returns:
            Observation in requested format. Default dict structure:
                {
                    "images": {"camera_name": np.ndarray (HWC, uint8)},
                    "state": np.ndarray (joint positions),
                    "timestamp": float (optional),
                }
        """
        ...

    # === Action ===

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        """Execute action on robot.

        Args:
            action: Joint positions/velocities as numpy array.
        """

    # === Context Manager ===

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()
```

### Observation Format Conversion

The `format` parameter follows the same pattern as `data_format` in `LeRobotDataModule`:

| Format        | Output Type                  | Use Case                           |
| ------------- | ---------------------------- | ---------------------------------- |
| `"dict"`      | `dict[str, Any]`             | Framework-agnostic (default)       |
| `"getiaction"`| `Observation`                | Native getiaction workflows        |
| `"lerobot"`   | `dict[str, Any]` (flattened) | LeRobot policy compatibility       |

```python
# Framework-agnostic (default)
obs = robot.get_observation()
obs["images"]["wrist"]  # np.ndarray

# getiaction native
obs = robot.get_observation(format="getiaction")
obs.images["wrist"]  # Works with Observation API

# LeRobot compatible
obs = robot.get_observation(format="lerobot")
obs["observation.images.wrist"]  # Flattened keys
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
        cameras: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize LeRobot robot wrapper.

        Args:
            robot_type: LeRobot robot type (e.g., "so101_follower").
            id: Robot identifier.
            cameras: Camera configurations as dicts.
            **kwargs: Additional robot-specific parameters.
        """
        ...

    @classmethod
    def from_config(cls, config: str | Path | dict[str, Any]) -> Self:
        ...

    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    @property
    def is_connected(self) -> bool:
        ...

    def send_action(self, action: np.ndarray) -> None:
        ...
```

**Specific wrappers** provide IDE autocomplete and accept Camera objects:

```python
# getiaction/robots/lerobot/so101.py
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from getiaction.cameras import Camera

class SO101(LeRobotRobot):
    """SO-101 robot with explicit parameters for IDE support."""

    def __init__(
        self,
        *,
        id: str = "so101",
        port: str = "/dev/ttyUSB0",
        cameras: dict[str, dict[str, Any] | "Camera"] | None = None,
        disable_torque_on_disconnect: bool = True,
        max_relative_target: float | dict[str, float] | None = None,
    ) -> None:
        """Initialize SO-101 robot.

        Args:
            id: Robot identifier.
            port: Serial port for robot connection.
            cameras: Camera configurations. Accepts:
                - Config dicts: {"top": {"type": "webcam", "index": 0}}
                - Camera objects: {"top": Webcam(index=0)}
                - Mixed: {"top": {...}, "wrist": RealSense(...)}
            disable_torque_on_disconnect: Whether to disable torque on disconnect.
            max_relative_target: Maximum relative target for joint positions.
        """
        ...
```

### Camera Integration

Robot implementations accept **both** config dicts and Camera objects:

```python
# Config dicts (framework agnostic)
robot = SO101(
    cameras={
        "top": {"type": "webcam", "index": 0, "width": 640},
        "wrist": {"type": "realsense", "serial": "123456"},
    }
)

# Camera objects (getiaction native)
from getiaction.cameras import Webcam, RealSense

robot = SO101(
    cameras={
        "top": Webcam(index=0, width=640),
        "wrist": RealSense(serial="123456"),
    }
)

# Mixed (both work together)
robot = SO101(
    cameras={
        "top": {"type": "webcam", "index": 0},
        "wrist": RealSense(serial="123456"),
    }
)
```

Internally, Camera objects are normalized to config dicts before passing to the underlying SDK. This maintains framework agnosticism at the base level while providing convenience for getiaction users.

See [Camera Interface Design](https://github.com/open-edge-platform/geti-action/tree/docs/design-docs/library/docs/design/camera/camera_interface_design.md) for the full Camera specification.

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

---

## Usage Patterns

### Pattern 1: Framework-Agnostic (Default)

```python
from getiaction.robots import SO101

robot = SO101.from_config("robot.yaml")

with robot:
    # Pure dict/numpy - works with any framework
    obs = robot.get_observation()  # format="dict" is default
    action = my_policy(obs["images"], obs["state"])
    robot.send_action(action)
```

### Pattern 2: getiaction Native

```python
from getiaction.inference import InferenceModel
from getiaction.robots import SO101

policy = InferenceModel.load("./exports/act_policy")
robot = SO101.from_config("robot.yaml")

with robot:
    policy.reset()
    for _ in range(1000):
        obs = robot.get_observation(format="getiaction")
        action = policy.select_action(obs)
        robot.send_action(action)
```

### Pattern 3: LeRobot Compatible

```python
from getiaction.robots import SO101

robot = SO101.from_config("robot.yaml")

with robot:
    # LeRobot-style flattened dict
    obs = robot.get_observation(format="lerobot")
    action = lerobot_policy.select_action(obs)
    robot.send_action(action.numpy())
```

### Pattern 4: CLI

```bash
getiaction infer \
    --model ./exports/openvino \
    --robot so101 \
    --robot-config robot.yaml \
    --episodes 10
```

### Pattern 5: Application Integration

Application imports the same interface:

```python
# application/backend/src/workers/inference_worker.py
from getiaction.inference import InferenceModel
from getiaction.robots import Robot, SO101

class InferenceWorker:
    def __init__(self, robot: Robot, model_path: str):
        self.robot = robot
        self.policy = InferenceModel.load(model_path)
```

One interface, multiple usage patterns.

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
    def get_observation(self, format: ObsFormat = "dict") -> dict[str, Any]: ...
    def send_action(self, action: np.ndarray) -> None: ...

    # Safety (optional, default raises NotImplementedError)
    def set_speed_scale(self, scale: float) -> None:
        """Set speed scaling 0.0-1.0."""
        raise NotImplementedError

    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        raise NotImplementedError

    def is_emergency_stopped(self) -> bool:
        """Check if robot is in emergency stop state."""
        raise NotImplementedError


class UR5e(Robot):
    def set_speed_scale(self, scale: float) -> None:
        self._rtde.setSpeedSlider(scale)  # Delegates to ur_rtde

    def emergency_stop(self) -> None:
        self._rtde.triggerProtectiveStop()
```

Core interface stays simple. Industrial features are opt-in. Alternatively, we can define a separate `IndustrialRobot` ABC if needed.

---

## Design Rationale

### Why `format` Parameter Instead of Separate Methods?

| Approach | Pros | Cons |
|----------|------|------|
| `get_observation_dict()` / `get_observation()` | Explicit method names | Two methods to maintain, unclear which is "primary" |
| `get_observation(format=...)` | Single method, extensible, consistent with `data_format` | Slightly more complex signature |

We chose the `format` parameter because:

1. **Consistency**: Matches existing `data_format` pattern in `LeRobotDataModule`
2. **Extensibility**: Easy to add new formats without new methods
3. **Single source of truth**: One method to document and maintain
4. **Default is framework-agnostic**: `format="dict"` requires no imports

### Why Config Dicts for Cameras in Base Interface?

The Robot ABC uses `dict[str, dict[str, Any]]` for cameras because:

1. **No dependencies**: Base class has no Camera import
2. **SDK compatibility**: All robot SDKs accept dict-like configs
3. **Serialization**: Config dicts are YAML/JSON serializable

Robot implementations (SO101, etc.) accept **both** dicts and Camera objects for convenience, normalizing internally.

### Why numpy Instead of torch?

The core interface uses `np.ndarray` because:

1. **Universal**: numpy is a de-facto standard, available everywhere
2. **No GPU assumptions**: Works on any device
3. **SDK compatibility**: Robot SDKs expect numpy, not torch
4. **Conversion is cheap**: `torch.from_numpy()` / `.numpy()` are zero-copy

Users can convert to torch at the policy boundary if needed.
