# Robot Interface Design

## Executive Summary

This document proposes adding a `Robot` interface to the physicalai runtime to enable programmatic robot control for inference and deployment. Today, robot interaction is only available through the Application backend. Users wanting to run trained policies on real robots must either run the full Application stack or build custom glue.

The proposed design:

- Adds a **framework-agnostic** `Robot` abstract base class to the library
- Uses standard Python types (`dict`, `np.ndarray`) at the core for maximum portability
- Keeps the core package inference‑first and dependency‑free
- Wraps vendor SDKs with thin adapters (optional extras)
- Shares the interface between Library and Application

This enables a simple deployment workflow: `pip install physicalai[robots]`, then run inference on real robots with ~10 lines of Python.

---

## Background

### Framework Landscape

physicalai provides the inference core, export pipeline, and runtime orchestration. What's missing: a library‑level robot hardware interface that can be used without the Application.

### Current Architecture

The system has two packages:

| Package                                | Purpose                             | Target Users                                                  |
| -------------------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| **Library** (`pip install physicalai`) | Inference, export, deployment       | ML researchers, robotics engineers                            |
| **Application** (Studio)               | Data collection, teleoperation, GUI | Subject matter experts such as Lab operators, non-programmers |

The library handles inference and deployment. The application handles human interaction. Robot control currently exists only in the Application, tightly coupled to its backend.

### The Gap

A robotics engineer who exports a policy to ONNX/OpenVINO cannot easily deploy it:

| Current Options         | Problem             |
| ----------------------- | ------------------- |
| Run Application backend | Requires web server |
| Write custom glue code  | Duplicates effort   |

---

## Design Principles

### Framework Agnosticism

The Robot interface must be usable in any inference runtime:

- **Core inference loops** - Plain Python dicts and numpy arrays
- **ROS/ROS2** - Standard message types
- **Custom pipelines** - Plain Python dicts and numpy arrays

This is achieved through:

1. **Core interface uses standard types** (`dict`, `np.ndarray`) - no framework imports required
2. **Optional adapters** live outside the core package (no circular dependencies)
3. **Lazy imports** for vendor SDKs and adapters

### Decoupled Camera Handling

The Robot ABC does **not** depend on Camera types. Instead:

- Robot ABC accepts camera configurations as `dict[str, dict[str, Any]]`
- Robot implementations (e.g., SO101) accept **both** config dicts and Camera objects
- Internal normalization converts Camera objects to config dicts

This ensures the base interface remains portable while providing convenience for adapter packages.

### Multi-Robot vs Multi-Arm

This design distinguishes **multiple robots** from **multi-arm robots**:

- **Multiple robots**: use **multiple `Robot` instances**. Each robot has its own connection lifecycle and produces its own observation/action space. Coordination happens at the application level (e.g., teleoperation leader/follower or a fleet controller).
- **Multi-arm robot**: use **one `Robot` subclass** that internally manages multiple hardware connections but exposes a single observation/action interface.

This keeps the API simple while allowing explicit composition where needed.

### Multi-Robot vs Multi-Arm

This design distinguishes **multiple robots** from **multi-arm robots**:

- **Multiple robots**: use **multiple `Robot` instances**. Each robot has its own connection lifecycle and produces its own observation/action space. Coordination happens at the application level (e.g., teleoperation leader/follower or a fleet controller).
- **Multi-arm robot**: use **one `Robot` subclass** that internally manages multiple hardware connections but exposes a single observation/action interface.

This keeps the API simple while allowing explicit composition where needed.

---

## Proposed Design

We design a `Robot` interface in the library, following the same patterns as our policy interface, where we could have both first party robot wrappers and third party robot integrations via vendor SDKs.

### Target Workflow

```bash
pip install physicalai[robots]
```

```python
from physicalai.inference import InferenceModel
from physicalai.robot import SO101

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
# physical_ai/robots/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np

class Robot(ABC):
    """Abstract interface for robot hardware.

    Framework-agnostic by default, with optional format conversion.

    Design Principles:
        - Core methods use standard Python types (dict, np.ndarray)
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

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable identifier for this robot instance.

        Used for logging, telemetry, and routing in multi-robot scenarios.
        Must be unique within an application process.
        """

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable identifier for this robot instance.

        Used for logging, telemetry, and routing in multi-robot scenarios.
        Must be unique within an application process.
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

    def get_observation(self) -> dict[str, Any]:
        """Read current robot state.

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
        ...

    # === Timing & Synchronization ===

    def get_timestamp(self) -> float:
        """Return the last observation timestamp in monotonic seconds.

        Composite robots should return a synchronized timestamp for the
        aggregated observation (e.g., max of per-arm capture times).
        """
        raise NotImplementedError

    # === Timing & Synchronization ===

    def get_timestamp(self) -> float:
        """Return the last observation timestamp in monotonic seconds.

        Composite robots should return a synchronized timestamp for the
        aggregated observation (e.g., max of per-arm capture times).
        """
        raise NotImplementedError

    # === Context Manager ===

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()
```

### Observation Conversion (Adapters)

The core API only guarantees a plain dict. Framework-specific formats must be provided by **adapter packages** outside the core to avoid circular dependencies. Adapters can expose helper functions or wrapper classes that convert the dict format into framework-specific types.

### Wrapper Architecture

The design mirrors our policy wrappers:

| Layer             | Policies   | Robots           |
| ----------------- | ---------- | ---------------- |
| Universal wrapper | (optional) | `VendorRobot`    |
| Specific wrappers | (optional) | `SO101`, `Aloha` |
| External SDKs     | (varies)   | vendor SDKs      |

**Universal wrapper** provides flexibility:

```python
class VendorRobot(Robot):
    """Universal wrapper for a vendor robot family.

    Accepts robot_type + explicit kwargs for a vendor SDK.
    """

    def __init__(
        self,
        robot_type: str,
        *,
        id: str = "robot",
        cameras: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize vendor robot wrapper.

        Args:
            robot_type: Vendor-specific robot type.
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
# physical_ai/robots/vendor/so101.py
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physicalai.capture import Camera

class SO101(VendorRobot):
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

# Camera objects (native)
from physicalai.capture import OpenCVCamera, RealSenseCamera

robot = SO101(
    cameras={
        "top": OpenCVCamera(index=0, width=640),
        "wrist": RealSenseCamera(serial_number="123456"),
    }
)

# Mixed (both work together)
robot = SO101(
    cameras={
        "top": {"type": "webcam", "index": 0},
        "wrist": RealSenseCamera(serial_number="123456"),
    }
)
```

Internally, Camera objects are normalized to config dicts before passing to the underlying SDK. This maintains framework agnosticism at the base level while providing convenience for adapter packages.

**Camera naming guidance**: Use explicit, collision-free names in multi-arm setups (e.g., `wrist_left`, `wrist_right`, `overhead`). Avoid ambiguous names like `wrist` when multiple arms are present.

See [Camera Interface Design](./camera-interface.md) for the full Camera specification.

### Supported Robots

All implementations wrap pip-installable SDKs where available:

| Vendor           | SDK              | Installation                 |
| ---------------- | ---------------- | ---------------------------- |
| Universal Robots | `ur_rtde`        | `pip install ur_rtde`        |
| ABB              | `abb_librws`     | `pip install abb-librws`     |
| Franka (Panda)   | `frankx`         | `pip install frankx`         |
| KUKA             | `py-openshowvar` | `pip install py-openshowvar` |

No vendored code—thin wrappers only.

---

## Usage Patterns

### Pattern 1: Framework-Agnostic (Default)

```python
from physicalai.robot import SO101

robot = SO101.from_config("robot.yaml")

with robot:
    # Pure dict/numpy - works with any framework
    obs = robot.get_observation()
    action = my_policy(obs["images"], obs["state"])
    robot.send_action(action)
```

### Pattern 2: Framework Adapter (Optional)

```python
from physicalai.inference import InferenceModel
from physicalai.robot import SO101

policy = InferenceModel.load("./exports/act_policy")
robot = SO101.from_config("robot.yaml")

with robot:
    policy.reset()
    for _ in range(1000):
        obs = robot.get_observation()  # dict
        action = policy.select_action(obs)
        robot.send_action(action)
```

### Pattern 3: External Adapter (Optional)

```python
from physicalai.robot import SO101

robot = SO101.from_config("robot.yaml")

with robot:
    # Adapter-specific conversion handled outside core
    obs = robot.get_observation()
    action = external_policy.select_action(obs)
    robot.send_action(action)
```

### Pattern 4: CLI

```bash
physical-ai infer \
    --model ./exports/openvino \
    --robot so101 \
    --robot-config robot.yaml \
    --episodes 10
```

### Pattern 5: Application Integration

Application imports the same interface:

```python
# application/backend/src/workers/inference_worker.py
from physicalai.inference import InferenceModel
from physicalai.robot import Robot, SO101

class InferenceWorker:
    def __init__(self, robot: Robot, model_path: str):
        self.robot = robot
        self.policy = InferenceModel.load(model_path)
```

One interface, multiple usage patterns.

---

## File Structure

```text
library/src/physical_ai/
├── robots/                      # NEW
│   ├── __init__.py              # Public API exports
│   ├── base.py                  # Robot ABC
│   ├── vendor/                  # Vendor-wrapped robots
│   │   ├── __init__.py
│   │   ├── universal.py         # VendorRobot (universal)
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
ur = ["ur_rtde>=1.5.0"]
abb = ["abb-librws>=1.0.0"]
franka = ["frankx>=0.3.0"]
robots = ["ur_rtde", "abb-librws", "frankx"]
```

```bash
pip install physicalai                    # Core (no robot support)
pip install physicalai[ur]                # Universal Robots only
pip install physicalai[robots]            # All robots
```

---

## physicalai vs Application

| Component         | Library | Application |
| ----------------- | :-----: | :---------: |
| Robot ABC         |    ✓    |   imports   |
| Vendor robots     |    ✓    |   imports   |
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
    def get_observation(self) -> dict[str, Any]: ...
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

## Multi-Robot Composition

A common question: if you have two robot arms with a shared camera (e.g., Aloha's bimanual setup), is that one `Robot` or two?

**From the DL perspective, it is one robot.** A bimanual policy produces a single action vector spanning both arms and consumes a single observation dict with images from shared cameras plus joint states from both arms. The policy doesn't know or care that the hardware is two separate serial connections.

**The `Robot` ABC already supports this.** A concrete implementation like `Aloha` manages multiple hardware connections internally and exposes a unified observation/action interface:

```python
class Aloha(VendorRobot):
    """Bimanual Aloha robot — two arms, shared cameras, one interface."""

    def __init__(
        self,
        *,
        leader_port: str = "/dev/ttyUSB0",
        follower_port: str = "/dev/ttyUSB1",
        cameras: dict[str, dict[str, Any] | "Camera"] | None = None,
        **kwargs: Any,
    ) -> None:
        ...

    def get_observation(self) -> dict[str, Any]:
        # Reads from both arms + shared cameras
        # Returns single observation with combined state vector
        ...

    def send_action(self, action: np.ndarray) -> None:
        # Splits action vector and dispatches to both arms
        ...
```

No `RobotGroup` abstraction is needed. The composite pattern lives inside the concrete implementation. Each multi-arm setup is a single `Robot` subclass that:

1. Manages N hardware connections internally
2. Merges joint states into one `state` array in `get_observation()`
3. Splits the action vector and dispatches to each arm in `send_action()`
4. Shares cameras naturally (cameras are keyed by name, not by arm)

This keeps the interface simple — upstream code (policies, inference loops, application) always sees one `Robot` with one observation space and one action space.

**Multiple robots** (independent arms or separate cells) should be represented by **multiple `Robot` instances**, each with its own `id`, connection lifecycle, and observation/action space. Coordination is handled by the application or teleoperation layer, not by the `Robot` ABC.

### Multi-Arm Schema Example

Example schema for a bimanual robot with 7‑DoF per arm:

```python
# Observation (dict format)
{
    "images": {
        "wrist_left": np.ndarray,   # HWC uint8
        "wrist_right": np.ndarray,  # HWC uint8
        "overhead": np.ndarray,
    },
    "state": np.concatenate([q_left, q_right]),  # length 14
    "timestamp": t,
}

# Action
action = np.concatenate([u_left, u_right])  # length 14
```

**Ordering rule**: left‑arm first, right‑arm second, unless a robot defines a different explicit order in its documentation.

**Multiple robots** (independent arms or separate cells) should be represented by **multiple `Robot` instances**, each with its own `id`, connection lifecycle, and observation/action space. Coordination is handled by the application or teleoperation layer, not by the `Robot` ABC.

### Multi-Arm Schema Example

Example schema for a bimanual robot with 7‑DoF per arm:

```python
# Observation (dict format)
{
    "images": {
        "wrist_left": np.ndarray,   # HWC uint8
        "wrist_right": np.ndarray,  # HWC uint8
        "overhead": np.ndarray,
    },
    "state": np.concatenate([q_left, q_right]),  # length 14
    "timestamp": t,
}

# Action
action = np.concatenate([u_left, u_right])  # length 14
```

**Ordering rule**: left‑arm first, right‑arm second, unless a robot defines a different explicit order in its documentation.

---

## Design Rationale

### Why `format` Parameter Instead of Separate Methods?

| Approach                                       | Pros                                                     | Cons                                                |
| ---------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------- |
| `get_observation_dict()` / `get_observation()` | Explicit method names                                    | Two methods to maintain, unclear which is "primary" |
| `get_observation(format=...)`                  | Single method, extensible, consistent with `data_format` | Slightly more complex signature                     |

We chose the `format` parameter because:

1. **Consistency**: Matches existing `data_format` pattern in other modules
2. **Extensibility**: Easy to add new formats without new methods
3. **Single source of truth**: One method to document and maintain
4. **Default is framework-agnostic**: plain dict requires no imports

### Why Config Dicts for Cameras in Base Interface?

The Robot ABC uses `dict[str, dict[str, Any]]` for cameras because:

1. **No dependencies**: Base class has no Camera import
2. **SDK compatibility**: All robot SDKs accept dict-like configs
3. **Serialization**: Config dicts are YAML/JSON serializable

Robot implementations (SO101, etc.) accept **both** dicts and Camera objects for convenience, normalizing internally.

Note that camera connection parameters are intentionally camera-type-specific rather than collapsed into a generic `device_key`. Webcams use an integer `index` (device enumeration order), RealSense cameras use a string `serial` (hardware serial number), and IP cameras use a `url`. These are semantically different types serving different purposes — collapsing them into a single string would lose type safety and make the API less self-documenting. The dict-based config (`dict[str, Any]`) already accommodates this flexibility naturally, since each camera type defines its own connection parameters.

### Lifecycle Semantics for Composite Robots

For multi-arm robots, `connect()`/`disconnect()` must manage all hardware links. If any required link fails to connect, the call should raise an error and the robot should be left in a safe, disconnected state. `is_connected` should return `True` only when **all** required links are connected.

### Concurrency and Atomicity

`get_observation()` should return a **synchronized** snapshot across arms and cameras when possible. `send_action()` should apply the full action vector as a single logical step; if the underlying SDK cannot guarantee simultaneity, implementations should document their behavior explicitly.

### Lifecycle Semantics for Composite Robots

For multi-arm robots, `connect()`/`disconnect()` must manage all hardware links. If any required link fails to connect, the call should raise an error and the robot should be left in a safe, disconnected state. `is_connected` should return `True` only when **all** required links are connected.

### Concurrency and Atomicity

`get_observation()` should return a **synchronized** snapshot across arms and cameras when possible. `send_action()` should apply the full action vector as a single logical step; if the underlying SDK cannot guarantee simultaneity, implementations should document their behavior explicitly.

### Why numpy Instead of torch?

The core interface uses `np.ndarray` because:

1. **Universal**: numpy is a de-facto standard, available everywhere
2. **No GPU assumptions**: Works on any device
3. **SDK compatibility**: Robot SDKs expect numpy, not torch
4. **Conversion is cheap**: `torch.from_numpy()` / `.numpy()` are zero-copy

Users can convert to torch at the policy boundary if needed.

---

## References

- [Strategy](../strategy.md) - Big-picture architecture
- [Camera Interface Design](./camera-interface.md) - Detailed camera interface specification
- [Teleoperation API](./teleoperation.md) - Teleoperation design

---

_Last Updated: 2026-02-13_
