# Robot Interface Design

## Overview

The robot interface defines how `physicalai` communicates with physical robots during inference deployment. It is deliberately minimal — the robot is plumbing in service of the inference loop, not the product itself.

The interface uses Python's `Protocol` for structural typing. Robot implementations do not inherit from a base class. They implement the required methods, and duck typing handles the rest.

### The Interface

Four methods. No base class.

```python
class Robot(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: np.ndarray) -> None: ...
```

Any class that implements these four methods is a valid robot. No inheritance, no registration, no dependency on `physicalai`.

### Usage

Cameras and robot state are read separately, then assembled into the observation the policy expects:

```python
from physicalai.inference import InferenceModel
from physicalai.robot import connect
from physicalai.capture import OpenCVCamera
from robots import SO100

model = InferenceModel("./my_policy")
robot = SO100(port="/dev/ttyUSB0")
camera = OpenCVCamera(index=0)

with connect(robot):
    robot_obs = robot.get_observation()
    frame = camera.read()

    obs = {
        "images": {"wrist": frame.data},
        "state": robot_obs["state"],
        "timestamp": robot_obs["timestamp"],
    }
    action = model(obs)
    robot.send_action(action)
```

A combined robot+camera approach is also possible. See [Cameras](#cameras) for trade-offs.

---

## Background

### Framework Landscape

physicalai provides the inference core, export pipeline, and runtime orchestration. What's missing: a library-level robot hardware interface that can be used without the Application.

### Current Architecture

The system has two packages:

| Package                                | Purpose                             | Target Users                                                  |
| -------------------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| **Library** (`pip install physicalai`) | Inference, deployment               | ML researchers, robotics engineers                            |
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

### Protocol, Not Inheritance

The interface uses `Protocol` (structural typing), not an Abstract Base Class. Implementations are plain classes that have the right methods. No base class to subclass, no import from `physicalai` required for third-party robots.

### Standard Python Types

Observations are `dict[str, Any]`, actions are `np.ndarray`. No custom message types, no protobuf, no ROS messages. This ensures maximum portability — works with any inference runtime, any framework.

`numpy` is used over torch because it is universally available, makes no GPU assumptions, is expected by robot SDKs, and conversion to/from torch is zero-copy (`torch.from_numpy()` / `.numpy()`).

### Synchronous

All methods are blocking. A robot control loop at 10–50Hz with 1–3 I/O sources does not benefit from `asyncio`. If an implementation needs async I/O internally (e.g., for cameras), it bridges to sync at the boundary. An `async def get_observation()` has a different return type (`Coroutine`) and will be flagged by `mypy` as incompatible with the Protocol.

### Validation via Manifest

The robot does not describe itself. The policy's `manifest.json` declares what it expects. The runtime validates observations against the manifest on first contact. See the [Validation](#validation) section below.

### Safe Disconnect

`disconnect()` must leave the robot in a safe, stationary state. Motors must be stopped or holding position before the connection is closed. This is a contractual requirement on every implementation, verified by the [conformance test suite](#conformance-testing).

### Rationale: Why Protocol

Protocols provide zero coupling for third parties (no import needed), avoid MRO conflicts when combining multiple robotics libraries, work directly with standard mocking tools, and support forward-compatible extension via separate Protocols (existing code stays untouched when new capabilities are added). The trade-off: no `TypeError` at class definition time if a method is missing — errors surface when the runtime calls the missing method. This is acceptable because the conformance test suite catches missing methods immediately, and static type checkers (`mypy`, `pyright`) flag Protocol violations before runtime.

---

## Protocol Definition

```python
# physicalai/robot/protocol.py
from typing import Any, Protocol
import numpy as np

class Robot(Protocol):
    """Structural interface for robot implementations.

    Any class that implements these four methods is a valid robot.
    No inheritance required. No registration required.
    """

    def connect(self) -> None:
        """Establish connection to the robot hardware.

        Called once before the inference loop begins. Must be idempotent —
        calling connect() on an already-connected robot should be a no-op
        or raise a clear error.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from the robot.

        Implementations MUST leave the robot in a safe, stationary state.
        Motors must be stopped or holding position before the connection
        is closed. This method is called automatically by the connect()
        context manager, including when exceptions occur.
        """
        ...

    def get_observation(self) -> dict[str, Any]:
        """Read the current robot state.

        Returns a dict with the following conventional structure:

            {
                "state": np.ndarray,            # joint positions, gripper, etc.
                "timestamp": float,             # time.monotonic() or equivalent
            }

        The exact keys and shapes must match what the policy expects,
        as declared in the policy's manifest.json under io.inputs.

        Note: cameras are managed separately from the robot interface.
        See the Cameras section for how to combine robot state and
        camera frames into a full observation for the policy.
        """
        ...

    def send_action(self, action: np.ndarray) -> None:
        """Send an action command to the robot.

        Args:
            action: A numpy array of joint commands. The shape and semantics
                    (positions, velocities, torques) depend on the policy
                    that produced the action. The robot implementation is
                    responsible for interpreting them correctly.
        """
        ...
```

---

## Context Manager

The Protocol cannot provide default method implementations. Instead, `physicalai` ships a `connect()` context manager (like `open()` for files):

```python
# physicalai/robot/utils.py
from contextlib import contextmanager

@contextmanager
def connect(robot):
    """Context manager for safe robot lifecycle.

    Calls robot.connect() on entry and robot.disconnect() on exit,
    including when exceptions occur.

    Usage:
        with connect(robot):
            obs = robot.get_observation()
            robot.send_action(action)
    """
    robot.connect()
    try:
        yield robot
    finally:
        robot.disconnect()
```

---

## Implementing a Robot

Adding a new robot is straightforward. Implement the four methods:

```python
# physicalai/robot/so100.py
import time
import numpy as np

class SO100:
    """Concrete implementation for the SO-100 robot arm."""

    JOINT_ORDER = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def __init__(self, port: str):
        self.port = port
        self._connection = None

    def connect(self) -> None:
        self._connection = serial.Serial(self.port, baudrate=1_000_000)

    def disconnect(self) -> None:
        # Hold current position (keep torque enabled) before closing.
        # This prevents the arm from dropping under gravity.
        # The servos will hold position until power is physically removed.
        #
        # Alternative: move to a known safe/home position before closing,
        # e.g. self._move_to_home_position(). This is safer if power
        # may be cut after disconnect, but takes time to execute.
        if self._connection:
            self._hold_position()
            self._connection.close()
            self._connection = None

    def get_observation(self) -> dict[str, Any]:
        return {
            "state": self._read_joint_positions(),
            "timestamp": time.monotonic(),
        }

    def send_action(self, action: np.ndarray) -> None:
        self._write_joint_positions(action)
```

No base class imported. No registration. No cameras to manage. The class satisfies the `Robot` protocol by having the right methods.

### Third-Party Robots

Third-party implementations follow the same pattern. They do not need to import anything from `physicalai`:

```python
# In a user's own code or separate package

class MyCustomRobot:
    def connect(self) -> None:
        # custom hardware setup
        ...

    def disconnect(self) -> None:
        # stop motors, close connection
        ...

    def get_observation(self) -> dict[str, Any]:
        return {
            "state": np.array([...]),
            "timestamp": time.monotonic(),
        }

    def send_action(self, action: np.ndarray) -> None:
        # send commands to hardware
        ...
```

This works with the `physicalai` runtime without modification:

```python
from physicalai.inference import InferenceModel
from physicalai.robots import connect
from my_package import MyCustomRobot

model = InferenceModel("./my_policy")
robot = MyCustomRobot(port="/dev/ttyUSB0")

with connect(robot):
    obs = robot.get_observation()
    action = model(obs)
    robot.send_action(action)
```

---

## Multi-Arm vs Multi-Robot

**Multi-arm** (e.g., bimanual robot): A single class with wider state and action vectors. Both arms are one robot.

```python
class BimanualRobot:
    def get_observation(self) -> dict[str, Any]:
        return {
            "state": np.concatenate([left_joints, right_joints]),  # (14,)
            "timestamp": time.monotonic(),
        }

    def send_action(self, action: np.ndarray) -> None:
        left_action = action[:7]
        right_action = action[7:]
        ...
```

From the policy's perspective, this is one robot with one observation space and one action space. The implementation manages multiple hardware connections internally. Camera naming should be explicit and collision-free (e.g., `wrist_left`, `wrist_right`, `overhead`).

**Ordering rule**: left-arm first, right-arm second, unless a robot defines a different explicit order in its documentation.

**Multiple independent robots**: Multiple instances, managed separately.

```python
left_robot = SO100(port="/dev/ttyUSB0")
right_robot = SO100(port="/dev/ttyUSB1")

with connect(left_robot), connect(right_robot):
    ...
```

Coordination between independent robots is handled at the application level (e.g., teleoperation leader/follower or a fleet controller), not by the robot interface.

---

## Cameras

Cameras are managed **separately** from the robot interface. The robot handles joint state and actuation. Cameras are independent devices with their own lifecycle.

### Why Separate?

- **Independent concerns.** You may want to read joint state without connecting cameras, e.g., for calibration or debugging.
- **Shared cameras.** A camera may be shared across multiple robots (e.g., an overhead camera used by two arms in an ALOHA setup).
- **Different frequencies.** Robot state and camera frames may need to be captured at different rates — joint state at 200Hz, images at 30Hz.
- **Simpler robot implementations.** Robot drivers only deal with motor communication. No camera logic to complicate `connect()` / `disconnect()`.

### Recommended Pattern: User Assembles the Observation

The user reads cameras and robot state separately, then assembles the observation dict that the policy expects:

```python
from physicalai.robots import connect

camera = OpenCVCamera(index=0)
robot = SO100(port="/dev/ttyUSB0")

with connect(robot):
    frame = camera.read()
    robot_obs = robot.get_observation()

    obs = {
        "images": {"wrist": frame.data},
        "state": robot_obs["state"],
        "timestamp": robot_obs["timestamp"],
    }
    action = model(obs)
    robot.send_action(action)
```

### Alternative: Combined Robot + Camera

The Protocol is deliberately open — nothing prevents an implementation from managing cameras internally:

```python
class RobotWithCameras:
    def __init__(self, cameras: dict):
        self.cameras = cameras
        ...

    def get_observation(self) -> dict[str, Any]:
        return {
            "images": {name: cam.read() for name, cam in self.cameras.items()},
            "state": self._read_joint_positions(),
            "timestamp": time.monotonic(),
        }
```

|                               | Separate (recommended)           | Combined                                |
| ----------------------------- | -------------------------------- | --------------------------------------- |
| Camera shared across robots   | Works naturally                  | Ambiguous ownership                     |
| Different capture frequencies | Easy — read each at its own rate | Locked to `get_observation()` call rate |
| Robot connect/disconnect      | Only motors                      | Must also manage camera lifecycle       |
| Simplicity for simple setups  | Slightly more user code          | Fewer lines in the loop                 |

We recommend the separate approach as the default. The combined approach is available for cases where simplicity in the loop is preferred and the trade-offs are acceptable.

See [Camera Interface Design](./camera-interface.md) for the full Camera specification.

---

## Validation

The robot does not describe its own capabilities. Instead, the policy's `manifest.json` declares what it expects, and the runtime validates against reality.

### Manifest as Source of Truth

The policy manifest declares expected hardware in dedicated `robot` and `camera` sections (see the [Manifest Format](inferencekit.md#manifest-format) for the full schema):

```json
{
    "format": "policy_package",
    "version": "1.0",
    "robots": [
        {
            "name": "main",
            "type": "SO-100",
            "state": {
                "shape": [6],
                "dtype": "float32",
                "joint_order": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper"
                ]
            },
            "action": {
                "shape": [6],
                "dtype": "float32",
                "joint_order": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper"
                ]
            }
        }
    ],
    "cameras": [
        {
            "name": "wrist",
            "shape": [3, 480, 640],
            "dtype": "uint8"
        },
        {
            "name": "top",
            "shape": [3, 480, 640],
            "dtype": "uint8"
        }
    ]
}
```

The `joint_order` field is optional but highly recommended. Without it, joint ordering is an implicit contract between training and deployment. If they disagree, shapes still validate but the policy can potentially receive scrambled input. This is especially dangerous for multi-arm robots where `[left, right]` vs `[right, left]` concatenation produces valid shapes with wrong semantics. When present, the runtime can compare `joint_order` against the robot's declared order and catch mismatches at startup. If absent, validation falls back to shape-only.

Note that `state` and `action` may have different `joint_order`, state can include extra sensor readings (e.g., force sensors) that are not actuated.

### Pre-Connection: Inspect Policy Requirements

Before connecting to any hardware, a user can inspect what the policy needs:

```python
model = InferenceModel("./my_policy")

print(model.expected_robot)
# [{'name': 'main', 'type': 'SO-100',
#   'state': {'shape': [6], 'dtype': 'float32'},
#   'action': {'shape': [6], 'dtype': 'float32'}}]

print(model.expected_cameras)
# [{'name': 'wrist', 'shape': [3, 480, 640], 'dtype': 'uint8'}]
```

### First-Contact: Automatic Validation

On the first call to `model(obs)`, the runtime validates observation shapes against the manifest:

```python
class InferenceModel:
    def __call__(self, inputs: dict) -> dict:
        if not self._validated:
            self._validate_inputs(inputs)
            self._validated = True
        return self._run(inputs)

    def _validate_inputs(self, inputs: dict) -> None:
        # Validate robot state
        for robot_spec in self._manifest["robots"]:
            name = robot_spec["name"]
            state_spec = robot_spec["state"]
            state = inputs.get("state")
            if state is None:
                raise IncompatibleInputError(
                    f"Policy expects 'state' for robot '{name}' "
                    f"but it was not found in the observation dict."
                )
            expected_shape = tuple(state_spec["shape"])
            if state.shape != expected_shape:
                raise IncompatibleInputError(
                    f"Policy expects 'state' with shape {expected_shape} "
                    f"(robot '{name}', type '{robot_spec.get('type', 'unknown')}') "
                    f"but got {state.shape}."
                )

        # Validate camera images
        for cam_spec in self._manifest.get("cameras", []):
            cam_name = cam_spec["name"]
            image = (inputs.get("images") or {}).get(cam_name)
            if image is None:
                raise IncompatibleInputError(
                    f"Policy expects camera '{cam_name}' but it was not "
                    f"found in observation['images']."
                )
            expected_shape = tuple(cam_spec["shape"])
            if image.shape != expected_shape:
                raise IncompatibleInputError(
                    f"Camera '{cam_name}' expected shape {expected_shape} "
                    f"but got {image.shape}."
                )
```

---

## Conformance Testing

`physicalai` ships a test utility that robot implementers can run against their implementation:

```python
# physicalai/robot/testing.py
import time
import numpy as np

def check_robot_conformance(robot, num_steps: int = 10):
    """Verify a robot implementation satisfies the Protocol contract.

    Checks:
    - connect/disconnect lifecycle
    - get_observation returns the expected dict structure
    - send_action accepts a numpy array
    - disconnect leaves the robot stationary
    """
    # Lifecycle
    robot.connect()

    # Observation structure
    obs = robot.get_observation()
    assert isinstance(obs, dict), "get_observation() must return a dict"
    assert "state" in obs, "observation must contain 'state'"
    assert isinstance(obs["state"], np.ndarray), "state must be np.ndarray"
    assert "timestamp" in obs, "observation must contain 'timestamp'"
    assert isinstance(obs["timestamp"], (int, float)), "timestamp must be numeric"

    if "images" in obs:
        assert isinstance(obs["images"], dict), "images must be a dict"
        for name, img in obs["images"].items():
            assert isinstance(img, np.ndarray), f"image '{name}' must be np.ndarray"
            assert img.ndim == 3, f"image '{name}' must be 3D (C, H, W)"

    # Action
    state_dim = obs["state"].shape[0]
    action = np.zeros(state_dim, dtype=np.float32)
    robot.send_action(action)

    # Safe disconnect
    robot.disconnect()
    robot.connect()
    obs1 = robot.get_observation()
    time.sleep(0.1)
    obs2 = robot.get_observation()
    assert np.allclose(obs1["state"], obs2["state"], atol=0.01), (
        "Robot must be stationary after disconnect(). "
        f"State changed from {obs1['state']} to {obs2['state']}"
    )
    robot.disconnect()

    print("All conformance checks passed.")
```

---

## Async and Concurrency

The Protocol is synchronous. All methods block until complete.

This is a deliberate choice. A robot control loop at 10–50Hz with 1–3 I/O sources does not benefit from `asyncio`. If an implementation needs internal concurrency (e.g., reading multiple sensors in parallel), it uses threads inside its own methods and returns synchronously at the boundary.

The sync Protocol enforces this at the type-checking level. An `async def get_observation()` has a different return type (`Coroutine`) and will be flagged by `mypy` as incompatible with the Protocol.

---

## Usage Patterns

### Pattern 1: Library (Default)

```python
from physicalai.inference import InferenceModel
from physicalai.robots import connect
from physicalai.robots import SO100

model = InferenceModel("./my_policy")
robot = SO100(port="/dev/ttyUSB0")

with connect(robot):
    obs = robot.get_observation()
    action = model(obs)
    robot.send_action(action)
```

### Pattern 2: CLI

The model's `manifest.json` provides robot type, state/action shapes, and camera requirements. The user supplies hardware connection details via `--port`/`--device` flags or a local config file (flags override config):

```bash
# Port flag
physical-ai infer \
    --model ./exports/act_policy \
    --port /dev/ttyUSB0 \
    --episodes 10

# Local config file
physical-ai infer \
    --model ./exports/act_policy \
    --hardware-config hardware.yaml \
    --episodes 10

# Flag overrides config
physical-ai infer \
    --model ./exports/act_policy \
    --hardware-config hardware.yaml \
    --port /dev/ttyUSB1 \
    --episodes 10
```

### Pattern 3: Application Integration

The Application imports the same interface:

```python
# application/backend/src/workers/inference_worker.py
from physicalai.inference import InferenceModel
from physicalai.robots import connect

class InferenceWorker:
    def __init__(self, robot, model_path: str):
        self.robot = robot
        self.policy = InferenceModel(model_path)

    def run_episode(self):
        with connect(self.robot):
            obs = self.robot.get_observation()
            action = self.policy(obs)
            self.robot.send_action(action)
```

One interface, multiple usage patterns.

---

## File Structure

```text
src/physicalai/
├── robot/
│   ├── __init__.py              # Public API exports
│   ├── protocol.py              # Robot Protocol definition
│   ├── utils.py                 # connect() context manager
│   ├── testing.py               # check_robot_conformance()
│   ├── so100.py                 # SO-100 implementation
│   └── ...                      # Other concrete implementations
└── ...
```

---

## Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "numpy>=1.24",
]
```

The core interface requires only numpy. Concrete robot implementations may have additional dependencies (e.g., `pyserial` for serial communication), installed as optional extras:

```bash
pip install physicalai                    # Core (no robot hardware support)
pip install physicalai[so100]             # SO-100 support
pip install physicalai[robots]            # All supported robots
```

---

## physicalai vs Application

| Component        | Library | Application |
| ---------------- | :-----: | :---------: |
| Robot Protocol   | ✓       | imports     |
| Concrete robots  | ✓       | imports     |
| Inference loop   | ✓       | uses        |
| Teleoperation    |         | ✓           |
| Recording/upload |         | ✓           |
| Calibration      |         | ✓           |
| GUI              |         | ✓           |

The library provides building blocks. The application provides workflows. Both share the same robot interface.

---

## Future: Extension Protocols

The Protocol approach makes future extensions clean. New capabilities are defined as separate Protocols — existing code stays untouched:

```python
class SupportsSafety(Protocol):
    def set_speed_scale(self, scale: float) -> None: ...
    def emergency_stop(self) -> None: ...
    def is_emergency_stopped(self) -> bool: ...

class SupportsIntrinsics(Protocol):
    def get_camera_intrinsics(self, name: str) -> np.ndarray: ...
```

Functions that need safety features accept `SupportsSafety`. Functions that don't still accept plain `Robot`. No existing implementation breaks when a new Protocol is introduced.

```python
class UR5e:
    # Satisfies Robot
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: np.ndarray) -> None: ...

    # Also satisfies SupportsSafety
    def set_speed_scale(self, scale: float) -> None:
        self._rtde.setSpeedSlider(scale)

    def emergency_stop(self) -> None:
        self._rtde.triggerProtectiveStop()

    def is_emergency_stopped(self) -> bool:
        return self._rtde.isProtectiveStopped()
```

---

## Summary

| Decision            | Choice                                                                             |
| ------------------- | ---------------------------------------------------------------------------------- |
| Interface mechanism | `Protocol` (structural typing, no inheritance)                                     |
| Data types          | `dict[str, Any]` for observations, `np.ndarray` for actions                        |
| Context manager     | `connect()` wrapper function                                                       |
| Safety              | `disconnect()` must leave robot stationary (documented contract, conformance test) |
| Cameras             | Separate from robot interface; user assembles observation                          |
| Concurrency         | Synchronous protocol, threads allowed internally                                   |
| Validation          | Policy manifest is source of truth, validated on first observation                 |
| Frequency control   | Runtime episode loop, target from manifest                                         |
| Built-in robots     | `physicalai` ships concrete implementations for supported hardware                 |
| Third-party robots  | Implement the four methods, no imports from `physicalai` required                  |

---

## References

- [Strategy](../architecture/strategy.md) — Big-picture architecture
- [Camera Interface Design](./camera-interface.md) — Camera interface specification
