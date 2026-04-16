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

# Robot Interface Design
### Protocol-Based Approach for `physicalai`

---

## What We're Solving

`physicalai` is an **inference deployment** library.

The robot interface exists to:
1. **Read observations** from hardware
2. **Send actions** from a trained policy

---

## The Interface

Four methods. No base class.

```python
class Robot(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: np.ndarray) -> None: ...
```

Any class that implements these four methods is a valid robot.

No inheritance. No registration. No dependency on `physicalai`.

Duck typing: "if it walks like a duck and it quacks like a duck, then it must be a **duck**"

---

## Design Principles

- **Protocol, not inheritance** — structural typing via duck typing
- **Standard Python types** — `dict` + `np.ndarray`, no custom messages
- **Synchronous** — `async def` changes return type to `Coroutine`, breaking Protocol compatibility. A 10–50Hz control loop doesn't need `asyncio`
- **Cameras separate** — robot handles joints, cameras are independent devices
- **Validation via manifest** — policy declares expectations, runtime validates
- **Safe disconnect** — `disconnect()` must leave robot stationary

---

## Why Protocol Over ABC?

---

## ABC Approach — Requires Inheritance

```python
# Third party MUST import and subclass
from physicalai.robot import Robot

class MyRobot(Robot):
    def connect(self): ...
    def disconnect(self): ...
    def get_observation(self): ...
    def send_action(self, action): ...
```

- Forces a dependency on `physicalai`
- Existing robot implementations must be wrapped or refactored
- Multiple ABC inheritance → MRO conflicts

---

## Protocol Approach — Just Implement the Methods

```python
# No import. No inheritance. No dependency.
class MyRobot:
    def connect(self): ...
    def disconnect(self): ...
    def get_observation(self): ...
    def send_action(self, action): ...
```

- Works with **existing** robot drivers as-is (or via a simple adapter)
- No conflicts with other libraries or existing class hierarchies
- Zero coupling to `physicalai`

---

## Side-by-Side Comparison

| Concern | ABC | Protocol |
|---|---|---|
| Third-party robots | Must import and subclass | Just implement the methods |
| Multiple libraries | MRO conflicts possible | No inheritance, no conflicts |
| Testing | Must subclass to create mocks | `Mock()` works directly |
| Adding capabilities later | Breaks existing implementations | Add separate Protocol |
| Missing method error | Immediate at instantiation | At first call |

---

## The Trade-Off

**Protocol:** no `TypeError` if a method is missing at class definition time.

**Mitigations:**
- Conformance test suite catches it immediately when run
- `mypy` / `pyright` catch it statically
- Robot drivers are tested immediately — nobody writes one and leaves it untested

The trade-off: slightly later error detection in exchange for **zero coupling** and **ecosystem compatibility**.

---

## Usage

Cameras and robot state are read separately, then assembled into the observation:

```python
from physicalai.inference import InferenceModel
from physicalai.robot import connect
from physicalai.capture import OpenCVCamera
from robots import SO101

model = InferenceModel("./my_policy")
robot = SO101(port="/dev/ttyUSB0")
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

---

## Context Manager

Protocol can't provide default implementations.
We ship a `connect()` wrapper — same pattern as `open()` for files.
Guarantees `disconnect()` runs even when exceptions occur:

```python
@contextmanager
def connect(robot):
    robot.connect()
    try:
        yield robot
    finally:
        robot.disconnect()
```

Every robot gets safe lifecycle management without inheriting anything.


---

## Why `dict` + `numpy`?

**`numpy`** over torch:
- Universal — no GPU assumptions, expected by robot SDKs
- Zero-copy conversion: `torch.from_numpy()` / `.numpy()`
- Runtime library shouldn't require a deep learning framework

**`dict[str, Any]`** over dataclass/TypedDict:
- Observation structure varies per robot — rigid typing adds friction with no safety gain
- Works with any inference runtime, any framework
- No custom message types to learn (not protobuf, not ROS messages)
- The manifest validates shapes at runtime, which is where it matters

---

## Cameras — Separate from Robot

The robot handles **joints only**. Cameras are independent devices with their own lifecycle.

**Why?**
- A camera may be **shared** across robots (e.g., overhead cam in ALOHA setup)
- Robot state and camera frames may run at **different frequencies** (joints at 200Hz, images at 30Hz)
- Robot drivers stay simple — no camera logic in `connect()` / `disconnect()`
- You can read joints **without** cameras (calibration, debugging)

The user assembles the full observation dict. The combined approach (cameras inside `get_observation()`) is still possible when simplicity is preferred.

---

## Validation — No Robot Descriptor Needed

**Problem:** How do we know the robot is compatible with the policy?

**Old approach:** Robot describes itself via a descriptor format.
→ Third parties must learn the format. Two sources of truth. Config drift.

**Our approach:** The policy manifest is the single source of truth.

---

## Manifest as Source of Truth

The manifest declares expected hardware in dedicated `robots` and `cameras` sections:

```json
{
    "format": "policy_package",
    "version": "1.0",
    "robots": [{
        "name": "main", "type": "SO-100",
        "state":  { "shape": [6], "dtype": "float32",
            "order": ["shoulder_pan", "shoulder_lift",
                        "elbow_flex", "wrist_flex", "wrist_roll", "gripper"] },
      "action": { "shape": [6], "dtype": "float32",
            "order": ["shoulder_pan", "shoulder_lift",
              "elbow_flex", "wrist_flex", "wrist_roll", "gripper"] }
    }],
    "cameras": [
        { "name": "wrist", "shape": [3, 480, 640], "dtype": "uint8" }
    ]
}
```

`order` is optional but recommended — without it, tensor ordering is an implicit contract between training and deployment. Shapes validate but the policy may receive **scrambled input**.

Example: a bimanual robot where `[left, right]` vs `[right, left]` concatenation produces valid shapes with **wrong semantics**. `order` catches this at startup.

---

## Two-Stage Validation

**Stage 1 — Before connecting (no hardware needed):**
```python
model = InferenceModel("./my_policy")
print(model.expected_robot)
# [{'name': 'main', 'type': 'SO-100', 'state': {'shape': [6]}, ...}]
print(model.expected_cameras)
# [{'name': 'wrist', 'shape': [3, 480, 640], 'dtype': 'uint8'}]
```

**Stage 2 — First inference call (automatic):**
```python
action = model(obs)  # checks obs shapes + order against manifest
# Mismatch → IncompatibleInputError with clear message
```

Zero work for robot implementers. No config files. No descriptor format.

---

## Conformance Testing

Ship a test utility for robot implementers:

```python
from physicalai.robot.testing import check_robot_conformance

robot = MyCustomRobot(port="/dev/ttyUSB0")
check_robot_conformance(robot)
# ✓ connect/disconnect lifecycle
# ✓ observation structure (dict with state, timestamp)
# ✓ send_action accepts np.ndarray
# ✓ disconnect leaves robot stationary
```

---

## Extension Protocols

New capabilities = new Protocols. Existing code stays untouched:

```python
class SupportsSafety(Protocol):
    def set_speed_scale(self, scale: float) -> None: ...
    def emergency_stop(self) -> None: ...
    def is_emergency_stopped(self) -> bool: ...
```

Functions that need safety accept `SupportsSafety`. Functions that don't still accept `Robot`.

No existing implementation breaks when a new Protocol is introduced.

---

## Summary

| Decision | Choice |
|---|---|
| Interface | `Protocol` — structural typing, no inheritance |
| Data types | `dict[str, Any]` observations, `np.ndarray` actions |
| Context manager | `connect()` wrapper function |
| Safety | `disconnect()` must leave robot stationary |
| Cameras | Separate from robot; user assembles observation |
| Concurrency | Synchronous protocol, threads allowed internally |
| Validation | Policy manifest with `order` → validated on first observation |
| Built-in robots | Ships with supported hardware implementations |
| Third-party robots | Four methods. Zero dependency on `physicalai`. |



