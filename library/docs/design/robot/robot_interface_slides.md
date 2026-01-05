---
marp: true
theme: default
paginate: true
footer: "Robot Interface Design | getiaction.robots"
---

# Robot Interface Design

## Programmatic robot control for getiaction

Train policies → Export models → Deploy on real robots

---

## Agenda

1. **Motivation** — The Gap, Target Workflow
2. **Design** — Principles, Robot ABC, Wrapper Architecture
3. **Implementation** — Universal & Specific Wrappers, Supported Robots
4. **Usage** — Python Script, CLI, Package Structure
5. **Roadmap** — Library vs Application, Industrial Extensions

---

<!-- _header: "1. Motivation" -->

## The Gap

A robotics engineer trains a policy and exports to OpenVINO...

| Current Options         | Problem                        |
| ----------------------- | ------------------------------ |
| Run Application backend | Requires web server            |
| Use LeRobot directly    | Bypasses Geti Action inference |
| Write custom glue code  | Duplicates effort              |

**Goal**: `pip install getiaction[robots]` + ~10 lines of Python

---

<!-- _header: "1. Motivation" -->

## Target Workflow

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

---

<!-- _header: "2. Design" -->

## Design Principles

| Principle            | Description                                     |
| -------------------- | ----------------------------------------------- |
| **Hparams-first**    | Explicit constructor args with IDE autocomplete |
| **Context manager**  | Safe resource management with `with` statement  |
| **Thin wrappers**    | No vendored code — wrap existing SDKs           |
| **Shared interface** | Same ABC used by Library and Application        |

---

<!-- _header: "2. Design" -->

## Robot ABC

```python
class Robot(ABC):
    @classmethod
    def from_config(cls, config: str | Path | dict) -> Self: ...

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def get_observation(self) -> Observation: ...
    def send_action(self, action: Action) -> None: ...

    @property
    def is_connected(self) -> bool: ...

    def __enter__(self) -> Self: ...  # Context manager
    def __exit__(self, *args) -> None: ...
```

---

<!-- _header: "2. Design" -->

## Wrapper Architecture

Mirrors our policy wrappers:

| Layer             | Policies           | Robots                      |
| ----------------- | ------------------ | --------------------------- |
| Universal wrapper | `LeRobotPolicy`    | `LeRobotRobot`              |
| Specific wrappers | `ACT`, `Diffusion` | `SO101`, `Aloha`            |
| External SDKs     | `lerobot.policies` | `lerobot.robots`, `ur_rtde` |

---

<!-- _header: "3. Implementation" -->

## Universal Wrapper

Flexibility for any LeRobot robot:

```python
class LeRobotRobot(Robot):
    def __init__(
        self,
        robot_type: str,
        *,
        id: str = "robot",
        **kwargs,
    ) -> None: ...

    def connect(self) -> None:
        config = make_robot_config(self._robot_type, **self._kwargs)
        self._robot = make_robot_from_config(config)
        self._robot.connect()
```

---

<!-- _header: "3. Implementation" -->

## Specific Wrapper

IDE autocomplete with explicit parameters:

```python
class SO101(LeRobotRobot):
    def __init__(
        self,
        *,
        id: str = "so101",
        port: str = "/dev/ttyUSB0",
        cameras: dict[str, Camera] | None = None,
        disable_torque_on_disconnect: bool = True,
        max_relative_target: float | None = None,
    ) -> None:
        super().__init__(robot_type="so101_follower", ...)
```

---

<!-- _header: "3. Implementation" -->

## Supported Robots

All implementations wrap pip-installable SDKs:

| Vendor                        | SDK              | Installation                 |
| ----------------------------- | ---------------- | ---------------------------- |
| LeRobot (SO-101, Aloha, Koch) | `lerobot`        | `pip install lerobot`        |
| Universal Robots              | `ur_rtde`        | `pip install ur_rtde`        |
| ABB                           | `abb_librws`     | `pip install abb-librws`     |
| Franka (Panda)                | `frankx`         | `pip install frankx`         |
| KUKA                          | `py-openshowvar` | `pip install py-openshowvar` |

## Trossen/Interbotix via LeRobot

---

<!-- _header: "4. Usage" -->

## Usage: Python Script

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

---

<!-- _header: "4. Usage" -->

## Usage: CLI

```bash
getiaction infer \
    --model ./exports/openvino \
    --robot so101 \
    --robot-config robot.yaml \
    --episodes 10
```

---

<!-- _header: "4. Usage" -->

## Package Structure

```bash
library/src/getiaction/robots/
├── __init__.py              # Public API exports
├── base.py                  # Robot ABC
├── lerobot/                 # LeRobot-wrapped robots
│   ├── universal.py         # LeRobotRobot
│   ├── so101.py             # SO101
│   ├── aloha.py             # Aloha
│   └── koch.py              # Koch
├── ur/                      # Universal Robots
│   └── ur5e.py
└── abb/                     # ABB
    └── irb.py
```

---

<!-- _header: "4. Usage" -->

## Dependencies

```bash
pip install getiaction              # Core (no robot support)
pip install getiaction[lerobot]     # LeRobot robots only
pip install getiaction[ur]          # Universal Robots only
pip install getiaction[robots]      # All robots
```

---

<!-- _header: "5. Roadmap" -->

## Library vs Application

| Component              | Library |   App   |
| ---------------------- | :-----: | :-----: |
| Robot ABC, Inference   |    ✓    | imports |
| LeRobot / Industrial   |    ✓    | imports |
| Teleop, Recording, GUI |         |    ✓    |

**Library** = blocks. **Application** = workflows.

---

<!-- _header: "5. Roadmap" -->

## Future: Industrial Extensions

Optional safety methods for industrial robots:

```python
class Robot(ABC):
    # Core (required)
    def get_observation(self) -> Observation: ...
    def send_action(self, action: Action) -> None: ...

    # Safety (optional)
    def set_speed_scale(self, scale: float) -> None:
        raise NotImplementedError

class UR5e(Robot):
    def set_speed_scale(self, scale: float) -> None:
        self._rtde.setSpeedSlider(scale)
```

---

<!-- _class: lead -->
<!-- _header: "" -->
<!-- _footer: "" -->

## Questions?

📄 Full reference: [robot_interface_design.md](robot_interface_design.md)
