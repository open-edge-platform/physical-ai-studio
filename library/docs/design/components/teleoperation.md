# Teleoperation API

High-level API for robot teleoperation, enabling leader/follower control patterns for data collection and real-time robot operation.

---

## Overview

The Teleoperation API provides a unified interface for controlling robots through leader/follower semantics. It is part of the **physicalai-train** library and integrates with the broader physicalai runtime architecture.

**Key Features:**

- Leader/follower control patterns
- Session-based lifecycle management
- Safety primitives (e-stop, limits)
- Integration with data collection

---

## Quick Start

```python
from physicalai.robot import SO101
from physicalai.teleop import TeleopSession

# Configure leader/follower robots
leader = SO101.from_config("leader.yaml")
follower = SO101.from_config("follower.yaml")

# Run teleoperation session
with TeleopSession(leader=leader, follower=follower) as session:
    for step in session:
        # step contains observation from follower, action from leader
        print(f"Action: {step.action}, Observation: {step.observation}")
```

---

## Core Concepts

### Leader/Follower Pattern

Teleoperation uses a leader/follower pattern where:

- **Leader robot**: Controlled by human operator (provides actions)
- **Follower robot**: Executes actions from leader (provides observations)

```
Human Operator
      │
      ▼
┌─────────────┐
│   Leader    │──── action ────┐
│   Robot     │                │
└─────────────┘                │
                               ▼
                        ┌─────────────┐
                        │  Follower   │
                        │   Robot     │
                        └─────────────┘
                               │
                               ▼
                          observation
```

### Session Lifecycle

```python
class TeleopSession:
    """Manages teleoperation session lifecycle."""

    def __init__(
        self,
        leader: Robot,
        follower: Robot,
        *,
        rate_hz: float = 30.0,
        safety_limits: SafetyLimits | None = None,
    ) -> None:
        """Initialize teleoperation session.

        Args:
            leader: Leader robot (human-controlled)
            follower: Follower robot (action execution)
            rate_hz: Control loop frequency
            safety_limits: Optional safety constraints
        """
        ...

    def __enter__(self) -> Self:
        """Start session - connects robots."""
        ...

    def __exit__(self, *args) -> None:
        """End session - safely disconnects robots."""
        ...

    def __iter__(self) -> Iterator[TeleopStep]:
        """Iterate through teleoperation steps."""
        ...
```

### TeleopStep

Each iteration yields a `TeleopStep` containing:

```python
@dataclass
class TeleopStep:
    """Single step of teleoperation."""

    observation: dict[str, Any]  # From follower robot
    action: np.ndarray           # From leader robot
    timestamp: float             # Step timestamp
    step_index: int              # Step counter
```

---

## Safety Features

### Safety Limits

```python
@dataclass
class SafetyLimits:
    """Safety constraints for teleoperation."""

    action_min: np.ndarray | float = -1.0
    action_max: np.ndarray | float = 1.0
    velocity_limit: float | None = None
    workspace_bounds: tuple[np.ndarray, np.ndarray] | None = None
```

### Emergency Stop

```python
with TeleopSession(leader=leader, follower=follower) as session:
    for step in session:
        if dangerous_condition:
            session.emergency_stop()
            break
```

---

## Integration with Data Collection

Teleoperation integrates seamlessly with the Data Collection API:

```python
from physicalai.teleop import TeleopSession
from physicalai.data import DatasetWriter

with TeleopSession(leader=leader, follower=follower) as session:
    with DatasetWriter(path="./dataset", format="lerobot") as writer:
        for step in session:
            writer.add_frame(step.observation, action=step.action)
        writer.save_episode(task="pick_and_place")
```

---

## Configuration

### YAML Configuration

```yaml
# teleop.yaml
leader:
  type: so101
  port: /dev/ttyUSB0
  role: leader

follower:
  type: so101
  port: /dev/ttyUSB1
  role: follower

session:
  rate_hz: 30.0
  safety_limits:
    action_min: -1.0
    action_max: 1.0
    velocity_limit: 0.5
```

### Loading from Config

```python
from physicalai.teleop import TeleopSession

session = TeleopSession.from_config("teleop.yaml")
with session:
    for step in session:
        ...
```

---

## Related Documentation

- **[Strategy](../strategy.md)** - Big-picture architecture
- **[Robot Interface Design](./robot-interface.md)** - Robot interface specification
- **[Data Collection API](./data-collection.md)** - Dataset collection design

---

_Last Updated: 2026-02-05_
