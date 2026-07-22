# Data Collection API

High-level API for collecting robot demonstration data, with support for multiple dataset formats and Hugging Face Hub integration.

---

## Overview

The Data Collection API provides a unified interface for recording robot episodes and saving them in standard formats. It is part of the **physicalai-train** library and integrates with the broader physicalai runtime architecture.

**Key Features:**

- Multiple format support (LeRobot, HDF5, custom)
- Frame-by-frame recording
- Episode management
- Hugging Face Hub upload
- Metadata and task annotations

---

## Quick Start

```python
from physicalai.robot import SO101
from physicalai.data import DatasetWriter

robot = SO101.from_config("robot.yaml")

with robot:
    with DatasetWriter(path="./dataset", robot=robot, format="lerobot") as writer:
        # Collect an episode
        for _ in range(100):
            obs = robot.get_observation()
            action = get_action_from_policy_or_teleop()
            robot.send_action(action)

            writer.add_frame(obs, action=action)

        # Save episode with metadata
        writer.save_episode(task="pick_and_place")

        # Finalize dataset
        writer.finalize()
```

---

## Core Concepts

### DatasetWriter

The main interface for data collection:

```python
class DatasetWriter:
    """Records robot episodes to dataset."""

    def __init__(
        self,
        path: str | Path,
        *,
        robot: Robot | None = None,
        format: str = "lerobot",
        fps: float = 30.0,
        overwrite: bool = False,
    ) -> None:
        """Initialize dataset writer.

        Args:
            path: Output directory for dataset
            robot: Robot instance (for metadata extraction)
            format: Dataset format ("lerobot", "hdf5", "custom")
            fps: Recording frame rate
            overwrite: Whether to overwrite existing data
        """
        ...

    def add_frame(
        self,
        observation: dict[str, Any],
        *,
        action: np.ndarray | None = None,
        reward: float | None = None,
    ) -> None:
        """Add single frame to current episode."""
        ...

    def save_episode(
        self,
        task: str,
        *,
        success: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Save current episode and return episode index."""
        ...

    def discard_episode(self) -> None:
        """Discard current episode without saving."""
        ...

    def finalize(self) -> None:
        """Finalize dataset (compute stats, write metadata)."""
        ...
```

### Supported Formats

| Format    | Description                    | Use Case                   |
| --------- | ------------------------------ | -------------------------- |
| `lerobot` | LeRobot-compatible HF datasets | LeRobot training pipelines |
| `hdf5`    | Single HDF5 file per episode   | Custom training pipelines  |
| `zarr`    | Zarr arrays (planned)          | Large-scale datasets       |

---

## Episode Management

### Recording Episodes

```python
with DatasetWriter(path="./dataset", format="lerobot") as writer:
    # Episode 1
    for frame in collect_episode_1():
        writer.add_frame(frame.observation, action=frame.action)
    writer.save_episode(task="grasp", success=True)

    # Episode 2
    for frame in collect_episode_2():
        writer.add_frame(frame.observation, action=frame.action)
    writer.save_episode(task="grasp", success=False)

    # Discard bad episode
    for frame in collect_episode_3():
        writer.add_frame(frame.observation, action=frame.action)
        if frame.is_corrupted:
            writer.discard_episode()
            break

    writer.finalize()
```

### Episode Metadata

```python
writer.save_episode(
    task="pick_and_place",
    success=True,
    metadata={
        "operator": "john_doe",
        "object": "red_cube",
        "lighting": "bright",
        "notes": "Clean grasp",
    }
)
```

---

## Hugging Face Hub Integration

### Upload to Hub

```python
from physicalai.data import DatasetWriter

writer = DatasetWriter(path="./dataset", format="lerobot")

# ... collect episodes ...

writer.finalize()

# Upload to Hugging Face Hub
writer.push_to_hub(
    repo_id="username/my-robot-dataset",
    private=True,
    tags=["robotics", "manipulation", "so101"],
)
```

### Download from Hub

```python
from physicalai.data import load_dataset

dataset = load_dataset("username/my-robot-dataset")
```

---

## Data Format: LeRobot

The LeRobot format stores data in Hugging Face Dataset format:

```
dataset/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   └── ...
├── videos/
│   ├── observation.images.top/
│   │   ├── episode_000000.mp4
│   │   └── ...
│   └── ...
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.json
│   └── episodes.json
└── README.md
```

### Schema

Each frame contains:

- `observation.images.*`: Camera images (stored as video)
- `observation.state`: Robot joint positions
- `action`: Commanded action
- `timestamp`: Frame timestamp
- `episode_index`: Episode identifier
- `frame_index`: Frame within episode

---

## Configuration

### YAML Configuration

```yaml
# data_collection.yaml
dataset:
  path: ./datasets/demo
  format: lerobot
  fps: 30.0

robot:
  type: so101
  port: /dev/ttyUSB0
  cameras:
    top:
      type: webcam
      index: 0
      width: 640
      height: 480

hub:
  repo_id: username/my-dataset
  private: true
```

### Loading from Config

```python
from physicalai.data import DatasetWriter

writer = DatasetWriter.from_config("data_collection.yaml")
```

---

## Integration with Teleoperation

```python
from physicalai.teleop import TeleopSession
from physicalai.data import DatasetWriter

with TeleopSession(leader=leader, follower=follower) as session:
    with DatasetWriter(path="./dataset", format="lerobot") as writer:
        for step in session:
            writer.add_frame(step.observation, action=step.action)

        writer.save_episode(task="demonstration")
        writer.finalize()
```

---

## Related Documentation

- **[Strategy](../strategy.md)** - Big-picture architecture
- **[Teleoperation API](./teleoperation.md)** - Teleoperation design
- **[Robot Interface Design](./robot-interface.md)** - Robot interface specification

---

_Last Updated: 2026-02-05_
