# Physical AI Studio Plugin

Types, protocols, and utilities for building robot catalog plugins for **Physical AI Studio**.

External robot types register themselves with Studio through an [entry-point](#entry-point-registration) mechanism, so they can be discovered, configured, and driven without modifying Studio's internal code.

---

## Installation

```bash
uv add physicalai-studio-plugin
```

Requires Python 3.12+. Dependencies are `pydantic>=2.12` and `physicalai`.

---

## Quick Start

A minimal plugin has this structure:

```
physicalai-my-robot-plugin/
├── pyproject.toml
├── README.md
└── src/
    └── physicalai_my_robot_plugin/
        ├── __init__.py
        └── studio_catalog.py
```

### `pyproject.toml`

```toml
[project]
name = "physicalai-my-robot-plugin"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "physicalai",
    "physicalai-studio-plugin",
]

[project.entry-points."physicalai.studio.catalog_plugins"]
my-robot = "physicalai_my_robot_plugin.studio_catalog:register_physicalai_studio_plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### `studio_catalog.py`

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from physicalai.robot.interface import Robot as PhysicalAIRobot
from physicalai_studio_plugin import (
    CatalogRobotFactory,
    PortScanner,
    RobotAdapterOptions,
    RobotAsset,
    RobotCatalogDefinition,
    RobotProbe,
    SerialPortInfo,
)
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from physicalai_studio_plugin import CatalogRobot


class MyRobotPayload(BaseModel):
    connection_string: str = ""
    serial_number: str = Field(...)


async def _build_my_robot(
    robot: CatalogRobot[MyRobotPayload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    port = await factory.find_port(
        SerialPortInfo(
            connection_string=robot.payload.connection_string or None,
            serial_number=robot.payload.serial_number or None,
        )
    )
    if port is None:
        msg = f"Robot not found: {robot.payload.serial_number}"
        raise RuntimeError(msg)
    # ... create and return your PhysicalAIRobot implementation ...


class MyRobotProbe:
    """Structurally implements RobotProbe[MyRobotPayload]."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        await manager.find_robots()
        return manager.robots

    async def identify(
        self, payload: MyRobotPayload, manager: PortScanner | None, joint: str | None = None
    ) -> None:
        pass

    async def is_online(
        self, payload: MyRobotPayload, manager: PortScanner | None = None
    ) -> bool:
        return True


def _definitions() -> list[RobotCatalogDefinition[MyRobotPayload]]:
    return [
        RobotCatalogDefinition[MyRobotPayload](
            type="MyRobot_Follower",
            display_name="My Robot Follower",
            role="follower",
            robot_builder=_build_my_robot,
            robot_payload=MyRobotPayload,
            asset=RobotAsset(
                urdf_relative_path=Path("my_robot/model.urdf"),
                packages={"my_robot": Path("my_robot")},
                joint_map={"gripper.pos": ["gripper"]},
                root_resolver=lambda: Path("/path/to/urdf"),
            ),
            adapter_options=RobotAdapterOptions(include_velocities=True),
            probe=MyRobotProbe(),
        ),
    ]


def register_physicalai_studio_plugin(registry: Any) -> None:
    for definition in _definitions():
        registry.register_robot(definition)
```

---

## API Reference

### `RobotCatalogDefinition`

The primary data class that describes a robot type to Studio. Generic over the payload model — use ``RobotCatalogDefinition[MyRobotPayload]`` to link the payload, probe, and robot builder types together.

```python
@dataclass
class RobotCatalogDefinition(Generic[_PayloadT]):
    type: str                        # Unique identifier, e.g. "MyRobot_Follower"
    display_name: str                # Human-readable name
    role: Literal["follower", "leader"]
    robot_builder: BuildRobotCallable | None = None
    robot_payload: type[_PayloadT] | None = None
    asset: RobotAsset | None = None
    adapter_options: RobotAdapterOptions = field(default_factory=RobotAdapterOptions)
    probe: RobotProbe[_PayloadT] | None = None
```

| Field | Description |
|-------|-------------|
| `type` | Stable identifier used in DB storage and API paths. Must be unique across all plugins. Convention: PascalCase with underscores, e.g. `"MyRobot_Follower"`. |
| `display_name` | Human-readable name shown in the Studio UI. |
| `role` | Either `"follower"` (executes actions) or `"leader"` (provides demonstrations). |
| `robot_builder` | Async callable that receives a robot payload and factory, returns a `PhysicalAIRobot` instance. |
| `robot_payload` | A Pydantic `BaseModel` subclass defining the configuration fields for this robot type (e.g. `serial_number`, `connection_string`). |
| `asset` | URDF and package maps for 3D visualization. |
| `adapter_options` | Controls velocity/effort forwarding behavior. |
| `probe` | Optional [`RobotProbe[_PayloadT]`](#robotprobe) typed to the same payload model. |

### `RobotAdapterOptions`

```python
@dataclass(frozen=True)
class RobotAdapterOptions:
    include_velocities: bool = False
    goal_time_scale: float = 1.0
    external_effort_gain: float | None = 0.1
```

### `RobotAsset`

```python
@dataclass(frozen=True)
class RobotAsset:
    urdf_relative_path: Path
    packages: dict[str, Path]
    joint_map: dict[str, list[str]]
    root_resolver: Callable[[], Path] | None = None
```

| Field | Description |
|-------|-------------|
| `urdf_relative_path` | Path to the URDF file, relative to the packages root. |
| `packages` | Maps ROS package names to their filesystem paths, e.g. `{"my_robot": Path("my_robot")}`. |
| `joint_map` | Maps Studio's observation key names (e.g. `"gripper.pos"`) to URDF joint name(s). |
| `root_resolver` | Callable that returns the root directory for URDF lookup. Used by Studio to resolve URDF paths for the API. |

### `RobotProbe`

```python
@runtime_checkable
class RobotProbe(Protocol[_PayloadT]):
    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]: ...
    async def identify(
        self, payload: _PayloadT, manager: PortScanner | None, joint: str | None = None
    ) -> None: ...
    async def is_online(
        self, payload: _PayloadT, manager: PortScanner | None = None
    ) -> bool: ...
```

Generic protocol over your robot's payload model. Implement it structurally — your class receives the typed payload directly instead of a raw dict. The ``_PayloadT`` type parameter is automatically inferred from ``identify`` / ``is_online`` signatures; you do not need to explicitly inherit from ``RobotProbe``.

### `PortScanner`

```python
class PortScanner(Protocol):
    async def find_robots(self) -> None: ...
    @property
    def robots(self) -> list[SerialPortInfo]: ...
```

Duck-type protocol for serial/network port scanners. Call `find_robots()` to refresh the device list, then read `robots` for the current results.

### `CatalogRobotFactory`

```python
class CatalogRobotFactory(Protocol):
    async def find_port(self, port_info: SerialPortInfo) -> str | None: ...
```

Factory protocol passed to your `robot_builder` callable. Use `find_port(SerialPortInfo(...))` to resolve a connection by serial number and/or configured connection string. Calibration data is now embedded in the robot payload model and does not require a factory method.

### `SerialPortInfo`

```python
class SerialPortInfo(BaseModel):
    connection_string: str | None
    serial_number: str | None
```

Describes a discovered serial or network connection.

### `PayloadContainer` / `CatalogRobot`

```python
class PayloadContainer(Protocol[_PayloadT]):
    payload: _PayloadT

class CatalogRobot(PayloadContainer[_PayloadT], Protocol[_PayloadT]):
    type: str
```

Protocols for the robot descriptor passed to `robot_builder`. The `payload` is an instance of your `robot_payload` model.

### `BuildRobotCallable`

```python
BuildRobotCallable = Callable[[_RobotT, _FactoryT], Awaitable[PhysicalAIRobot]]
```

Type alias for the `robot_builder` callable signature.

---

## Entry Point Registration

Studio discovers plugins via Python [entry points](https://packaging.python.org/en/latest/specifications/entry-points/) in the group `physicalai.studio.catalog_plugins`.

In your `pyproject.toml`:

```toml
[project.entry-points."physicalai.studio.catalog_plugins"]
my-robot = "physicalai_my_robot_plugin.studio_catalog:register_physicalai_studio_plugin"
```

The callable must accept a single argument — the registry — and call `registry.register_robot(definition)` for each robot type:

```python
def register_physicalai_studio_plugin(registry: Any) -> None:
    for definition in _definitions():
        registry.register_robot(definition)
```

Studio calls all discovered entry points at startup. Duplicate `type` values raise a `ValueError`.

---

## Robot Builder Pattern

The `robot_builder` is an async function that receives a robot descriptor and a factory, performs connection setup, and returns a `PhysicalAIRobot`:

```python
from physicalai_studio_plugin import CatalogRobot

async def _build_my_robot(
    robot: CatalogRobot[MyRobotPayload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    # `robot.payload` is already a validated MyRobotPayload instance.
    # 1. Resolve the connection
    port = await factory.find_port(
        SerialPortInfo(
            connection_string=robot.payload.connection_string or None,
            serial_number=robot.payload.serial_number or None,
        )
    )
    if port is None:
        msg = f"Robot not found: {robot.payload.serial_number}"
        raise RuntimeError(msg)

    # 2. Return the driver
    return MyRobotDriver(port=port, ...)
```

---

## Testing

Create a minimal test file alongside your plugin:

```python
from __future__ import annotations

from physicalai_studio_plugin import RobotCatalogDefinition, SerialPortInfo


def _fake_registry():
    class _FakeRegistry:
        def __init__(self):
            self.definitions: list[RobotCatalogDefinition] = []

        def register_robot(self, definition: RobotCatalogDefinition) -> None:
            self.definitions.append(definition)

    return _FakeRegistry()


def test_plugin_registration():
    from physicalai_my_robot_plugin.studio_catalog import (
        register_physicalai_studio_plugin,
    )

    registry = _fake_registry()
    register_physicalai_studio_plugin(registry)
    assert len(registry.definitions) == 1
    assert registry.definitions[0].type == "MyRobot_Follower"
```
