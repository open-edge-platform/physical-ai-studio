# Robot Plugins

Robot plugins add robot to Physical AI Studio without requiring changes to
Studio itself. A plugin provides a Python entry point, one or more catalog
definitions, and the code needed to build and connect each robot.

This page covers installing and using plugins, then developing a plugin. For the
internal design and lifecycle, see [Robot Plugin Architecture](./explanation/robot-plugin-architecture.md).

## Install a Curated Plugin

Curated plugins are listed on Studio's **Plugins** page.

1. Open **Plugins** from the main navigation.
2. Select **Install** next to the plugin you want to use.
3. Wait for the installation to finish.
4. Restart Studio when prompted.

The restart is required because Python entry points and the robot catalog are
loaded when the backend starts.

[//]: # "Screenshot suggestion: Plugins page showing available and installed plugin rows with an Install button."
[//]: # "Screenshot suggestion: Plugin installation progress followed by the restart-required prompt."

## Use A Plugin Robot

After the plugin has been installed and Studio has restarted:

1. Open a project.
2. Open **Robots** and select **Add robot**.
3. Select the follower or leader robot type supplied by the plugin.
4. Complete the configuration form and save the robot.

The robot picker and robot form are automatically updated after installation
and restart. The form is generated from the plugin's catalog definition and
payload model, so different robot types can expose different configuration
fields and connection controls.

A **follower** executes actions during teleoperation or inference. A **leader**
provides input for teleoperation.

[//]: # "Screenshot suggestion: Robot type picker showing a newly available plugin robot after restart."
[//]: # "Screenshot suggestion: Plugin-provided robot configuration form with a connection selector and advanced options."

## Install An Unofficial Plugin

An unofficial plugin can be installed directly into the backend environment.
This is useful for private integrations, experiments, and plugins that have not
been added to Studio's curated list.

From the Studio repository:

```bash
cd application/backend
uv add physicalai-my-robot-plugin
uv sync
```

Install from Git:

```bash
cd application/backend
uv add "physicalai-my-robot-plugin @ git+https://github.com/example/physicalai-my-robot-plugin.git"
```

For a local editable plugin, add it directly from the backend directory:

```bash
cd application/backend
uv add --editable ../../physicalai-my-robot-plugin
uv sync
```

This adds the dependency and its path source to `pyproject.toml`. You can also
write the equivalent source override manually:

```toml
[tool.uv.sources]
physicalai-my-robot-plugin = { path = "../../physicalai-my-robot-plugin", editable = true }
```

Then run `uv sync` and restart Studio. Direct installation still requires the
plugin to declare the `physicalai.studio.catalog_plugins` entry point. It will
appear in the robot picker after restart, but it will not appear on the
**Plugins** page unless it is added to Studio's curated manifest.

For Docker deployments, installing a package into a running container is not a
persistent deployment method. Rebuilding or replacing the container can remove
the package. Use a custom image or a development setup when the plugin must
survive container replacement.

## Plugin Structure

A minimal plugin can have this structure:

```text
physicalai-my-robot-plugin/
├── pyproject.toml
├── README.md
├── src/
│   └── physicalai_my_robot_plugin/
│       ├── __init__.py
│       └── studio_catalog.py
├── urdf/                    # optional robot models and meshes
└── tests/
```

The [physicalai-plugins repository][physicalai-plugins] contains complete
examples. In particular, the ReBot package demonstrates multiple robot types,
serial discovery, payload validation, URDF assets, and driver builders. The
bimanual SO-101, LeKiwi, LeRobot, and MuJoCo packages demonstrate other common
patterns.

[physicalai-plugins]: https://github.com/MarkRedeman/physicalai-plugins

Install the SDK used by a plugin with:

```bash
uv add physicalai-studio-plugin
```

Declare the Studio entry point in `pyproject.toml`:

```toml
[project.entry-points."physicalai.studio.catalog_plugins"]
my-robot = "physicalai_my_robot_plugin.studio_catalog:register_physicalai_studio_plugin"
```

## The `studio_catalog.py` Interface

The entry-point callable receives Studio's catalog registry. Register one
`RobotCatalogDefinition` for every robot type:

```python
from typing import Any

from physicalai_studio_plugin import (
    CatalogRobot,
    CatalogRobotFactory,
    RobotCatalogDefinition,
)
from physicalai.robot.interface import Robot
from pydantic import BaseModel


class MyRobotPayload(BaseModel):
    connection_string: str = ""


async def build_my_robot(
    robot: CatalogRobot[MyRobotPayload],
    factory: CatalogRobotFactory,
) -> Robot:
    # Resolve the configured connection and return a Physical AI Robot.
    raise NotImplementedError


def register_physicalai_studio_plugin(registry: Any) -> None:
    registry.register_robot(
        RobotCatalogDefinition(
            type="MyRobot_Follower",
            display_name="My Robot Follower",
            role="follower",
            robot_payload=MyRobotPayload,
            robot_builder=build_my_robot,
        )
    )
```

The example shows the important contract; a real plugin should implement the
builder rather than raise `NotImplementedError`. Add a `RobotProbe` when the
robot supports discovery, identification, or online-status checks.

`RobotCatalogDefinition` fields have these responsibilities:

- `type`: stable, globally unique identifier persisted in Studio projects.
- `display_name`: human-readable name shown in the UI.
- `role`: `follower` or `leader`.
- `robot_payload`: Pydantic model defining configuration data.
- `robot_builder`: async callable that returns a Physical AI robot driver.
- `probe`: optional discovery, identification, and online-status implementation.
- `asset`: optional URDF, mesh, and joint-map information.
- `adapter_options`: optional control and effort-forwarding behavior.

The `type` value must not be casually renamed. It is stored in project data and
must remain unique across all installed plugins.

The full SDK reference is in [`application/plugin/README.md`](../plugin/README.md).

## Render The Robot Form

Studio renders a plugin's robot form from the Pydantic payload model's JSON
Schema. Standard Pydantic metadata provides most of the UI:

- `title` becomes a field label.
- `description` becomes help text.
- `default` pre-fills a value.
- `enum` renders a selection control.
- Required fields and validation come from the model.
- Nested Pydantic models render recursively.

Use `robot_field_ui(...)` for the Studio-specific required-field override:

```python
from physicalai_studio_plugin import robot_field_ui
from pydantic import Field

timeout: float = Field(
    default=10.0,
    json_schema_extra=robot_field_ui({"required": True}),
)
```

Use `robot_payload_ui(...)` when fields need ordering, sections, guidance, or a
first-party connection control. The supported item kinds are:

- `section`: groups items under an optional heading.
- `field`: places a normal payload field.
- `connection`: renders Studio's serial-device selector and owns its bindings.
- `info`: renders read-only guidance or warnings.

```python
from physicalai_studio_plugin import robot_payload_ui
from pydantic import BaseModel, ConfigDict


class MyRobotPayload(BaseModel):
    connection_string: str = ""
    serial_number: str = ""

    model_config = ConfigDict(
        json_schema_extra=robot_payload_ui(
            [
                {
                    "kind": "connection",
                    "label": "Select robot",
                    "device_discovery": True,
                    "bind": {
                        "connection": "connection_string",
                        "serial_number": "serial_number",
                    },
                },
            ]
        )
    )
```

Connection bindings are relative to the payload model that declares them. A
nested payload should define its own connection metadata instead of using a
dotted path such as `left_arm.connection_string`.

Explicit UI items customize only the fields they place or own. Fields that are
not mentioned continue to render automatically from JSON Schema.

[//]: # "Screenshot suggestion: Plugin form showing a connection item, a section, an advanced-options switch, and inline information text."

## Add A Plugin To The Plugins Page

The entry point controls runtime discovery. The curated manifest controls what
appears on Studio's **Plugins** page and what the UI is allowed to install.

Add a reviewed entry to
`application/backend/src/plugins/manifest.json`:

```json
{
  "id": "physicalai-my-robot-plugin",
  "name": "My Robot Plugin",
  "description": "Integration for My Robot.",
  "repo_url": "https://github.com/example/physicalai-my-robot-plugin",
  "install_source": "physicalai-my-robot-plugin>=0.1.0",
  "robots": [
    {
      "type": "MyRobot_Follower",
      "display_name": "My Robot Follower",
      "role": "follower"
    }
  ]
}
```

The `id` must match the Python distribution name. `install_source` is the
requirement passed to `uv`; it must be a valid and reviewed package, Git, or
path source. The manifest's robot list supplies user-facing information before
installation. A plugin such as LeRobot can discover additional definitions at
runtime, so its manifest list may be empty.

After changing the manifest, restart the backend so the updated curated list is
loaded.

## Troubleshooting

### The plugin is installed but its robots are missing

Restart Studio. Entry points and the catalog are loaded at backend startup.
If the robots are still missing, verify the entry-point group, import path, and
that `register_physicalai_studio_plugin` calls `registry.register_robot(...)`.

### The plugin is missing from the Plugins page

Manual `uv add` installation does not add a plugin to the curated UI. Add a
reviewed entry to `application/backend/src/plugins/manifest.json`.

### The form has missing or duplicated fields

Check the Pydantic model's JSON Schema metadata and the names used by
`robot_payload_ui(...)`. Connection items own their bound fields, and nested
models must define bindings relative to themselves.

### Registration fails with a duplicate robot type

Choose a globally unique and stable `RobotCatalogDefinition.type`. Do not reuse
an identifier from another plugin or rename an identifier already persisted in
project data.

### Local plugin changes do not appear

Run `uv sync` if the dependency source changed, then restart Studio. Changes to
`studio_catalog.py`, payload models, and entry points are not reliably picked up
by a running backend.

### A robot cannot connect

Check the payload values, driver dependencies, serial permissions, probe
implementation, and `CatalogRobotFactory` connection resolution. A successful
catalog registration does not guarantee that the physical device is reachable.

### The URDF preview fails

Check the `RobotAsset` URDF path, package map, joint map, and root resolver. An
asset is optional: a plugin without one can still be configured and used, but
Studio cannot show a 3D preview.

### The plugin disappears after a Docker rebuild

Packages installed into a running container are not persistent. Build them into
a custom image or use a development setup with the plugin source available.
