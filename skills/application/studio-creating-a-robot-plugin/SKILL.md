---
name: studio-creating-a-robot-plugin
description: Creates or modifies an external Physical AI robot plugin for Studio. Use when implementing a Runtime Robot driver outside Studio, adding studio_catalog.py, RobotCatalogDefinition, RobotProbe, RobotAsset, the physicalai.studio.catalog_plugins entry point, or a curated entry in application/backend/src/plugins/manifest.json.
license: Apache-2.0
---

# Creating a Studio Robot Plugin

Build the Physical AI robot driver independently before adding Studio support. A plugin owns its driver, device integration, Pydantic payloads, catalog definitions, and optional assets. Studio owns project persistence, generated configuration forms, runtime orchestration, and the curated Plugins page.

Read `application/docs/robot-plugins.md` and `application/plugin/README.md` before editing. For complete plugin examples, use the [Physical AI plugins repository](https://github.com/MarkRedeman/physicalai-plugins), including its ReBot package. Use Studio's built-in network robot implementation at `application/backend/src/robots/catalog/widowxai.py` as a schema example.

## Workflow

1. **Choose the plugin boundary and stable identifiers.** Keep the driver and its vendor dependencies in the plugin repository; do not add the robot to Studio's built-in catalog. Choose a globally unique, stable `RobotCatalogDefinition.type` for every configuration, such as `AcmeArm_Follower` and `AcmeArm_Leader`.

   - `type` is persisted in Studio projects and must never be casually renamed or duplicated by another installed plugin.
   - Decide which configurations are `follower` robots that execute actions and which are `leader` robots that supply teleoperation input.
   - Done when: every supported driver mode has a stable type, display name, role, connection method, and expected joint order.

2. **Implement the Runtime robot without Studio imports.** Follow the Runtime skill `physicalai-runtime-adding-a-robot-integration` in the Physical AI repository. The driver structurally implements `physicalai.robot.interface.Robot`; it does not need a Studio base class.

   - Implement idempotent `connect()`, safe `disconnect()`, and `is_connected()`.
   - Expose `joint_names` in exactly the order used by observations and actions. `get_observation()` returns joint positions in that order and a `time.monotonic()` timestamp. `send_action(action, *, goal_time=...)` accepts an action with shape `(len(joint_names),)`.
   - Expose `device_ids` from constructor arguments without hardware I/O, including every exclusive device for composite robots.
   - Put vendor SDK imports behind the driver connection path when they are optional or heavy. Validate user-configured ports and addresses; do not invoke a shell with them.
   - Decorate every driver class returned to Studio with `@physicalai.config.export_config`. Studio serializes the disconnected driver to start its hardware-owner process; an undecorated driver fails with `ConfigError`.
   - Done when: mocked-hardware tests prove `isinstance(driver, Robot)`, lifecycle behavior, safe disconnected behavior, joint/action order, and configuration round-tripping, with no import of `physicalai_studio_plugin`.

3. **Package the standalone driver.** Create a normal Python distribution with the driver package, tests, and optional URDF/mesh files. Add `physicalai`, `physicalai-studio-plugin`, vendor dependencies, and test extras as appropriate.

   - Declare the Studio entry point in the plugin package's `pyproject.toml`:

   ```toml
   [project.entry-points."physicalai.studio.catalog_plugins"]
   acme-arm = "physicalai_acme_arm_plugin.studio_catalog:register_physicalai_studio_plugin"
   ```

   - Include URDF resources in both wheel and source distributions if the catalog exposes a `RobotAsset`.
   - Build or install the package in the same Python environment used by `application/backend/`; entry points from another virtual environment are invisible to Studio.
   - Done when: the package imports, its entry point resolves, and its standalone test suite passes.

4. **Define one typed payload per Studio robot configuration.** Add `studio_catalog.py` in the plugin package. Define Pydantic `BaseModel` payloads for connection values and driver options; the payload is the persisted data and generated form contract.

   - Use Pydantic `Field` titles, descriptions, defaults, literals/enums, and validators for ordinary configuration behavior.
   - Use `robot_field_ui({"advanced_configuration": True})` only for fields that belong behind Studio's advanced configuration control.
   - Use `robot_payload_ui(...)` to order fields, add sections or guidance, and render a serial connection selector. A `connection` item owns the fields named by `bind.connection` and optional `bind.serial_number`, so do not also render those fields as `field` items.
   - Connection bindings are relative to the Pydantic model declaring the UI metadata. Nested arm models define their own bindings; never use dotted paths such as `left.connection_string`.
   - Done when: `validate_robot_payload_ui(Payload)` and `Payload.model_rebuild(raise_errors=True)` pass for every payload model.

5. **Build and register catalog definitions.** Implement an async builder for each driver shape and register a `RobotCatalogDefinition` for every stable type.

   - The builder receives `CatalogRobot[Payload]` and `CatalogRobotFactory`. Read `robot.payload`, validate or normalize it if necessary, resolve serial or network connections with `factory.find_port(SerialPortInfo(...))` when appropriate, and return the plain `@export_config` Runtime driver. Do not return a `SharedRobot`; Studio wraps the driver.
   - Raise a clear error if a configured device cannot be resolved.
   - Add a structural `RobotProbe` only when discovery, visual identification, or online checks have meaningful implementations. Keep probe behavior separate from driver construction.
   - Add `RobotAsset` only when the plugin ships a valid URDF, mesh package map, and observation-key-to-joint mapping. An asset-less robot is still supported but has no 3D preview.
   - Register each definition from the entry-point function:

   ```python
   def register_physicalai_studio_plugin(registry) -> None:
       for definition in _definitions():
           registry.register_robot(definition)
   ```

   - Done when: direct registration into a fake registry produces exactly the expected types, payload models, builders, roles, and optional assets/probes.

6. **Test the catalog contract without Studio.** Add focused tests alongside the plugin package.

   - Test the entry-point registration and exact set of catalog types.
   - Test payload defaults, required-field and cross-field validation, `model_rebuild`, and `validate_robot_payload_ui(...)`.
   - Test each builder with fake payload containers and a fake `CatalogRobotFactory`, including both a resolved connection and a missing-device error.
   - Test `RobotAsset.root_resolver()` and the URDF path when assets are supplied.
   - Run the package's focused test command from its repository, for example:

   ```bash
   uv run pytest packages/physicalai-acme-arm-plugin/tests/
   ```

   - Done when: driver and catalog tests pass without physical hardware.

7. **Install locally and restart Studio for catalog discovery.** From `application/backend/`, install the plugin into the backend environment, then restart the backend. For a local plugin, use an editable dependency relative to the backend directory:

   ```bash
   uv add --editable ../../physicalai-acme-arm-plugin
   uv sync
   ```

   - Entry points and catalog schemas load only at backend startup. Re-run `uv sync` after dependency-source changes and restart the backend after changes to entry points, payload models, or `studio_catalog.py`.
   - With the UI started by `npm run start` from `application/ui/`, its development server proxies `/api` to the backend and is available on port 3000.
   - Done when: the backend starts without catalog registration errors and the plugin type is available from the live catalog.

8. **Debug through the live catalog APIs before debugging the UI.** Use the browser or `curl` against the UI proxy:

   ```bash
   curl --fail http://localhost:3000/api/robots/catalog
   curl --fail http://localhost:3000/api/robots/catalog/Trossen_Bimanual_WidowXAI_Follower/schema
   curl --fail http://localhost:3000/api/robots/catalog/AcmeArm_Follower/schema
   ```

   - Treat `Trossen_Bimanual_WidowXAI_Follower/schema` as the known-good reference for a network bimanual payload. Compare your schema's `properties`, required fields, defaults, descriptions, nested `$defs`, and `x-physicalai-ui` metadata against the desired form.
   - If the plugin type is absent from `/catalog`, inspect backend startup logs for entry-point import, registration, duplicate-type, or UI-schema validation failures. Confirm the distribution is installed in the backend environment and restart it.
   - If the type appears but its form is wrong, inspect `/{robot_type}/schema`. Correct Pydantic field metadata or `robot_payload_ui` ownership/bindings rather than adding plugin-specific React code.
   - For a probe, test `/{robot_type}/discover`, `/{robot_type}/identify`, and `/{robot_type}/is-online` with a JSON body that conforms to the schema. For a build failure, inspect the saved payload, `find_port` result, device permissions, vendor dependencies, and driver logs.
   - If visualization fails, request `/{robot_type}/urdf` and verify the asset root resolver, relative URDF path, package map, and joint map.
   - Done when: catalog and schema endpoints return the expected definition and JSON Schema, and the generated Studio form reflects it without duplicate or missing fields.

9. **Add curated UI installation only after the package is installable.** Add a reviewed entry to `application/backend/src/plugins/manifest.json` when Studio should expose the plugin on its Plugins page.
   - `id` must equal the Python distribution name. `install_source` must be a reviewed package, Git, or path requirement accepted by `uv pip install`. Add user-facing metadata and known robot types under `robots`.
   - The manifest controls what the UI may install; it does not replace the package entry point that registers the actual robot definitions.
   - Restart the backend after a manifest change.
   - Done when: the Plugins page lists the plugin before installation, installs the reviewed source, prompts for restart, and the restarted backend exposes the registered types through `/api/robots/catalog`.

## Verify

Run the plugin's driver and catalog tests first. Then validate the Studio skill changes from the Studio repository root:

```bash
python3 .github/scripts/skills/agent_skills.py sync
python3 .github/scripts/skills/agent_skills.py validate
prek run --all-files
```

## References

- `application/docs/robot-plugins.md` - installation, catalog, form, manifest, and troubleshooting contract.
- `application/plugin/README.md` - Studio plugin SDK types and examples.
- `application/backend/src/api/robot_catalog.py` - live catalog, schema, probe, and asset endpoints.
- `application/backend/src/robots/catalog/widowxai.py` - built-in single-arm and bimanual network payload example.
- [Physical AI Runtime robot-integration skill](https://github.com/open-edge-platform/physicalai/blob/main/skills/runtime/physicalai-runtime-adding-a-robot-integration/SKILL.md) - Runtime-only driver workflow.
