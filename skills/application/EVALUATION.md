# Studio Skills Evaluation

## `studio-creating-a-robot-plugin`

1. **Serial follower package**: Ask the agent to create a serial follower plugin with discovery and a connection selector. It should implement and test the Runtime driver first, add an exportable driver, use a payload-level `connection` UI item, test catalog registration and builders with fakes, and use the catalog and schema endpoints after local installation.
2. **Bimanual network robot**: Ask the agent to create a two-arm TCP follower and leader package. It should use distinct stable types, a typed payload with left and right addresses, no serial connection picker, a plain Runtime composite driver, and compare its schema with `Trossen_Bimanual_WidowXAI_Follower/schema` at port 3000.
3. **Curated URDF plugin**: Ask the agent to prepare a published robot plugin with an included URDF and meshes for UI installation. It should include resources in distribution artifacts, define `RobotAsset` paths and joint mapping, test asset resolution, add a reviewed manifest entry only after the package entry point works, and verify `/catalog`, `/{type}/schema`, and `/{type}/urdf` after restart.

## `studio-adding-robot-form-ui-fields`

1. **Add a new upload-based field kind**: Ask the agent to add a `robot_payload_ui` item kind that uploads and parses structured JSON, with SDK validation and a kind-based `SchemaFormItem` renderer path (no heuristic schema detection).
2. **Preserve advanced/required semantics**: Ask the agent to wire the new field through shared visibility rules so `advanced_configuration` hides it by default while required/optional labeling remains accurate.
3. **Document plugin adoption and migration**: Ask the agent to update plugin docs and architecture docs with the new supported kind, include a plugin snippet, and create a handoff note for plugin maintainers covering rollout guidance and tests.

## `studio-commanding-so101-servos-safely`

1. **Hold a pose**: Ask the agent to hold the SO-101 follower at frame 0 of a dataset episode while a camera is adjusted. It should clamp the target off the end stops, reach it with a bounded guarded ramp, send the final target once, and write no loop that re-sends a constant target; any position reads it adds must not call `connect()` on a live arm.
2. **Pose-capture mirror**: Ask the agent to add a Qt dialog that mirrors the leader arm onto the follower so users can capture poses. It should add a deadband, a timeout, the arm-joint clamp, and a tracking-error stall check (gripper excluded), save the follower's measured position rather than the leader's, and keep motion off the GUI thread with a stop control.
3. **Policy on hardware**: Ask the agent to run an exported ACT package on an SO-101 with `RobotRuntime`. It should add the clamp and stall-abort callback through `callbacks=`, run inside `with runtime:` so an abort ends with the follower holding its measured pose, and name the residual risk (pushes below the stall threshold) with the passive at-rest check.
4. **Hot servo**: Report that a servo got hot after a demo ran all afternoon. The agent should stop whatever holds the serial port, read registers passively (goal vs present, load, temperature, status, position limits), check stored poses and dataset actions for values at +/-100, and check whether torque was left on at an unsupported pose — before proposing any code change.
