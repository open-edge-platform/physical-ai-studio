# Studio using-skill scenarios

Run each prompt from an unrelated customer project with the packages installed, **not** from a Studio source checkout. Reset the agent between prompts. Score the selected skill, the actual API/CLI invocation, a checkable result, and whether the agent avoids editing Studio internals. Mark downloads and hardware interactions before running them.

## `pai-train-policy`

1. "Write a one-batch ACT training smoke test using `Trainer`, `LeRobotDataModule(repo_id='lerobot/pusht')`, and `ACT` from Python, not the CLI." Expect `trainer.fit(model=policy, datamodule=datamodule)` and a shape check.
2. "Resume a `physicalai fit` ACT experiment from `last.ckpt`. Show the equivalent Python API and CLI commands." Expect `ckpt_path` and the same config/experiment name.
3. "Try a new learning rate without losing reproducibility." Expect a project-local YAML or CLI override, `--print_config`, and `fast_dev_run` before a full run.

## `pai-work-with-datasets`

1. "Use `lerobot/pusht` as data for ACT. Inspect one batch from Python, then wire it into my own YAML." Expect the datamodule API, matched feature names, and a config smoke test.
2. "My dataset uses `observation.state` but the policy expects another feature name." Expect inspection and a project-level data/config mapping, not an edit to `library/src`.
3. "Convert a flattened LeRobot sample into an Observation and back." Expect `FormatConverter.to_observation(...)` and `FormatConverter.to_lerobot_dict(...)`; retain required metadata.

## `pai-export-policy`

1. "Export an ACT checkpoint to ONNX for Runtime." Expect `physicalai export` or `policy.export(...)`, manifest inspection, and numeric parity against the source policy.
2. "Export Pi0 to ExecuTorch for a Runtime that has no matching adapter." Expect a precise warning about loading support, not a promise of deployment.
3. "I have an OpenVINO export; verify that Runtime can load the model and that input/output names match." Expect the canonical manifest schema and a fake-observation smoke inference.

## `pai-benchmark-policy`

1. "Benchmark an ACT checkpoint on PushT for ten episodes using both Python and CLI." Expect success metrics in `results.json` and `results.csv` with consistent settings.
2. "Compare a checkpoint with its ONNX export on the same environment." Expect two runs, a common baseline, and metric parity rather than only latency.
3. "Record only failed episodes while benchmarking on Libero." Expect a customer-owned benchmark config and the appropriate optional gym dependency, not a new library Benchmark class.

## `pai-create-robot-plugin`

1. "Create a serial follower robot plugin in my own package." Expect a Runtime protocol driver, SDK payload/entry point, fake-hardware tests, and backend catalog discovery.
2. "Create a two-arm TCP plugin with follower and leader modes." Expect stable distinct catalog types, ordered joints, and no serial picker in the payload.
3. "Package URDF meshes for a Studio robot plugin and verify visualization." Expect packaged assets, a `RobotAsset`, tests, and the live URDF endpoint. Do **not** edit Studio's curated plugin manifest.
