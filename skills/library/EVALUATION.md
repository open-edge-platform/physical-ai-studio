# Skill Evaluation Scenarios

Use these prompts to test whether an agent correctly invokes and follows each library skill. Each scenario checks one realistic failure mode. Run the agent from the repo root with no extra hints beyond the prompt.

Expected rubric per scenario:

- **Activates the right skill** — loaded `SKILL.md` matches the topic.
- **Uses real paths and commands** — references `library/src/physicalai/...`, `physicalai ...`, `uv run pytest ...` as documented.
- **Follows the workflow checklist** — does not skip Required checks / Verify steps.
- **Produces a checkable artifact** — a command run, a file written, or a test result.

## `library-adding-a-policy`

### Scenario 1: Add a new native policy family

> "Add a new policy family called `mynet` under `physicalai.policies`. It should be constructible from Python, available through `get_policy`, trainable with `physicalai fit`, and follow the same config/model/policy split as ACT. Keep it minimal."

Expected behavior:

- Creates `library/src/physicalai/policies/mynet/{config.py,model.py,policy.py}` mirroring `policies/act/`.
- Registers `Mynet`, `MynetConfig`, `MynetModel` in `policies/__init__.py` and the `get_policy(...)` dispatch.
- Verifies `from physicalai.policies import Mynet, get_policy` and `get_policy("mynet")` work without CLI involvement.
- Adds `library/configs/physicalai/mynet.yaml` wiring `model`, `data`, and `trainer`.
- Adds at least one test under `library/tests/unit/policies/`.
- Runs `uv run pytest tests/unit/policies -k mynet` and `physicalai fit --config configs/physicalai/mynet.yaml --trainer.fast_dev_run=true`.

### Scenario 2: Extend an existing policy for export

> "ACT currently trains but is not exportable. Add ONNX export support to ACT and validate the artifact."

Expected behavior:

- Loads `library-exporting-and-validating` in addition to `library-adding-a-policy`.
- Adds `ExportablePolicyMixin` to `ACT` only after providing a valid sample input.
- Does not claim the export works until a parity check is described or run.
- Updates docs and tests; does not change action/feature contracts silently.

### Scenario 3: Refactor a LeRobot wrapper into a native policy

> "We currently call into LeRobot for Diffusion. Port it to a native Studio policy with the config/model/policy split."

Expected behavior:

- Creates a first-party package under `policies/` rather than editing `policies/lerobot/`.
- Keeps LeRobot adapter code behind a factory/helper; avoids mixing LeRobot internals with Lightning `Policy` semantics.
- Validates direct Python construction and `physicalai fit --config configs/physicalai/<name>.yaml --trainer.fast_dev_run=true` both pass.

## `library-training-a-policy`

### Scenario 4: Run an API smoke test

> "Write a short Python smoke test that trains ACT on `lerobot/pusht` for one fast-dev batch using `Trainer`, `LeRobotDataModule`, and `ACT`. Do not invoke the CLI."

Expected behavior:

- Constructs `LeRobotDataModule(repo_id="lerobot/pusht", train_batch_size=2)`, `ACT()`, and `Trainer(fast_dev_run=True)` from Python.
- Calls `trainer.fit(model=policy, datamodule=datamodule)`.
- If it fails, diagnoses whether the error is in policy construction, dataloading, or trainer setup.

### Scenario 5: Resume training from a checkpoint

> "My ACT training was interrupted at epoch 5. Show both the Python API and CLI ways to resume it from the last checkpoint while keeping the same experiment name."

Expected behavior:

- Finds the latest `.ckpt` under `experiments/` matching the experiment name.
- Shows `Trainer(...).fit(model=policy, datamodule=datamodule, ckpt_path=<path>)` and `physicalai fit --config configs/physicalai/act.yaml --ckpt_path <path>`.
- Preserves logger version / experiment name so metrics continue in the same run.

### Scenario 6: Tune a hyperparameter safely

> "I want to try a larger learning rate for ACT. Show me the Python API change and the equivalent CLI/config override without breaking reproducibility."

Expected behavior:

- Shows direct API construction with the changed policy config/constructor argument and the equivalent CLI override.
- Uses `--print_config` to confirm CLI serialization matches the API intent.
- Runs `fast_dev_run` before any full run.

## `library-benchmarking-a-policy`

### Scenario 7: Benchmark a trained checkpoint

> "I have an ACT checkpoint at `experiments/act/version_0/checkpoints/last.ckpt`. Benchmark it on PushT with 10 episodes from a Python script, and show the equivalent CLI."

Expected behavior:

- Loads the policy and evaluates with `PushTBenchmark(num_episodes=10).evaluate(policy)` from Python.
- Shows the equivalent `physicalai benchmark --config configs/benchmark/pusht.yaml --policy physicalai.policies.ACT --ckpt_path <path> --benchmark.num_episodes 10`.
- Confirms `results.json` and `results.csv` are written to `--output_dir`.
- Interprets the success metric and compares it to a baseline if one exists.

### Scenario 8: Add a new benchmark suite

> "Add a Libero benchmark config and make sure it can run one episode from an ACT checkpoint."

Expected behavior:

- Reuses `LiberoBenchmark` in `physicalai.benchmark.gyms` rather than inventing a new class.
- Creates `configs/benchmark/libero.yaml` if it does not already exist.
- Runs one episode through `LiberoBenchmark(...).evaluate(policy)` and the equivalent CLI smoke test.
- Marks heavy gym deps as optional/lazy.

### Scenario 9: Benchmark an exported artifact

> "Compare the ONNX export of my ACT checkpoint against the original `.ckpt` on the same benchmark."

Expected behavior:

- Loads `library-exporting-and-validating` to produce the ONNX artifact.
- Runs benchmark twice through the Python API and/or CLI with paths pointing to `.ckpt` and export dir.
- Reports metric parity, not just runtime.

## `library-working-with-datasets`

### Scenario 10: Wire a new dataset into a config

> "I want to train ACT on `lerobot/pusht_reduced`. Verify the datamodule loads from Python, then update the config."

Expected behavior:

- Constructs `LeRobotDataModule(repo_id="lerobot/pusht_reduced")`, runs `prepare_data()`, `setup("fit")`, and inspects one train batch.
- Edits `data.init_args.repo_id` in the config and runs `physicalai fit --config <config> --trainer.fast_dev_run=true` to confirm CLI parity.
- Checks feature names/action dim alignment with the policy `Config`.

### Scenario 11: Debug a feature-name mismatch

> "Training fails with a missing feature error in the datamodule. The dataset has `observation.state` but the policy expects `observation.robot_state`. Fix it."

Expected behavior:

- Inspects `data/observation.py` and the policy's preprocessor.
- Changes the policy's expected feature names or maps them in the datamodule, with justification.
- Does not silently rename fields in a way that breaks export/Runtime.

### Scenario 12: Add a data converter test

> "We added a new field to the physicalai observation layout. Make sure the LeRobot converter round-trips it."

Expected behavior:

- Edits/tests `data/lerobot/converters.py` (`DataFormat.physicalai` ↔ `DataFormat.lerobot`).
- Adds a unit test under `tests/unit/data/`.
- Runs `uv run pytest tests/unit/data -k <field>`.

## `library-exporting-and-validating`

### Scenario 13: Export a checkpoint to ONNX

> "Export my ACT checkpoint to ONNX so Runtime can load it."

Expected behavior:

- Exports through `policy.export("./export", backend=ExportBackend.ONNX)` after loading the checkpoint from Python.
- Shows the equivalent `physicalai export --policy physicalai.policies.ACT --ckpt_path <path> --backend onnx --output_dir ./export`.
- Verifies the export directory contains a model file and metadata.
- Runs a numerical parity check against the Torch policy path.

### Scenario 14: Fix an ONNX tracing failure

> "ACT export to ONNX fails with a dynamic control-flow error. Fix it without changing training behavior."

Expected behavior:

- Identifies the non-traceable code path in the model.
- Refactors to a traceable equivalent or uses a scripted helper; validates training still passes `fast_dev_run`.
- Does not change action semantics or output shapes.

### Scenario 15: Document an unsupported backend

> "A user asks for ExecuTorch export of Pi0. We don't support it yet. Update the code/docs so this is clear."

Expected behavior:

- Does not promise ExecuTorch in user-facing docs.
- Adds a clear error or docstring noting the missing support.
- Cross-checks `references/executorch.md` and `references/export-contract.md`.

## Running the evaluations

1. Reset the agent context between scenarios.
2. Give only the prompt; do not hint at which skill to use.
3. Score against the expected behavior rubric above.
4. If an agent fails a scenario, update the corresponding `SKILL.md` or `references/*.md` and rerun.
