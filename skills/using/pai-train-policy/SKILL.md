---
name: pai-train-policy
description: Trains, validates, tests, and predicts with Physical AI Studio policies. Use when running physicalai fit/validate/test/predict, wiring a Policy, LeRobotDataModule and Trainer in a project config or Python script, resuming a checkpoint, or debugging an existing run. DO NOT USE FOR implementing a new policy family inside the Studio library.
license: Apache-2.0
---

# Training a policy (library)

Training uses `physicalai.train.Trainer` with a `Policy` and a `DataModule`. Both entry points work from an installed `physicalai-train` package; the Studio repository contains example configs and implementation code for reference:

- **CLI** — `physicalai fit` (and `validate`, `test`, `predict`): jsonargparse YAML in your project (examples live under Studio's `library/configs/`), overrides on the command line; checkpoints under `experiments/{name}/version_N/` by default. See the [CLI guide](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/how-to/training/cli.md).
- **Python API** — construct `Policy`, `LeRobotDataModule` (or another datamodule), and `Trainer`, then `trainer.fit(model=policy, datamodule=datamodule)` (and `validate` / `test` / `predict` with a checkpoint as needed). See the [quickstart](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/getting-started/quickstart.md).

The CLI subcommands and the Python API share the same objects; YAML `class_path` / `init_args` should match what you would wire in code.

The four CLI subcommands share the same `--model` / `--data` / `--trainer.*` shape; `validate`/`test`/`predict` additionally take `--ckpt_path`. Prefer the Python API when the user needs a script, and the CLI for a repeatable YAML-driven run.

## Anatomy of a config

A config wires three pieces via `class_path` / `init_args`:

- `model` — a `Policy` subclass (e.g. `physicalai.policies.ACT`).
- `data` — a `DataModule`, usually `physicalai.data.lerobot.LeRobotDataModule` with a `repo_id` (e.g. `lerobot/pusht`).
- `trainer` — Lightning args (`max_epochs`, `accelerator`, `devices`, callbacks…).

Start with a [first-party example config](https://github.com/open-edge-platform/physical-ai-studio/tree/main/library/configs/physicalai) or a [LeRobot example](https://github.com/open-edge-platform/physical-ai-studio/tree/main/library/configs/lerobot), copy it into your project, and override values on the CLI (`--trainer.max_epochs 200 --data.train_batch_size 64`).

## Python API workflow

Use this path for scripts, notebooks, tests in the user's project, or direct library integration.

```python
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT
from physicalai.train import Trainer

datamodule = LeRobotDataModule(repo_id="lerobot/pusht", train_batch_size=2)
policy = ACT()
trainer = Trainer(fast_dev_run=True)
trainer.fit(model=policy, datamodule=datamodule)
```

1. **Construct the same objects the CLI would instantiate**: a `Policy`, a `DataModule`, and `Trainer`.
   - Done when: construction works without relying on jsonargparse YAML.
2. **Smoke-test the API wiring** with `Trainer(fast_dev_run=True)`.
   - Done when: one train + one val batch complete without shape or feature errors.
3. **Validate / test / predict from Python** with the corresponding `Trainer` method and `ckpt_path` when needed.
   - Done when: the API call and the equivalent CLI command agree on checkpoint/config behavior.

## CLI workflow

Use this path for terminal commands, YAML configs, and reproducible experiments.

1. **Start from an example config** matching your policy family; copy it into your project rather than editing Studio's source.
   - Done when: `physicalai fit --config <your.yaml> --print_config` renders the fully-resolved config with no errors.
2. **Smoke-test the wiring** before a real run:
   ```bash
   physicalai fit --config <your-config.yaml> --trainer.fast_dev_run=true
   ```
   - Done when: one train + one val batch complete without shape or config errors.
3. **Run training**, overriding on the CLI as needed:
   ```bash
   physicalai fit --config <your-config.yaml> --trainer.max_epochs 200
   ```
   - Done when: checkpoints appear under `experiments/{name}/version_N/`.
4. **Validate / test / predict** from a checkpoint:
   ```bash
   physicalai validate --config <your-config.yaml> --ckpt_path experiments/<name>/version_0/checkpoints/last.ckpt
   ```
5. **Iterate on metrics**, not just loss — confirm the val metric relevant to the task moves, and record the config + checkpoint that produced it.

## Debugging a run

- API: construct `Policy`, `DataModule`, and `Trainer` directly in a short script or test to isolate whether failure is in object construction, dataloading, or CLI parsing.
- `--trainer.fast_dev_run=true` — one batch each stage; the first thing to try on any failure.
- `--print_config` — see the exact resolved config jsonargparse built.
- Shape/feature mismatches usually mean the datamodule's `Feature` names or action dim disagree with the selected policy; inspect a batch with `pai-work-with-datasets`.
- Dataset download stalls: the run is pulling a LeRobot `repo_id`; see `pai-work-with-datasets`.

## Required checks

- Config resolves (`--print_config`) and `fast_dev_run` passes before any long run.
- The equivalent Python API construction path passes if the project uses the API.
- `accelerator`/`devices` match the installed backend extra (`xpu`/`cuda`/`cpu`).
- Config field names stay consistent with the installed policy's `Config` class.

## Verify

```bash
# from your project directory
physicalai fit --config <your-config.yaml> --trainer.fast_dev_run=true
```

For a Python API integration, run an equivalent smoke test that constructs `Policy`, `DataModule`, and `Trainer` directly and calls `trainer.fit(...)`.

## Related skills

- `pai-work-with-datasets` — for the `data` half of the config.
- `pai-benchmark-policy` — to evaluate a trained checkpoint in a gym.
