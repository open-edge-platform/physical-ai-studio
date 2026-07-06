---
name: library-training-a-policy
description: Trains, validates, tests, and runs prediction for Physical AI Studio policies via the library Lightning stack. Use when running physicalai fit/validate/test/predict, calling physicalai.train.Trainer and Policy APIs from Python, writing or editing YAML configs under library/configs, wiring a model + datamodule + trainer, resuming from a checkpoint, or debugging a training run. Covers ACT, Pi0, Pi0.5, GR00T, and SmolVLA.
license: Apache-2.0
---

# Training a policy (library)

Training uses `physicalai.train.Trainer` (`library/src/physicalai/train/trainer.py`, a `lightning.Trainer` subclass) with a `Policy` and a `DataModule`. The library deliberately supports two equal entry points:

- **CLI** — `physicalai fit` (and `validate`, `test`, `predict`): jsonargparse YAML under `library/configs/`, overrides on the command line; checkpoints under `experiments/{name}/version_N/` by default. See `library/docs/how-to/training/cli.md`.
- **Python API** — construct `Policy`, `LeRobotDataModule` (or another datamodule), and `Trainer`, then `trainer.fit(model=policy, datamodule=datamodule)` (and `validate` / `test` / `predict` with a checkpoint as needed). See `library/docs/getting-started/quickstart.md` and `library/docs/explanation/trainer/README.md`.

The CLI subcommands and the Python API share the same objects; YAML `class_path` / `init_args` should match what you would wire in code.

The four CLI subcommands share the same `--model` / `--data` / `--trainer.*` shape (see `cli/_dispatch.py`); `validate`/`test`/`predict` additionally take `--ckpt_path`. When a task is about library behavior rather than shell usage, prefer the Python API path first and then verify CLI parity if the change is user-facing.

## Anatomy of a config

A config wires three pieces via `class_path` / `init_args`:

- `model` — a `Policy` subclass (e.g. `physicalai.policies.ACT`).
- `data` — a `DataModule`, usually `physicalai.data.lerobot.LeRobotDataModule` with a `repo_id` (e.g. `lerobot/pusht`).
- `trainer` — Lightning args (`max_epochs`, `accelerator`, `devices`, callbacks…).

Configs live in `library/configs/physicalai/` (first-party: `act.yaml`, `pi0.yaml`, `pi05.yaml`, `groot.yaml`, `smolvla.yaml`) and `library/configs/lerobot/` (LeRobot-wrapped). Compose with `__base__` and override any field on the CLI (`--trainer.max_epochs 200 --data.train_batch_size 64`).

## Python API workflow

Use this path when the user asks for code, notebooks, tests, direct library integration, or changes to `Trainer`, `Policy`, or datamodules.

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

Use this path when the user asks for terminal commands, docs under `library/docs/how-to/`, YAML configs, reproducible experiments, or entry-point behavior.

1. **Start from an existing config** matching your policy family; copy it rather than writing from scratch.
   - Done when: `physicalai fit --config <your.yaml> --print_config` renders the fully-resolved config with no errors.
2. **Smoke-test the wiring** before a real run:
   ```bash
   physicalai fit --config configs/physicalai/<name>.yaml --trainer.fast_dev_run=true
   ```
   - Done when: one train + one val batch complete without shape or config errors.
3. **Run training**, overriding on the CLI as needed:
   ```bash
   physicalai fit --config configs/physicalai/<name>.yaml --trainer.max_epochs 200
   ```
   - Done when: checkpoints appear under `experiments/{name}/version_N/`.
4. **Validate / test / predict** from a checkpoint:
   ```bash
   physicalai validate --config configs/physicalai/<name>.yaml --ckpt_path experiments/<name>/version_0/checkpoints/last.ckpt
   ```
5. **Iterate on metrics**, not just loss — confirm the val metric relevant to the task moves, and record the config + checkpoint that produced it.

## Debugging a run

- API: construct `Policy`, `DataModule`, and `Trainer` directly in a short script or test to isolate whether failure is in object construction, dataloading, or CLI parsing.
- `--trainer.fast_dev_run=true` — one batch each stage; the first thing to try on any failure.
- `--print_config` — see the exact resolved config jsonargparse built.
- Shape/feature mismatches usually mean the datamodule's `Feature` names or action dim disagree with the policy — cross-check against the `library-adding-a-policy` skill.
- Dataset download stalls: the run is pulling a LeRobot `repo_id`; see the `library-working-with-datasets` skill.

## Required checks

- Config resolves (`--print_config`) and `fast_dev_run` passes before any long run.
- The equivalent Python API construction path passes for library-facing changes.
- `accelerator`/`devices` match the installed backend extra (`xpu`/`cuda`/`cpu`).
- New or renamed config fields stay consistent with the policy's `Config` class.
- Doc code blocks that show training commands still pass `tests/test_docs.py`.

## Verify

```bash
# from library/
physicalai fit --config configs/physicalai/<name>.yaml --trainer.fast_dev_run=true
uv run pytest tests/unit/train
```

For API-facing changes, add or run an equivalent Python smoke test (not a shell heredoc) that constructs `Policy`, `DataModule`, and `Trainer` directly and calls `trainer.fit(...)`.

## Related skills

- `library-adding-a-policy` — when the model itself needs changes.
- `library-working-with-datasets` — for the `data` half of the config.
- `library-benchmarking-a-policy` — to evaluate a trained checkpoint in a gym.
