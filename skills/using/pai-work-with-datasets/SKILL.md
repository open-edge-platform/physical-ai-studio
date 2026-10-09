---
name: pai-work-with-datasets
description: Works with Physical AI Studio datasets and Lightning datamodules using the LeRobot format. Use when wiring LeRobotDataModule into a project training config, choosing a repo_id, inspecting batch shapes, converting between physicalai and lerobot layouts with FormatConverter, or debugging dataloading. DO NOT USE FOR changing the Studio library's converters or dataset implementation.
license: Apache-2.0
---

# Working with Studio Datasets

Studio datasets use the **LeRobot format** and are consumed through Lightning datamodules. The datamodules are first-class Python API objects; YAML/CLI configs are a serialization of the same construction path. These modules ship with `physicalai-train`:

Key modules:

- `data/lerobot/datamodule.py` — `LeRobotDataModule` (the class configs reference as `physicalai.data.lerobot.LeRobotDataModule`).
- `data/lerobot/dataset.py` — LeRobot dataset wrapper.
- `data/lerobot/converters.py` — `DataFormat` and public `FormatConverter.to_lerobot_dict(...)` / `to_observation(...)`.
- `data/observation.py` — `Observation`, `Feature`, `FeatureType`, `NormalizationParameters` (match your policy's features).
- `data/datamodules.py` — base `DataModule` (Lightning `LightningDataModule`, auto num-workers heuristic).
- `data/dataset.py` — base `Dataset`; `data/gym.py` — `GymDataset` for gym-generated data.

## Python API usage

Use this path for notebooks, tests, direct batch inspection, or debugging dataloading without involving the training CLI.

```python
from physicalai.data import LeRobotDataModule

datamodule = LeRobotDataModule(repo_id="lerobot/pusht", train_batch_size=2)
datamodule.prepare_data()
datamodule.setup("fit")
batch = next(iter(datamodule.train_dataloader()))
```

Done when: the batch contains the observation/action fields the policy expects, with the expected batch/action dimensions.

## Wiring data into a training config

In a `physicalai fit` config, the `data` block selects the datamodule and its `repo_id`:

```yaml
data:
  class_path: physicalai.data.lerobot.LeRobotDataModule
  init_args:
    repo_id: lerobot/pusht
    train_batch_size: 64
```

`repo_id` points at a LeRobot/HuggingFace dataset; the datamodule pulls it on first use. See `pai-train-policy` for the full config.

## Workflow

1. **Pick the dataset** by `repo_id` and confirm its features (image keys, state dim, action dim) match the target policy's `Config`.
   - Done when: the policy's expected `Feature` names and action dimension line up with the dataset.
2. **Verify a batch through the Python API** before training:
   ```python
   datamodule.prepare_data()
   datamodule.setup("fit")
   batch = next(iter(datamodule.train_dataloader()))
   ```
   - Done when: the batch has correct keys and shapes without invoking the CLI.
3. **Verify CLI parity** when the dataset is configured through YAML:
   ```bash
   physicalai fit --config <config.yaml> --trainer.fast_dev_run=true
   ```
   - Done when: one batch flows through with correct shapes and no missing-feature errors.
4. **Convert layouts** only when needed via `FormatConverter.to_lerobot_dict(batch)` or `FormatConverter.to_observation(batch)`; keep field names stable, since they propagate to training and export.
5. **Set normalization** through `NormalizationParameters`/`Feature` consistently with what the selected policy expects at inference.

## Debugging dataloading

- Missing/renamed feature → the dataset features disagree with the policy; align the user's config or data mapping with the policy's `Feature` names.
- Slow/stalled first batch → the LeRobot `repo_id` is downloading; expected on first run (see the `requires_download` test marker for tests that need this).
- Wrong batch dimensions → check `train_batch_size` and the produced batch keys before changing the project config.
- OOM or heavy swapping → try `pin_memory=False` and/or `persistent_workers=False` on the `DataModule`; see the [datamodule guide](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/explanation/data/datamodules.md).

## Required checks

- Feature names, `FeatureType`, action dim, and normalization match between dataset, `Config`, and any export metadata.
- Conversions round-trip without dropping or renaming fields.
- Direct datamodule API construction and YAML config construction produce compatible batches.
- Downloading a new dataset requires network access; ask before starting a large transfer.

## Verify

Inspect one train batch in the project's Python environment, then run `physicalai fit --config <config.yaml> --trainer.fast_dev_run=true` when the dataset is used from YAML.

## Related skills

- `pai-train-policy` — the `data` block is one half of a training config.
