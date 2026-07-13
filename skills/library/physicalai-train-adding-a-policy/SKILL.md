---
name: physicalai-train-adding-a-policy
description: Adds or modifies a Physical AI Studio policy under library/src/physicalai/policies. Use when creating a new policy family with the config/model/policy split, registering it in the get_policy factory and package exports, or keeping a policy compatible with Lightning training and export. Covers Pi0.5, Pi0, ACT, GR00T, SmolVLA, and LeRobot-wrapped policies.
license: Apache-2.0
---

# Adding a Studio Policy

Policies live in `library/src/physicalai/policies/<name>/`. Each family is a Lightning-facing `Policy` wrapping a `torch.nn.Module` `Model`, split across three files. Base classes are in `policies/base/` (`Policy` in `policy.py`, `Model` in `model.py`); the config base `Config` is in `library/src/physicalai/config/`.

## Workflow

1. **Read a nearby family first.** Study `policies/pi05/` (current reference implementation): `config.py` (`Pi05Config(Config)`), `model.py` (`Pi05Model(Model)`), `policy.py` (`Pi05(ExportablePolicyMixin, Policy)`), `preprocessor.py`, and any extra modules the architecture needs (e.g. `pi_gemma.py`). For a deliberately minimal family, `policies/act/` is a smaller three-file layout without the VLM stack.
   - Done when: you can name which existing file each new file mirrors.
2. **Create the three-file split** in `policies/<name>/`:
   - `config.py` — `<Name>Config(Config)`, all hyperparameters as typed fields.
   - `model.py` — `<Name>Model(Model)`, pure `torch.nn.Module` logic.
   - `policy.py` — `<Name>(Policy)` (add `ExportablePolicyMixin` only when export is implemented).
   - Done when: `from physicalai.policies.<name> import <Name>, <Name>Config, <Name>Model` imports cleanly.
3. **Implement the policy interface** used by both training and inference through the base `Policy`:
   - `forward(...)` — training path; return values compatible with `training_step`.
   - `predict_action_chunk(...)` — inference path; return a tensor with the configured action horizon.
   - `select_action(...)` — use base-class action-queue behavior unless a specialized flow is justified.
   - Done when: shapes match the checks below for a synthetic batch.
4. **Register the family** so both API and CLI users can find it:
   - Add exports to `policies/__init__.py` (`__all__` and imports, e.g. `<Name>`, `<Name>Config`, `<Name>Model`).
   - Add the lowercase name to the `get_physicalai_policy_class(...)` / `get_policy(...)` dispatch in `policies/__init__.py`.
   - Done when: `from physicalai.policies import <Name>, get_policy` works, `get_policy("<name>")` returns an instance, and `--model physicalai.policies.<Name>` resolves.
5. **Prove direct API construction** before adding CLI config:

   ```python
   from physicalai.policies import get_policy

   policy = get_policy("<name>")
   ```

   - Done when: direct construction, config round-trip, and synthetic `forward(...)` / `predict_action_chunk(...)` shape checks pass.

6. **Add a training config** in `library/configs/physicalai/<name>.yaml` when the policy is user-facing from the CLI. Wire `model.class_path`, a `data.class_path` (usually `physicalai.data.lerobot.LeRobotDataModule`), and `trainer.*`. Mirror `configs/physicalai/pi05.yaml`.
   - Done when: `physicalai fit --config configs/physicalai/<name>.yaml --trainer.fast_dev_run=true` completes one step.
7. **Wire export only when ready.** Add `ExportablePolicyMixin` and a valid sample input, then follow the `physicalai-train-exporting-and-validating` skill. If export is intentionally unsupported, say so explicitly in the policy docstring.
8. **Add tests** under `library/tests/unit/policies/` next to existing policy tests: at least one construction/config path and one shape-validation test.
   - Done when: `uv run pytest tests/unit/policies -k <name>` passes.
9. **Update docs** if the policy is user-visible: `library/docs/explanation/policy/` and any config/API examples.

## Required checks

Account for every item below (not just "looks fine"):

- **Action shape semantics** — batch, horizon/chunk length, and action dimension are correct and unchanged from the family's convention.
- **Observation features** — feature names align with dataset/config conventions (`data/observation.py`: `Feature`, `FeatureType`).
- **API construction path** — imports, `get_policy(...)`, direct constructor use, and synthetic shape checks pass without CLI involvement.
- **Config path** — construction works through the jsonargparse CLI path used by `physicalai fit` (`class_path`/`init_args`) when the policy is CLI-visible.
- **Heavy dependencies** — gate large families behind an optional extra in `library/pyproject.toml` and import lazily, matching `pi05`/`pi0`/`groot`/`smolvla`.
- **No silent contract changes** — do not alter action dims, feature names, or preprocessing without coordinating export/Runtime.

## Verify

From `library/`:

```bash
uv run pytest tests/unit/policies -k <name>
physicalai fit --config configs/physicalai/<name>.yaml --trainer.fast_dev_run=true
prek run --all-files library/
```

## References

- `references/base-classes.md` — the `Policy`/`Model` contract and file-split expectations.
