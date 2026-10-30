# Human In The Loop Plan

## Why?

Human in the loop (HIL) runs a model on the follower and lets the operator take over with the leader when they see a failure. That puts the data where the model is weakest: the moments it fails, and how a human recovers from them.

There are two forms:

- **Basic:** only the takeovers are recorded. Each takeover is a plain teleop episode into any dataset.
- **Full:** the whole rollout is recorded, model frames included, with the source of each frame's action (`model`, `teleop`, ...). Afterwards a value function, trained on the original dataset, scores every frame of the HIL dataset, and `value` + `source` steer training.

The full form needs features a normal dataset does not have, and that is what this plan is about: where those features live without making datasets with and without them incompatible everywhere.

## What the code actually constrains

- **Recording is the only sharp edge.** LeRobot's `validate_frame` (`lerobot/datasets/feature_utils.py:221`) wants every frame to carry exactly the dataset's features, no more and no fewer. So a dataset with `source` cannot be recorded into without one, and one without it cannot take a frame that has one.
- **Training does not care.** The library passes unknown columns through as `Observation.extra` (`library/src/physicalai/data/lerobot/converters.py:129`), and policies pick only state, images and action. An extra column breaks no existing training.
- **The session spec does not see it.** `dataset_spec` (`runtime_session/dataset.py:55`) reads only `observation.state`, `action` and `observation.images.*`. `source` and `value` never reach `FeatureSpec`, so they need no change to it or to the store, the model or the environment checks.
- **LeRobot already handles both dtypes and rewrites.** String features are validated, and per-episode stats skip them (`compute_stats.py`). `dataset_tools.add_features` / `remove_feature` rewrite a dataset with columns added or removed. LeRobot's own DAgger (`lerobot/rollout/strategies/dagger.py`) adds `intervention: bool[1]`, only in the `rollout_*` datasets it creates. That is the "specific spec" approach.

## Recommendation: separate what is recorded from what is derived

1. **`value` is derived, never recorded.** It is computed after the fact by a job that runs the value model over a dataset and writes a copy with a `value` column through `add_features`. That copy is the training input; the recorded dataset is never touched. So `value` puts nothing in the recording path.

2. **`source` is the one recorded extra, kept out of `FeatureSpec`.** Treat a dataset's non-joint, non-camera features as *frame annotations* that the recording supplies from the session, not from the store's shape.
   - `dataset_spec` → also returns the dataset's annotation keys (anything outside state, action and images, and outside LeRobot's default features).
   - `Recording` / `DatasetWorker.record_frame` (`workers/dataset_worker.py:101`) adds `source` to each frame when the dataset has it, and leaves it out otherwise.
   - What fills it: the control that wrote the action. A HIL control switches source inside one control, so this has to be per write: the store gains a small `action_source` slot (an int code, written by the control alongside `ACTION`), mapped to the string when the frame is recorded. Each control names its source: `teleop`, `model`, `replay`, `move`; HIL writes `model` or `teleop`.

3. **Full-form HIL records into a dataset of its own** (decided), like LeRobot's `rollout_*` datasets. `source` is a string per frame (decided).
   - The HIL mode creates a new dataset with `source`, linked to the dataset the model was trained on. That link is what the value job needs: it trains on the original and scores the HIL one.
   - A dataset *with* `source`, recorded by teleop or replay: gets `teleop` / `replay`. No edge.
   - A dataset *without* `source`: unchanged, recorded as today. Full-form HIL into it is refused with "HIL records into its own dataset". No upgrade flow is needed.
   - Basic-form HIL (only takeovers recorded) needs no `source` at all. It is a plain teleop recording that starts and stops on takeover, into any dataset.

4. **Dataset-level edge to accept:** `merge_datasets` needs equal features, so a HIL dataset cannot be merged with a plain one as is. If that is ever wanted, `remove_feature("source")` on a copy, or `add_features` filling `teleop` on the plain one, covers it. Not built now.

## Alternatives considered (and why not)

- **A HIL-specific spec with optional parts:** this pushes the difference into `FeatureSpec`, which then has to model optional features for every check (environment, model, dataset). It is the same information as the annotation keys above, but in the wrong layer.
- **A sidecar file with per-frame annotations next to the LeRobot data:** keeps the LeRobot schema untouched. But it drifts out of step on episode deletion, splits and merges, which reindex frames, and every reader has to join it.
- **Episode-level intervention intervals in the episode metadata:** compact, since takeovers are contiguous. But LeRobot's dataset tools do not carry custom episode columns reliably, and the per-frame form is what training wants anyway.

## Files touched

- `runtime_session/dataset.py`: `dataset_spec` returns annotation keys; `lerobot_features` adds `source` for new HIL datasets.
- `runtime_session/store.py`: an `action_source` slot; `Store.write(ACTION, ..., source=...)`.
- `runtime_session/controls/*.py`: each control names its source; a new `controls/hil.py`.
- `workers/dataset_worker.py`: `record_frame` fills annotations.
- A value-function job (training side) that writes the derived dataset with `add_features`.

## Verification

- Unit: `record_frame` with and without `source` in the dataset; the HIL control switches `action_source` on takeover.
- Full HIL into a dataset without `source` is refused; the HIL mode's new dataset has `source` (string) in `info.json`.
- Training an ACT model on a dataset with `source` (and a derived `value`) runs unchanged.
- Hardware: an SO-101 rollout with two takeovers; check the `source` column flips at the takeover frames.
