# Studio contributor-skill scenarios

Run from the Studio checkout. Reset agent context between prompts; check that the matching contributor skill activates and focused source tests pass. Customer API workflows should instead use `skills/using/`.

## `pai-add-policy`

1. "Add a native `mynet` policy family under `physicalai.policies`." Expect config/model/policy split, `get_policy` registration, synthetic shape tests, and CLI smoke config.
2. "Make ACT exportable without changing action semantics." Expect a valid sample input, `ExportablePolicyMixin`, a parity check with `pai-export-policy`, and tests.
3. "Port a LeRobot-wrapped family into first-party Studio policy code." Expect separate first-party files, direct Python construction, and optional dependency checks.

## `prepare-release`

1. **Coordinated app and library release (`requires_download`)**: Ask for a Studio and training-library release. It should use each package's `pyproject.toml` version, update compatibility floors where needed, regenerate locks, write separate notes, and publish the library before the app.
2. **App-only patch release**: Ask to release `physicalai-studio` without a library API change. It should bump only the app project version, read the API version from installed distribution metadata, and use the `app/vX.Y.Z` tag.
3. **Premature tag request**: Ask to tag an unmerged branch or publish while checks/reviews are pending. It should require a merged main commit, passing checks, and explicit publish authorization; it must not bypass protection.

## `pai-add-benchmark`

1. "Add a new first-party gym benchmark with a success metric." Expect a `Benchmark` subclass, results schema, unit tests, and a one-episode evaluation.
2. "Expose a new benchmark suite through the library CLI." Expect a config under `library/configs/benchmark/`, matching API and CLI metrics, and optional extras.
3. "Fix a benchmark that reports zero success despite completed episodes." Expect a focused regression test and a comparison against a known-good baseline.

## `pai-add-robot-form-field`

1. "Add a new upload field kind in `robot_payload_ui`." Expect SDK validation, kind-based React renderer, and UI tests.
2. "Keep advanced/required visibility consistent for the new kind." Expect shared visibility rules and a failing-then-passing test.
3. "Show plugin authors how to adopt the field." Expect Studio docs, SDK examples, tests, and explicit rollout guidance.
