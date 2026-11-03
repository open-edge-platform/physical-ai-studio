# Policy Base Classes

Studio policies are Lightning modules that wrap PyTorch models for training, validation, testing, checkpointing, and inference.

## Expected Structure

```text
library/src/physicalai/policies/<policy_name>/
├── config.py
├── model.py
└── policy.py
```

## Key Contracts

- The policy wrapper inherits from the repo's `Policy` base class and uses the same action queue semantics as existing policies.
- `forward(...)` is the training path and should return values compatible with the policy's `training_step` and evaluation logic.
- `predict_action_chunk(...)` is the inference path and returns a tensor with the configured action horizon.
- `select_action(...)` should use the base policy behavior unless the policy has a well-justified specialized action selection flow.
- Config and checkpoint construction must stay compatible with Lightning and jsonargparse-driven CLI usage.

## Implementation Notes

- Start from a nearby native policy when possible. **`policies/pi05/`** is the best-maintained end-to-end example (train, export, inference). Use LeRobot wrappers only when the policy is intentionally a LeRobot integration.
- Keep policy-specific tensor transforms local to the policy package unless they are reusable across policies.
- Do not silently change action dimensions, feature names, or preprocessing conventions; these affect export and Runtime loading.
- If export support is added, validate the same sample input shape used during training/inference.
