# Policy Migration

Migrate one policy at a time. Keep tests passing after each phase.

See [Current and Proposed Policy Construction](current-policies.md) for existing
policy patterns.

## Legacy Arguments

Keep existing constructor and CLI arguments during migration.

- Group legacy arguments at the end of the constructor.
- Translate them in one compatibility helper.
- Mark them as deprecated in CLI metadata when supported.
- Emit a `FutureWarning` when used.
- Name the replacement and removal version.
- Reject conflicts between old and new arguments.
- Test old and new arguments against the same resolved config.
- Remove them only after the deprecation window.

Do not add legacy branches to `configure_model()` or processors.

## 1. Input and Output Features

Make the config the feature source of truth.

### Changes

1. Add ordered `input_features` and `output_features` to the config.
2. Add derived model dimensions such as `action_dim`.
3. Stop deriving feature identity from `dataset_stats`.
4. Keep normalization values in dataset or processor state.
5. Read dataset features directly in `setup("fit")`.
6. Validate dataset features against an existing config.
7. Pass plain dimensions to the model. Do not pass `Feature` objects.
8. Return the full action chunk from the model.
9. Apply denormalization, dimension changes, and `n_action_steps` afterward.

### Complete When

- Config serialization preserves feature names, types, shapes, and order.
- Model construction does not need `dataset_stats`.
- Changing normalization does not change model architecture.
- Dataset mismatches produce a clear error.
- The model returns `(batch, chunk_size, model_action_dim)`.

## 2. Reusable Mixins

Replace policy-local capability code with existing mixins.

### Changes

1. List local capability flags, helpers, and checkpoint fields.
2. Use config, model, and policy mixins at their matching layers.
3. Use the PEFT mixins for LoRA.
4. Keep default LoRA targets on the model.
5. Use the RTC model and policy mixins.
6. Keep gradient-checkpointing behavior on the model.
7. Remove duplicate hooks and helpers.
8. Preserve cooperative `super()` calls.

Apply capabilities in this order:

```text
construct model
    -> load pretrained base weights
    -> inject LoRA
    -> restore checkpoint tensors
    -> restore runtime state
```

### Complete When

- LoRA is injected once before checkpoint loading.
- RTC state survives checkpoint restoration.
- RTC receives the full action chunk.
- Disabling a capability preserves base behavior.
- The policy contains no duplicate capability logic.

## 3. Config and `configure_model()`

Use one model materialization path.

### Changes

1. Put all reconstruction values in the policy config.
2. Keep the model constructor flat and config-free.
3. Create the model and processors only in `configure_model()`.
4. Start with `if self.model is not None: return`.
5. Keep dataset interaction in `setup(stage)`.
6. Make pretrained helpers return config and weight paths only.
7. Route every construction source through `configure_model()`.
8. Translate legacy arguments before config resolution.
9. Save the resolved config in checkpoints.
10. Clear pretrained paths before checkpoint construction.
11. Materialize the model before Lightning loads checkpoint tensors.
12. Delete old `_initialize_model()` and `_initialize_policy()` helpers.

| Route | Config source | Weight source |
| --- | --- | --- |
| Explicit config | Caller | None |
| Constructor | Features and defaults | None |
| Training | Config validated against dataset | None |
| Pretrained | Artifact config | Artifact weights |
| Checkpoint | Saved config | Lightning state dictionary |

### Complete When

- Every route calls `configure_model()`.
- Calling it twice keeps the same model.
- Pretrained resolution does not create a model.
- Checkpoint loading does not fetch external weights.
- Config, model, and processors agree on features and horizons.
- Legacy and replacement arguments resolve to the same config.
- Conflicting arguments fail clearly.

## 4. Export and Final Refactors

Move export code out of the core policy.

### Changes

1. Put export behavior in a policy-specific export mixin.
2. Move `inputs_schema`, `outputs_schema`, and `sample_input` there.
3. Move `extra_export_args` and supported backends there.
4. Derive schemas from the policy config.
5. Return `None` when config is unavailable.
6. Raise clear errors for invalid resolved configs.
7. Express backend options through `extra_export_args`.
8. Add a chunk trimmer when `chunk_size != n_action_steps`.
9. Inherit `to_onnx()` and `to_openvino()` when possible.
10. Keep narrow backend overrides only when required.
11. Remove stale feature, initialization, and checkpoint code.
12. Keep deprecated public arguments until their removal version.

### Complete When

- Export schema order matches runtime order.
- Samples have the correct names, shapes, dtypes, and device.
- Every advertised backend exports successfully.
- Unsupported backends are not advertised.
- Restored policies export without the original pretrained artifact.
- The core policy is readable without opening the export module.

## Final Checks

Run:

- policy unit tests;
- config and checkpoint round trips;
- one training step;
- one inference pass;
- each supported export backend;
- legacy argument tests.

The migration is complete when the config owns the feature contract, mixins own
reusable capabilities, `configure_model()` owns materialization, and export lives in
its dedicated mixin.

## Related Pages

- [Policy Architecture](architecture.md)
- [Required Interfaces](interfaces.md)
- [Advanced Patterns](advanced.md)
- [Export API](export.md)
