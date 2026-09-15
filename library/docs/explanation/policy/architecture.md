# Policy Architecture

A policy connects a PyTorch model to data, training, inference, checkpoints, and
export. Keep the concrete policy short enough to read in one pass.

> Every construction route resolves one complete config and calls
> `configure_model()`.

## Structure

```text
policy_name/
|-- config.py        # Features and model settings
|-- model.py         # Network, loss, and prediction
|-- policy.py        # Construction and Lightning flow
|-- preprocessor.py  # Input conversion and normalization
`-- postprocessor.py # Output conversion and denormalization
```

Small policies may combine files. Keep the same ownership boundaries.

## Ownership

| Owner | Responsibility |
| --- | --- |
| Config | Ordered features, architecture, and action horizons |
| Policy | Construction, processing flow, training, and optimizer settings |
| Base policy | Config property, checkpoint plumbing, device transfer, and action queue |
| Dataset | Observed features and normalization values |
| Model | Network, loss, temporal indices, and full-chunk prediction |
| Processors | Conversion, normalization, and external action shape |
| Pretrained resolver | Artifact config and weight paths |
| Capability mixins | PEFT, RTC, and other shared behavior |
| Export mixin | Schemas, samples, backend settings, and manifests |

The config does not contain optimizer settings, artifact paths, or export destinations.
The model does not store the config or accept `Feature` objects.

## Feature State

The config owns feature names, types, shapes, and order. Dataset or processor state
owns means, standard deviations, quantiles, and other normalization values.

The dataset validates its features against the config. It does not define model
architecture through `dataset_stats`.

## Materialization

`configure_model()` is the only method that creates the model and processors.

```text
resolve config
    -> configure_model
    -> create model
    -> create processors
    -> load base weights
    -> apply reconstruction capabilities
```

It starts with:

```python
if self.model is not None:
    return
```

This prevents duplicate models, weight loss, and repeated adapter injection.

Use this order:

1. Store the resolved config.
2. Create the model.
3. Create processors.
4. Load pretrained base weights.
5. Apply state-dict-shaping capabilities.
6. Sync runtime capabilities.

### Lightning Constraints

`configure_model()` is a Lightning lifecycle hook, so implementations must follow
these constraints:

| Constraint | Required behavior |
| --- | --- |
| Lightning may call the hook more than once | Make it idempotent and return when the model already exists |
| Strategies such as FSDP control module creation | Create large modules inside the hook, not before Lightning invokes it |
| Checkpoint tensors load after module construction | Rebuild the exact module and adapter structure from saved config first |
| Checkpoint restore must be self-contained | Do not fetch a pretrained artifact or overwrite checkpoint weights |
| Dataset state may arrive later in `setup()` | Keep architecture independent of normalization state and do not rebuild the model |
| Manual inference may call the hook without a Trainer | Do not depend on trainer, dataloader, or device state during construction |
| Config fields and defaults may change | Version or migrate saved config so old checkpoints rebuild the same structure |

Keep the hook deterministic and free of network access. It should materialize module
structure from resolved config; `setup()` may validate dataset features and attach
normalization state without changing that structure.

## Construction Routes

| Route | Config source | Weight source |
| --- | --- | --- |
| Explicit config | Caller | None |
| Constructor | Features and defaults | None |
| Training | Config validated against dataset | None |
| Pretrained | Artifact config | Artifact weights |
| Checkpoint | Saved config | Lightning state dictionary |

All routes call `configure_model()`. Pretrained resolvers do not create models.
Checkpoint loading does not fetch the original artifact.

## Runtime Flow

```text
Observation
    -> preprocessor
    -> model
    -> full action chunk
    -> runtime capabilities
    -> postprocessor
    -> action queue
```

The model returns `(batch_size, chunk_size, model_action_dim)`. Postprocessing trims to
`n_action_steps` and removes padded dimensions.

## Checkpoints

Save the resolved config. During restore:

1. clear the pretrained path;
2. load the saved config;
3. call `configure_model()`;
4. rebuild state-dict-shaping capabilities;
5. let Lightning load tensors;
6. restore runtime state.

See [Advanced Patterns](advanced.md) for details.

## Related Pages

- [Required Interfaces](interfaces.md)
- [Implement a Policy](how-to.md)
- [Advanced Patterns](advanced.md)
- [Export API](export.md)
- [Policy Migration](migration.md)
