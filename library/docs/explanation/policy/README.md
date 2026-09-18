# Policy Design

Native policies are lean Lightning orchestration layers around flat PyTorch models.
They own a serializable feature and architecture config, route every construction
source through `configure_model()`, and delegate shared lifecycle,
processing, export, and optional capabilities to their owning abstractions.

## Reading Order

1. [Policy Architecture](architecture.md) explains ownership, configuration, model
   materialization, runtime flow, and the checkpoint reconstruction.
2. [Required Interfaces](interfaces.md) identifies the minimum config, model, policy,
   processor, and temporal-context contracts.
3. [Implement a Policy](how-to.md) builds the minimal policy path and shows its
   explicit runtime and training flow.
4. [Advanced Patterns](advanced.md) adds PEFT/LoRA, RTC,
   checkpoint details, feature adaptation, and exceptional export customization.
5. [Export API](export.md) documents export-only properties, backend entry points,
   sample preparation, validation, and override boundaries.

Existing policies should follow [Policy Migration](migration.md), which stages the
move through feature ownership, reusable mixins, unified initialization, and export
cleanup.

## Implementation Rules

- Keep the concrete policy short enough to understand by inspection.
- Put ordered feature identity, architecture, and horizons in one policy-owned config.
- Keep normalization statistics in dataset or processor state.
- Keep model constructors flat, plain-typed, and config-free.
- Materialize model-dependent objects only in `configure_model()`.
- Return the model's full native action chunk before runtime adaptation.
- Inherit stable lifecycle and backend flows instead of repeating them per policy.
- Isolate optional capabilities and export behavior in cooperative mixins.

## Related Systems

- [Data](../data/README.md)
- [Trainer](../trainer/README.md)
- [Export system overview](../export/README.md)
