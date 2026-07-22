# Policy Configuration Architecture

## Status

This document records the recommended direction for policy configuration. It distinguishes current compatibility requirements from the longer-term architecture.

## Goals

- Define policy configuration once.
- Support typed Python construction, CLI, API, GUI, checkpoints, and export.
- Preserve existing YAML, checkpoint, and Runtime loading behavior while the architecture evolves.
- Make adding a policy or field mechanical and testable.
- Support schema evolution without silently changing behavior.

The goal is one authoritative **semantic schema**, not necessarily one physical representation at every boundary. Constructors, API payloads, UI schemas, and checkpoints have different requirements and may use explicit, validated projections of that schema.

## Current Constraints

The repository currently has these compatibility boundaries:

- The CLI uses `jsonargparse` to inspect `Policy` constructor signatures.
- Existing YAML uses flat `model.init_args` fields.
- Lightning restores first-party policies from flat constructor-compatible hparams.
- Torch export copies Lightning hparams, and Runtime loads the policy with `load_from_checkpoint()`.
- Existing policy code reads flat config fields such as `optimizer_lr`.
- `Config` can serialize nested dataclasses, but the generic `FromConfig` implementation does not map nested groups to differently named flat constructor arguments.
- Dynamic policy configuration is not yet exposed as a complete API or GUI schema.

These constraints explain why changing the constructor or persisted field names is currently expensive. They do not imply that flat constructors must remain the long-term source of truth.

## Current Problem

Policy configuration is duplicated across several surfaces:

1. Policy constructor parameters and defaults.
2. Policy config dataclass fields and defaults.
3. Constructor-to-config mapping code.
4. Lightning checkpoint hparams.
5. Future API and GUI schemas.
6. Upstream model or hub config mappings.

This duplication has already allowed defaults and field coverage to drift. Changing flat fields to nested fields would not solve that problem by itself; without central adapters it would add another mapping surface.

## Independent Decisions

The following are separate design dimensions, not mutually exclusive options.

### Construction API

Choices include explicit keyword arguments, `config=`, and `from_config()`.

### Domain Config Shape

The library config may be flat or contain nested domain components.

### API Validation

The backend may use Pydantic whether the library config is flat or nested.

### UI Grouping

Fields may be grouped using semantic metadata whether their persisted shape is flat or nested.

### Persistence

Checkpoint and serialized config representations require versioning and migration independently of the in-memory shape.

## Design Principles

### One Semantic Source Of Truth

The policy config class should own:

- stable field identifiers
- types and defaults
- semantic descriptions
- validation rules
- deprecation and aliases
- semantic group identifiers

Other schemas should be generated from it or checked mechanically against it.

### Explicit Boundaries

Conversions should be named and testable:

```text
API payload -> policy config -> constructor/checkpoint projection
hub config  -> policy config
old config  -> migration -> current policy config
policy config + UI overlay -> GUI schema
```

External model translation and schema migration are different operations and should not share implicit conversion logic.

### Strictness Over Silent Fallback

- Reject unknown customer-supplied fields by default.
- Reject ambiguous aliases and naming collisions.
- Validate unsupported field combinations before model initialization.
- Write only the current canonical format.
- Read older formats only through explicit migrations or temporary aliases.

The current generic dataclass deserializer ignores unknown keys. That behavior may be useful for selected upstream translations, but it should not be the default for user configuration.

## Recommended Architecture

### Short Term

1. Use a flat policy `Config` dataclass as the authoritative semantic schema.
2. Keep explicit flat policy constructor kwargs as a compatibility and convenience API.
3. Standardize `from_config()` across first-party policies.
4. Keep flat Lightning hparams for existing checkpoint and Runtime compatibility.
5. Put stable semantic grouping metadata in the library.
6. Keep labels, order, widgets, visibility, and product copy in the application.
7. Use Pydantic at the API boundary as a generated or parity-tested projection.
8. Add schema versions, strict loading, migrations, and compatibility tests before renaming or nesting persisted fields.

This is Option C from the earlier discussion combined with an application API schema layer. Pydantic is complementary to Option C, not an alternative to it.

### Long Term

1. Make CLI and checkpoint infrastructure config-aware.
2. Remove duplicated defaults from policy constructors where compatibility permits.
3. Consider `config=` as the core internal construction path.
4. Retain explicit convenience construction only where it remains useful.
5. Introduce nested configs only for genuine reusable domain components.

The migration to a config-centric constructor must include old-checkpoint handling. It should not be implemented as a signature-only refactor.

## Constructor Guidance

### Explicit Arguments

```python
policy = Pi05(chunk_size=100, optimizer_lr=1e-5)
```

Explicit arguments are appropriate when:

- the parameter count is small or moderate
- users commonly construct the class directly
- frameworks inspect the signature
- the arguments are a stable public API

Strengths:

- IDE discovery and static typing
- immediate rejection of misspelled arguments
- direct CLI generation
- direct Lightning checkpoint restoration

Costs for large policy signatures:

- duplicated defaults and validation
- repetitive constructor-to-config mapping
- accidental field omissions
- mixed architecture, training, inference, loading, and runtime concerns

For the current repository, explicit constructors should be treated as a supported compatibility boundary, not as the authoritative schema.

### Config Argument

```python
policy = Pi05(config=Pi05Config(...))
```

A config-based constructor scales better when a model has many evolving parameters because defaults, validation, serialization, and composition can have one owner.

It is not the immediate recommendation because current CLI and checkpoint restoration inspect or invoke the flat constructor. A future transition is viable once those paths reconstruct the typed config explicitly.

If `config=` is introduced, avoid ambiguous APIs such as:

```python
Pi05(config=config, optimizer_lr=1e-5)
```

Do not merge config values and overriding kwargs unless precedence is explicit, narrow, and justified. Prefer creating a new validated config first.

### `from_config()`

```python
policy = Pi05.from_config(Pi05Config(chunk_size=100))
```

`from_config()` is the recommended adapter while flat constructors remain necessary.

Its contract should be:

- accept the policy's exact config type
- validate before constructing the policy
- map every supported config field exactly once
- reject unsupported fields instead of dropping them
- preserve tuples, enums, optional values, and other typed values
- produce only arguments accepted by the constructor

For a flat config with identical names, the adapter can be mechanical:

```python
@classmethod
def from_config(cls, config: Pi05Config) -> Self:
    return cls(**config.to_init_kwargs())
```

`to_init_kwargs()` should exclude fields that are intentionally not constructor inputs and make that allowlist visible. It should not depend on silent filtering.

Hub-specific conversion should remain separate:

```python
config = Pi05Config.from_hf_config(hf_config)
policy = Pi05.from_config(config)
```

## Config Structure

### Flat By Default

```python
@dataclass(frozen=True)
class Pi05Config(Config):
    chunk_size: int = field(default=50, metadata={"group": "io"})
    optimizer_lr: float = field(default=2.5e-5, metadata={"group": "optimizer"})
```

A flat shape is the current recommendation because it preserves existing field identifiers and requires the fewest adapters.

Flat storage does not prevent a grouped GUI or API documentation.

### Nest Only Real Domain Components

Nesting is appropriate when a child configuration is:

- independently meaningful and validated
- reusable across policies
- replaceable as a unit
- naturally configured and persisted as a unit

Potential examples are an optimizer or scheduler specification shared by multiple policies.

Nesting should not be introduced only to match a UI page, accordion, or an `advanced` section. Presentation structure changes more frequently than the model's semantic contract.

Before adopting nested persisted fields, define:

- canonical paths and aliases
- flatten and unflatten behavior, if required
- collision rejection
- old-schema migration
- constructor and checkpoint projections
- API payload compatibility

## Config Content

Separate durable semantic configuration from construction instructions and runtime state.

Durable configuration generally includes architecture, feature dimensions, preprocessing semantics, normalization, and training hyperparameters needed to reproduce behavior.

Review these categories before persisting them as policy config:

- hub or checkpoint paths are loading instructions
- dataset statistics may be learned or dataset-derived artifacts
- device placement is runtime state
- compilation settings may be execution policy rather than model semantics
- caches and temporary paths are never semantic configuration

When loading from an upstream model, persist the resolved semantic config rather than relying only on the original loading instruction.

## Immutability And Validation

Policy configs should normally be frozen after validation:

```python
@dataclass(frozen=True)
class Pi05Config(Config):
    ...
```

Use `dataclasses.replace()` to derive a modified config. Mutable model and training state belong on the policy, not in its config.

Validation belongs in the authoritative config and should cover:

- field-level constraints
- cross-field constraints such as `n_action_steps <= chunk_size`
- unsupported combinations
- backend-specific restrictions where they are semantically relevant

API validation may provide earlier user feedback, but it must not define conflicting defaults or rules.

## Metadata Ownership

### Library

- group identifier
- field description
- validation constraints or hints
- deprecation state
- stable aliases during migration

### Application

- display labels
- field and group order
- widgets
- advanced or hidden presentation
- conditional layout
- product-facing copy

Example:

```python
@dataclass(frozen=True)
class Pi05Config(Config):
    optimizer_lr: float = field(
        default=2.5e-5,
        metadata={"group": "optimizer", "description": "Adam learning rate."},
    )
```

```python
PI05_UI_OVERLAY = {
    "optimizer": {"label": "Optimizer", "order": 90},
    "optimizer_lr": {"label": "Learning Rate", "widget": "number"},
}
```

## API And GUI

Pydantic should be used at the FastAPI boundary for request validation and OpenAPI generation. It should project from or map explicitly to the library config.

```python
class TrainJobPayload(BaseModel):
    policy: Literal["pi05", "act", "smolvla"]
    policy_config: Pi05ConfigModel | ACTConfigModel | SmolVLAConfigModel
```

Avoid manually maintaining equivalent defaults independently in dataclasses, constructors, and Pydantic models. If generation is not practical, enforce parity in CI.

The GUI schema should remain a separate response contract:

```python
class ConfigFieldSchema(BaseModel):
    path: str
    type: str
    default: Any
    description: str | None = None
    validation: dict[str, Any] = Field(default_factory=dict)


class ConfigGroupSchema(BaseModel):
    id: str
    label: str
    order: int = 100
    fields: list[ConfigFieldSchema]
```

This schema can group flat config fields without changing the accepted training payload or checkpoint format.

## Persistence And Evolution

Serialized customer config and checkpoint config should carry a schema version, either in the config or its envelope:

```json
{
  "schema_version": 2,
  "policy": "pi05",
  "config": {
    "chunk_size": 100,
    "optimizer_lr": 0.00001
  }
}
```

Evolution rules:

- Load current schemas strictly.
- Migrate older versions through explicit ordered functions.
- Test migrations using committed old-format fixtures.
- Use aliases only for intentional compatibility windows.
- Emit only the latest canonical names.
- Do not silently ignore misspelled or removed fields.
- Keep flat checkpoint hparams until Runtime loading no longer depends on them.

## Required Invariants And Tests

Every first-party policy should have tests for:

1. Config fields map completely to construction inputs or an explicit exclusion list.
2. Shared constructor and config defaults are identical.
3. `from_config(config)` preserves all supported values.
4. Config serialization round-trips without type loss.
5. Unknown user fields fail clearly.
6. CLI config performs real policy instantiation, not only parser validation.
7. A checkpoint restores with the expected resolved config.
8. At least one prior schema/checkpoint fixture migrates successfully after a schema change.
9. API models and GUI schemas cover all intended config fields.
10. Library semantic metadata and application overlays contain no stale field references.

These tests are more important for scalability than whether the dataclass is flat or nested.

## Decision Summary

| Dimension | Current Decision | Future Direction |
| --- | --- | --- |
| Semantic source of truth | Flat library `Config` | Remains authoritative; may contain real domain components |
| Policy constructor | Explicit flat kwargs | Config-centric after CLI/checkpoint migration |
| `from_config()` | Standard adapter into kwargs | May become the primary construction path |
| Checkpoint hparams | Flat constructor-compatible keys | Versioned typed config when Runtime supports it |
| UI grouping | Library semantic group + app overlay | Same |
| API validation | Pydantic projection in application | Generated where practical |
| Persistence | Existing flat fields | Versioned, strict, explicitly migrated |

## Decision

Adopt the following baseline:

- flat `Config` dataclasses are authoritative for now
- explicit flat constructors remain supported for compatibility
- `from_config()` is standardized and parity-tested
- library metadata owns semantic grouping
- application overlays own presentation
- Pydantic provides the API projection
- schema evolution is versioned and strict
- nested configuration is reserved for genuine reusable domain boundaries

This direction preserves the existing pipeline while creating a controlled path away from duplicated constructor schemas.
