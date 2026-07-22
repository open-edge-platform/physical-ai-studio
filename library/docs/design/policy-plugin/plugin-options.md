# Policy Plugin and Model Enablement Strategy

> Status: design proposal. The APIs in this document are illustrative
> pseudocode, not implemented public APIs.

## Executive Summary

Physical AI does not need a second policy abstraction or several kinds of
policy plugins. The existing `Policy` class and the composition-based LeRobot
wrapper are the right foundation.

The proposal is to make an installed `Policy` subclass the unit of extension:

```text
Python entry point -> Physical AI Policy class -> upstream native policy
```

An integration package can contribute one or more policy classes through
standard Python entry points. Each class:

- Implements the existing Physical AI and Lightning policy interface.
- May contain an unchanged upstream policy object.
- May wrap an existing upstream object with `wrap(native)`.
- Reports implementation capabilities through class methods and attributes.
- Uses the existing export interface for Torch, ONNX, OpenVINO, or
  ExecuTorch support.

Physical AI keeps the existing `get_policy()` factory. It learns to resolve
installed entry points instead of relying on hard-coded source and policy
lists. Studio builds its catalog by inspecting those discovered policy
classes.

The design deliberately does not introduce:

- A `PolicyPlugin` object.
- A public `Policies` or registry object.
- Separate batch, training, optimization, action, artifact, or dataset
  adapter facets.
- A new training runner or executor protocol.
- A universal abstraction over upstream framework APIs.

This is a refinement of the existing policy integration, not a replacement
for it.

## Objective

Physical AI's model-enablement goal is to support as many useful policies as
possible on Intel hardware while minimizing forks, copied implementations,
and permanent downstream maintenance.

The preferred result is support in the original framework or model
repository. When that is not immediately possible, Physical AI should provide
the narrowest downstream integration that meets product requirements and
continue contributing reusable fixes upstream.

The strategy is:

1. Use generic adapters to increase availability without creating a
   model-specific support commitment.
2. Add a named integration when Studio commits to documented workflows,
   tests, compatibility, and maintenance.
3. Attempt OpenVINO through the adapter or upstream model before building a
   first-party implementation.
4. Assess OpenVINO feasibility for every Studio-supported model. OpenVINO is
   required for Deployment-supported status.
5. Evaluate quantization for every OpenVINO-enabled model. Adopt it only when
   measured deployment benefit justifies the quality and maintenance cost.
6. Build first-party implementations only when adapters cannot meet product,
   dependency, export, performance, or maintenance requirements.
7. Deliver downstream fixes when needed while contributing reusable changes
   to frameworks, model repositories, PyTorch XPU, OpenVINO, and related
   projects.

## Design Principles

### Upstream remains authoritative

When Physical AI wraps an upstream policy, the upstream implementation,
configuration, processors, and native artifact format remain the source of
truth. Physical AI should not copy model code merely to make it fit the
Lightning interface.

### One integration mechanism

Generic adapters, named integrations, and first-party implementations should
all be ordinary `Policy` subclasses. Their support commitments differ, but
their integration mechanism does not.

### Class-centric discovery

The policy class owns its behavior and reports its capabilities. Entry points
only make classes discoverable; they do not create a second object model.

### Support is evidence-based

Method existence can establish that code implements an operation. It cannot
establish that XPU training, OpenVINO export, numerical parity, or deployment
passed validation. Verified workflows must be explicit and backed by tests.

### Escalate ownership only when necessary

Prefer upstream use, then a wrapper, then a narrow compatibility shim or
downstream patch, and only then a first-party implementation.

## Model Enablement Levels

The levels below describe increasing Physical AI ownership. They are not
different plugin types.

| Level                      | Implementation                                                       | Physical AI commitment                                            |
| -------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Generic adapter            | Wrap an upstream policy through a framework-level `Policy` class     | Best-effort availability; no model-specific compatibility promise |
| Named integration          | A stable, model-specific `Policy` class built on the generic adapter | Documented and tested Studio workflows with pinned compatibility  |
| First-party implementation | Model implementation owned by Physical AI                            | Full implementation, XPU, export, and maintenance ownership       |

### Generic adapter

A generic framework adapter lets users try upstream models without Physical AI
claiming that every model registered by that framework is supported.

The existing `LeRobotPolicy` is the reference pattern:

```text
Physical AI LeRobotPolicy (Policy / LightningModule)
└── native LeRobot PreTrainedPolicy
    ├── native config
    ├── native model
    ├── native processors
    └── native serialization
```

Users can select an upstream policy dynamically:

```python
from physicalai.policies.lerobot import LeRobotPolicy

policy = LeRobotPolicy(
    policy_name="vqbet",
    config=native_config,
)
```

This is an availability path, not a claim that VQ-BeT has passed Physical AI
XPU, training-parity, export, or deployment qualification.

### Named integration

A named class represents an intentional support commitment:

```python
from physicalai.policies.lerobot import ACT

policy = ACT(config=native_config)
```

The named class should remain a thin specialization of the generic adapter.
It establishes a stable import path, stable policy ID, documented workflow,
and model-specific qualification tests without duplicating the upstream model.

### First-party implementation

A first-party policy remains appropriate when an upstream adapter cannot meet
requirements such as:

- Reliable XPU training or fine-tuning.
- Acceptable dependency compatibility.
- OpenVINO conversion or Runtime deployment.
- Required performance or memory behavior.
- Stable maintenance against supported upstream versions.
- Required product customization that upstream cannot accept.

First-party implementation is the final escalation step, not the default
enablement path.

## User-Facing APIs

There are two user-facing construction paths. Both produce a normal Physical
AI `Policy`.

### Physical-AI-first construction

Physical AI constructs the registered policy from a stable ID:

```python
from physicalai.policies import get_policy

policy = get_policy(
    "lerobot.act",
    config=native_config,
)
```

`get_policy()` performs only class resolution and construction:

```text
get_policy("lerobot.act", ...)
    -> find the installed policy entry point
    -> load the Policy subclass
    -> instantiate the class with the supplied arguments
```

It does not train, export, launch another environment, or create a service.

Existing source-based calls can resolve to the same class during migration:

```python
policy = get_policy("act", source="lerobot", config=native_config)
```

### Native-first wrapping

Users may construct an upstream object with the original framework API and
wrap that exact object:

```python
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from physicalai.policies.lerobot import ACT

native = ACTPolicy(ACTConfig(...))
policy = ACT.wrap(native)

assert policy.native is native
```

`wrap()` is intentionally used instead of `from_native()` because the native
object is retained rather than converted, copied, or reconstructed.

The wrapper enables Physical AI workflows:

```python
from physicalai.train import Trainer

Trainer(max_epochs=100).fit(policy, data)
action = policy.select_action(observation)
```

The original framework API remains available on the same object:

```python
native_action = policy.native.select_action(native_batch)
policy.native.save_pretrained("artifacts/act")
```

Framework-specific properties such as `policy.lerobot_policy` may remain for
compatibility, but `policy.native` is the common convention for integrations.

### Native-only usage

Physical AI does not need to own every operation. A user can continue using
the original framework directly and use explicit conversion helpers only at
Physical AI boundaries:

```python
from physicalai.integrations.lerobot import (
    from_lerobot_action,
    to_lerobot_batch,
)

native_batch = to_lerobot_batch(observation)
native_action = native.select_action(native_batch)
action = from_lerobot_action(native_action)
```

These helpers should be ordinary framework-specific functions. They do not
require a codec hierarchy or registration protocol.

## Policy Discovery

### Entry points map directly to classes

An integration package registers each named policy class in its package
metadata:

```toml
[project.entry-points."physicalai.policies.v1"]
"lerobot.act" = "physicalai_lerobot:ACT"
"lerobot.diffusion" = "physicalai_lerobot:Diffusion"
```

The entry-point name is the stable policy ID. The target is directly the
`Policy` subclass. There is no registration callback and no `PolicyPlugin`
container.

Installation makes the classes discoverable:

```bash
pip install physicalai-lerobot
physicalai policies list
```

Discovery uses `importlib.metadata.entry_points()`. Reading installed entry
point metadata does not import model code. A selected class is loaded only
when it is inspected or constructed.

Policy modules should therefore keep import-time behavior lightweight. Model
weights, datasets, and expensive framework initialization must not occur while
loading the class.

### Why one external registration remains necessary

Most metadata can live on the `Policy` class, but discovery cannot live only
inside the class. Decorators and `__init_subclass__` run only after Python has
already imported the module. Physical AI needs package metadata to learn that
an unimported class exists.

The entry point solves only that discovery problem:

```text
entry point: where is the class?
Policy class: what does it do and what is supported?
```

### No public registry object

`physicalai.policies` remains the existing Python module, not a singleton
`Policies` object.

The public API can remain function-based:

```python
from physicalai.policies import get_policy, get_policy_class, list_policies

policy_cls = get_policy_class("lerobot.act")
policy = get_policy("lerobot.act", config=native_config)
available = list_policies()
```

The implementation may cache discovered entry points in a private dictionary,
but that is an internal optimization rather than a plugin-author-facing API.

Duplicate IDs, entry points that do not load a `Policy` subclass, and failed
imports must produce deterministic errors.

## Capabilities and Catalog Metadata

### The Policy class is the source

Physical AI already follows this pattern for export. Policies expose
`get_supported_export_backends()`, and the backend inspects that class method
to report supported backends.

The same approach can provide the small amount of metadata needed for Studio:

```python
class ACT(LeRobotPolicy):
    display_name = "LeRobot ACT"
    implementation_source = "upstream"
    support_level = "deployment-supported"
    verified_workflows = frozenset({
        "xpu_train",
        "xpu_finetune",
        "openvino_deploy",
    })

    @classmethod
    def wrap(cls, native: ACTPolicy) -> "ACT":
        ...

    @staticmethod
    def get_supported_export_backends():
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]
```

The exact attribute names should be finalized with the implementation. The
important design decision is that the information belongs to the policy class,
not to a separate descriptor object that can drift from the implementation.

### Derived versus verified information

The catalog should distinguish information that can be derived mechanically
from information that represents a qualification claim.

Derived information includes:

- The entry-point policy ID.
- The Python class path.
- Whether the class is a Physical AI `Policy` and Lightning module.
- Whether it provides `wrap()`.
- Export backends returned by `get_supported_export_backends()`.
- Constructor or configuration type information that can be inspected.

Explicit, test-backed information includes:

- XPU training and fine-tuning qualification.
- Wrapper-versus-native numerical parity.
- Supported upstream package versions.
- OpenVINO conversion and Runtime deployment qualification.
- Quantization evaluation and adopted quantization modes.
- The support level promised by Studio.

Physical AI must not infer XPU or deployment support merely because a method
exists or a model can be constructed.

### Catalog metadata is not a model card

Catalog metadata describes a Physical AI integration with a model family. It
answers questions such as:

- Which implementation is installed?
- Can Studio construct it?
- Which workflows have been verified?
- Which export backends are supported?

A model card describes a specific trained checkpoint, its data, intended use,
metrics, and limitations. Model cards remain artifact-specific and may live on
Hugging Face or alongside a trained artifact. The policy catalog does not
replace them.

### Support levels

Support should be represented independently from implementation ownership:

| Support level        | Meaning                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| Available            | A generic adapter can attempt the policy; no model-specific support promise                               |
| Studio-supported     | Named integration with documented workflows, pinned compatibility, and required CI coverage               |
| Deployment-supported | Studio-supported plus verified OpenVINO export, Runtime loading, and deployment parity on target hardware |

Quantization is a separate evaluated capability rather than another support
level.

A first-party model is not automatically Deployment-supported, and an
upstream model may be Deployment-supported without being copied into Physical
AI.

## Studio Integration

Studio should consume the same discovered policy classes instead of
maintaining independent policy catalogs.

Today, backend policy APIs, model import, and construction contain separate
hard-coded policy lists. These should be replaced incrementally by functions
that resolve and inspect installed `Policy` classes:

```python
policy_cls = get_policy_class(request.policy_id)
policy_info = describe_policy(policy_cls)
```

The backend can return normalized JSON derived from the class:

```json
{
  "id": "lerobot.act",
  "display_name": "LeRobot ACT",
  "implementation_source": "upstream",
  "support_level": "deployment-supported",
  "verified_workflows": ["xpu_train", "xpu_finetune", "openvino_deploy"],
  "export_backends": ["torch", "openvino"]
}
```

`describe_policy()` is an internal inspection function, not another plugin
object. The UI renders what the backend reports and submits the stable policy
ID back to the backend.

Configuration schema generation should reuse the policy constructor,
configuration dataclass, and existing jsonargparse behavior where possible.
Plugin v1 does not need a second configuration model solely for Studio.

Only named integrations should be listed as curated Studio choices by
default. A generic framework adapter may be exposed through an advanced,
clearly best-effort path without representing every upstream model as
Studio-supported.

## Training and Original Framework APIs

### Lightning path

The existing Physical AI `Trainer` remains the training path for adapted and
first-party policies:

```python
native = ACTPolicy(native_config)
policy = ACT.wrap(native)

Trainer(max_epochs=100).fit(policy, data)
```

An adapter may need framework-specific code for batch conversion, loss
handling, optimizer and scheduler construction, preprocessing, action
selection, and native serialization. That code belongs inside the adapter
class or ordinary private helper functions. It does not require public facet
interfaces.

Input/output wrapping alone is enough for inference interoperability. A policy
must also preserve optimizer, scheduler, clipping, EMA, processor, and
checkpoint semantics before it claims native training parity.

### Native framework path

Users remain free to train or invoke the native policy with the upstream API:

```python
native = ACTPolicy(native_config)

for observation in physicalai_dataloader:
    native_batch = to_lerobot_batch(observation)
    loss, loss_info = native(native_batch)
    loss.backward()
    optimizer.step()
```

Physical AI does not need to wrap this loop merely to claim that the
integration exists.

### Separate environments

Separate-environment process control is not part of the policy plugin v1
contract.

When an upstream repository needs a conflicting environment, users can run
its original CLI in that environment and produce a native artifact. Physical
AI can then import, benchmark, or export the artifact through the named policy
integration.

```text
upstream environment -> native training command -> native artifact
                                              -> Physical AI import/export
```

If Studio later needs to launch those commands, it should build on its
existing job orchestration and the upstream CLI. That requirement should be
designed from a concrete integration rather than adding `TrainingExecutor`,
`JobHandle`, or another runner hierarchy to the policy API now.

This also avoids confusion with Runtime's existing `InferenceRunner`, such as
`SinglePass`, which describes inference-time execution behavior in an exported
model manifest and is unrelated to training orchestration.

## OpenVINO and Deployment Enablement

OpenVINO qualification should follow the same escalation strategy as training
enablement.

1. Attempt export from the unchanged upstream PyTorch model through the
   adapter and existing `ExportablePolicyMixin`.
2. Add the smallest policy-specific export preparation or graph workaround
   necessary.
3. Contribute reusable fixes to the upstream model, framework, PyTorch, or
   OpenVINO.
4. Carry a narrow, version-gated downstream fix while the upstream change is
   unavailable.
5. Build a first-party model only when the upstream implementation cannot meet
   deployment requirements within an acceptable maintenance cost.

The existing `get_supported_export_backends()` class method remains the API
for reporting implemented export paths:

```python
@staticmethod
def get_supported_export_backends():
    return [ExportBackend.TORCH, ExportBackend.OPENVINO]
```

Reporting OpenVINO as implemented is necessary but not sufficient for
Deployment-supported status. Qualification must include at least:

- Successful conversion from a supported checkpoint.
- Processor and feature-schema preservation.
- Numerical parity within documented tolerances.
- Runtime loading through `InferenceModel(...)`.
- Execution on the intended Intel deployment target.

Every OpenVINO-enabled policy should be evaluated for quantization. The
result may be "not adopted" when latency, memory, quality, hardware coverage,
or maintenance results do not justify it.

## Native and Exported Artifacts

An adapted policy should preserve native artifacts whenever possible:

```python
policy.native.save_pretrained("artifacts/native-act")
```

Physical AI export remains a separate deployment operation:

```python
policy.export("artifacts/openvino-act", backend="openvino")
```

The exported manifest should continue recording the policy source, model
artifacts, Runtime inference runner, preprocessors, postprocessors, and feature
schemas. For adapted policies it should additionally preserve enough
provenance to reproduce the integration:

- Stable policy ID and Python class path.
- Integration package version.
- Upstream framework version or commit.
- Upstream model/config identifier.
- Native artifact format and source checkpoint when applicable.
- Verified export and deployment target information.

This keeps the native checkpoint usable in its original framework while
making the exported artifact loadable by Physical AI Runtime.

## Downstream Patch Policy

Copying a complete upstream model into Physical AI should be exceptional.
Use this order:

1. Upstream configuration or existing PyTorch/XPU support.
2. Generic Physical AI adapter.
3. Narrow integration shim or subclass in the adapter package.
4. Versioned downstream patch or fork while an upstream contribution is
   pending.
5. First-party Physical AI implementation.

Every maintained downstream fix should have:

- A focused reason tied to a product requirement.
- A supported upstream version range.
- A regression or qualification test.
- An upstream issue or pull request where reusable.
- A documented condition for removing the downstream fix.

Avoid broad monkey-patching and copied model trees. Narrow, reviewable changes
are easier to validate, rebase, upstream, and remove.

## Qualification Requirements

### Generic adapter contract

The framework-level adapter should test its shared behavior without claiming
that every upstream policy is supported:

- Construction around an upstream config.
- `wrap(native)` preserves object identity.
- Physical AI observation and action conversion.
- Lightning device transfer.
- Native save/load preservation where supported.
- Clear errors for unsupported upstream policy behavior.

### Named integration contract

Each named Studio-supported integration should add model-specific coverage:

- Supported upstream dependency versions.
- Configuration and dataset-feature construction.
- CPU smoke coverage where practical.
- XPU construction, forward, training, and fine-tuning coverage.
- Native-versus-wrapper loss and inference parity.
- Gradient and optimizer-step parity when Lightning parity is claimed.
- Checkpoint save, resume, and native artifact round-trip.
- Documented known limitations.

### Deployment-supported contract

Deployment-supported integrations additionally require:

- OpenVINO conversion coverage.
- Exported-versus-native numerical parity.
- Processor and feature-schema round-trip.
- Runtime loading and inference.
- Target Intel hardware validation.
- A recorded quantization evaluation.

Capability metadata should reflect these tests rather than replace them.

## Proposed Migration

1. Preserve the existing `Policy`, `Trainer`, `ExportablePolicyMixin`, named
   policy classes, and YAML class paths.
2. Add a versioned `physicalai.policies.v1` entry-point group whose targets are
   directly `Policy` subclasses.
3. Extend `get_policy()` and class resolution to use installed entry points
   while preserving the existing built-in and `source="lerobot"` paths during
   migration.
4. Add the `wrap(native)` and `native` conventions to `LeRobotPolicy` while
   preserving its existing `lerobot_policy` property.
5. Keep generic LeRobot construction as the best-effort availability path and
   register curated named LeRobot aliases as discoverable integrations.
6. Put the minimum support and qualification metadata on named policy classes,
   reusing `get_supported_export_backends()` for export support.
7. Replace backend `_POLICY_CLASSES`, `_SUPPORTED_POLICIES`, importer lists,
   and UI constants with one class-derived catalog.
8. Classify existing policies as generic, named, or first-party and as
   Available, Studio-supported, or Deployment-supported based on current test
   evidence.
9. Validate the design with one additional upstream PyTorch integration, such
   as OpenPI's PyTorch implementation, before expanding the public metadata
   contract.
10. Continue upstreaming XPU, export, and framework fixes; add first-party
    implementations only when the documented escalation criteria are met.

## Final Recommendation

Physical AI should polish and generalize its existing composition-based policy
integration rather than build a new plugin framework.

The architecture is:

```text
Installed package metadata
    -> entry point maps stable ID directly to Policy class
        -> Policy class implements Physical AI / Lightning behavior
            -> optional native upstream policy retained as policy.native
                -> existing Trainer, benchmark, export, and Runtime workflows
```

The model-enablement ladder is:

```text
Upstream enablement
    -> generic adapter
        -> named, qualified integration
            -> OpenVINO and Runtime qualification
                -> first-party implementation only when necessary
```

This gives Physical AI a broad upstream model surface, a clear support
commitment for curated models, and a disciplined path to Intel XPU and
OpenVINO enablement without accumulating copied implementations or a second
policy architecture.

## See Also

- [Policy Design](library/docs/explanation/policy/README.md)
- [LeRobot Policy Integration](library/docs/explanation/policy/lerobot.md)
- [LeRobot Data Integration](library/docs/explanation/data/lerobot.md)
- [Python plugin discovery](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)
