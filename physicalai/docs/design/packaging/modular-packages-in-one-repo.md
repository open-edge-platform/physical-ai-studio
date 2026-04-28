# Modular Packages in One Repo

Design note for shipping reusable libraries such as `inferencekit` and `capturekit`
from the `physical-ai` repo without forcing an immediate repo split.

---

## Problem

Some parts of the `physicalai` runtime are more general than physical AI itself.
For example,

- `physicalai.inference` is a generic inference engine with backends, manifests, and runners
- `physicalai.capture` is a generic camera interface with hardware backends

These modules may eventually be useful to other domains and other repositories. At the
same time, creating separate repositories too early adds release and maintenance overhead.

We want a structure that lets us:

- keep development in one repository for now
- publish reusable pieces as standalone packages
- avoid duplicate implementations
- make future extraction cheap if a separate repo becomes justified

---

## Approach

This document proposes to use a **multi-package monorepo** approach.

We could keep `inferencekit`, `capturekit`, and `physicalai` in the same repository, but publish
them as separate Python distributions.

```text
physical-ai/
├── packages/
│   ├── inference/
│   │   ├── pyproject.toml
│   │   └── src/inferencekit/
│   ├── capture/
│   │   ├── pyproject.toml
│   │   └── src/capturekit/
│   └── physicalai/
│       ├── pyproject.toml
│       └── src/physicalai/
└── pyproject.toml
```

This gives us one repo, one CI surface, and one development workflow, while still allowing:

- `pip install inferencekit`
- `pip install capturekit`
- `pip install physicalai`

The important point is that each distribution owns its own import path.

---

## Motivation

This could be a better fit than either of the two common alternatives.

### 1. Better than premature repo splits

Creating a new repository for every reusable module sounds clean, but it creates real cost:

- separate CI and release pipelines
- version coordination across repos
- more difficult local development
- more coordination overhead for small API changes

If there is not yet a concrete external consumer, that overhead is usually wasted.

### 2. Better than one giant package with extras

Putting everything under `physicalai` and depending on lazy imports alone keeps the code in
one package, but the boundaries stay soft. Over time, it becomes easy for generic modules to
accidentally import runtime-specific or training-specific code.

For example, we have seen that we could easily inject physical-ai related context
into this generic `inferencekit` interfaces.

Separate distributions create a harder boundary:

- `inferencekit` cannot import `physicalai`
- `capturekit` cannot import `physicalai`
- `physicalai` depends on them, not the reverse

That dependency direction is the main design goal.

---

## Package Roles

### `inferencekit`

Owns the domain-agnostic inference layer:

- `InferenceModel`
- backend adapters and adapter registry
- manifest loading
- runners, preprocessors, postprocessors
- plugin and optional-backend integration points

It must not import robotics, cameras, policies, or training code.

### `capturekit`

Owns the generic camera layer:

- `Camera` interface / ABC
- `Frame`
- device discovery
- multi-camera reads
- hardware-specific camera backends

It must not import inference or robot code.

### `physicalai`

Owns the physical-AI runtime:

- runtime orchestration
- robot integration
- observation assembly
- safety and lifecycle management
- CLI and deployment workflows

It depends on `inferencekit` and `capturekit` and may re-export selected APIs for user
convenience.

---

## Import Strategy

Ideally we should **not** split one import path across multiple distributions if we can avoid it.

For example:

```text
packages/inferencekit/src/physicalai/inference/
packages/physicalai/src/physicalai/inference/
```

Even if namespace packaging makes this technically possible, it would increase complexity:

- more fragile import resolution
- harder IDE and type-checker behavior
- less obvious ownership of modules
- more room for accidental shadowing and circular design

Instead, we could have a clear top-level package names such as:

```text
packages/inferencekit/src/inferencekit/
packages/capturekit/src/capturekit/
packages/physicalai/src/physicalai/
```

Then `physicalai` can provide thin re-exports:

```python
# physicalai/inference/__init__.py
from inferencekit import InferenceModel

__all__ = ["InferenceModel"]
```

```python
# physicalai/capture/__init__.py
from capturekit import Camera, Frame, UVCCamera

__all__ = ["Camera", "Frame", "UVCCamera"]
```

This keeps the implementation single-sourced while preserving ergonomic imports for
runtime users.

---

## Optional Dependencies and Plugins

Reusable packages should keep their base installs small.

### Inference backends

`inferencekit` could expose optional extras and lazy backend registration for heavy
runtime dependencies:

```toml
[project.optional-dependencies]
onnx = ["onnxruntime"]
openvino = ["openvino"]
tensorrt = ["tensorrt"]
```

Backends should register lazily so file-based backend detection does not import every
runtime dependency at import time.

### Training-specific loaders

Training-specific compatibility loaders should stay outside `inferencekit`.

For example, a backend that reconstructs a policy from a Lightning checkpoint is not a
generic inference backend. It belongs in `physicalai-train` as a plugin, even if it uses
the same backend registry interface.

Rule of thumb:

- deployment artifact loader -> belongs in `inferencekit`
- training artifact compatibility loader -> belongs in `physicalai-train`

The same rule applies to `capturekit`: camera SDKs should be optional extras, and imports
should stay lazy at backend boundaries.

---

## Naming Guidance

Despite not being final, we could have package names that are clear and likely to be available:

- `inferencekit`
- `capturekit`
- `physicalai`

These are boring, in a good way, but enterprise friendly names that we could eaisily
get clearance. We would, of course, welcome better namings if you might have any.

---

## Migration Path

We should not extract everything at once. The safer path is incremental.

### Phase 1

Keep the code in `physical-ai`, but organize it as if extraction were already done:

- no cross-imports from generic packages into `physicalai`
- package-local tests
- package-local `pyproject.toml`
- package-local README and API ownership

### Phase 2

Publish `inferencekit` and/or `capturekit` from the same repo.

`physicalai` switches from local module ownership to dependency + re-export.

### Phase 3 (Most Probably Not Needed)

Only if external demand justifies it, move one package to its own repository. If the
package already has:

- its own import path
- its own distribution metadata
- its own tests
- no forbidden imports

then extraction is mostly a repository move, not a redesign.

This is, however, most likely not needed. We could justify that camera, robot,
inference is part of physicalai that they can be a standalone pip package, but
can still be part of a physicalai repo.

---

## Best Practices

- Keep reusable packages domain-agnostic in both naming and dependencies
- Enforce one-way dependency flow in CI
- Use separate distributions, not duplicated code
- Re-export from `physicalai` for convenience, but keep implementation in the generic package
- Treat lazy imports as a performance tool, not as the primary architecture boundary
- Extract to a new repository only when there is a real consumer or release need

---

## Summary

We will treat reusable modules such as inference and capture as **independent packages
hosted in the same repository** until a concrete need for separate repositories emerges.

This keeps the architecture clean today without paying the organizational cost of an early
repo split.
