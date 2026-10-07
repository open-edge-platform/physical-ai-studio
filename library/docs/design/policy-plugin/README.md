# Policy Integration Design

This directory records Physical AI Studio's agreed design for integrating first-party and upstream robot-learning policies. It supersedes the earlier `Policy`-subclass-only plugin proposal.

## Design

- [User experience and architecture](policy-integration-experience.md)
- [Research and evidence](policy-integration-research.md)

## Decisions

- [0001 — Library-first orchestration without replacing Lightning](0001-library-first-policy-orchestration.md)
- [0002 — Upstream-first integrations and external contributors](0002-upstream-first-policy-integration.md)
- [0003 — Upstream APIs and dataset contracts as the source of truth](0003-upstream-apis-as-source-of-truth.md)
- [0004 — Official checkpoints and Runtime inference adapters](0004-official-checkpoints-and-runtime-inference.md)

The repository-wide glossary remains in [`CONTEXT.md`](../../../../CONTEXT.md). These documents record design decisions only; implementation and execution are deferred to the first proof of concept.
