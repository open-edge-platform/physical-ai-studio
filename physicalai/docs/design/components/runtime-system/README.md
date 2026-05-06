# Runtime System Design

This directory contains the design docs for PhysicalAI's policy runtime system: the runtime loop that connects robot IO, policy inference, action execution, and remote serving.

## Start Here

Choose the reading path based on why you are here.

### For design review or pre-read

1. Read [design_review_summary.md](./design_review_summary.md).
2. Read or present [design_review_deck.md](./design_review_deck.md).
3. Read [policy_runtime_design.md](./policy_runtime_design.md) for the full design.
4. Read [policy_server_design.md](./policy_server_design.md) if you care about remote inference or `physicalai serve`.
5. Use [inference_comparison_report.md](./inference_comparison_report.md) only if you want the gap analysis behind the design.

This is the best path for most readers. It starts with the shortest material, then the presentation, then the full specification.

### For implementation or detailed review

1. Read [policy_runtime_design.md](./policy_runtime_design.md).
2. Read [policy_server_design.md](./policy_server_design.md).
3. Use [design_review_summary.md](./design_review_summary.md) and [design_review_deck.md](./design_review_deck.md) as condensed versions for communication.

This path is for engineers who need the APIs, boundaries, phases, and acceptance criteria.

## What Each Document Is For

### [design_review_summary.md](./design_review_summary.md)
One-page summary for stakeholders and reviewers.

Use it when you want:
- the problem statement in one page
- the proposed shape of the solution
- the phase breakdown and major decisions

### [design_review_deck.md](./design_review_deck.md)
Presentation deck for the team review.

Use it when you want:
- the review narrative
- the main diagrams and framing
- backup slides for CLI, remote execution, and design rationale

Render with:

```bash
npx @marp-team/marp-cli design_review_deck.md --pdf
```

### [policy_runtime_design.md](./policy_runtime_design.md)
Primary design specification for the runtime itself.

It defines:
- scope, principles, and ownership boundaries
- `InferenceModel` additions
- `Execution`, `PolicyRuntime`, `ActionQueue`, `Callback`, and validation
- CLI design for `physicalai run`
- implementation phases, acceptance criteria, and deferred components

This is the source of truth for the runtime design.

### [policy_server_design.md](./policy_server_design.md)
Detailed design for remote inference and `PolicyServer`.

It defines:
- why remote execution exists
- server and client responsibilities
- gRPC streaming behavior
- warmup, shutdown, handshake, and failure handling

Read this after the main runtime design. It extends `RemoteExecution`; it does not replace the main design.

### [inference_comparison_report.md](./inference_comparison_report.md)
Reference report comparing existing systems to the proposed runtime design.

Use it when you want:
- the original gap analysis
- traceability from the old systems to the new design
- context for why the runtime is structured this way

## What Is Decided Here

These docs fix the following design direction:

1. `PolicyRuntime` is the single runtime loop.
2. `Execution` owns when and where inference runs.
3. `AsyncExecution` and `RemoteExecution` are sibling execution models.
4. Action smoothing stays split across denoising, runner-time, and queue-time layers.
5. `physicalai run` and `physicalai serve` live in the runtime distribution, not in the training CLI.
6. Product workflows such as sentry, HIL, highlight, and DAgger are composed by consumers around the runtime rather than baked into it.

## Current Status

- Phase 4 is the current focus: hardening and API stabilization.
- Phase 5 adds remote execution and `PolicyServer`.
- Phase 6 covers intentionally deferred components.

See [policy_runtime_design.md](./policy_runtime_design.md) §15 for the full phased plan and acceptance criteria.

## Directory Contents

```text
runtime-system/
├── README.md
├── design_review_summary.md
├── design_review_deck.md
├── policy_runtime_design.md
├── policy_server_design.md
└── inference_comparison_report.md
```

## Related Docs

- Broader stack vision: `../../architecture/robot_stack_vision.md`
- Top-level design index: `../../README.md`

## Questions

See [policy_runtime_design.md](./policy_runtime_design.md#20-open-questions).
