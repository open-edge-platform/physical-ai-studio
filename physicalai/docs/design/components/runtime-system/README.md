# Runtime System Design

This directory contains the complete design specification for PhysicalAI's policy runtime system.

## Overview

The policy runtime is the core execution engine that runs trained policies on robotic hardware. It handles:
- **Inference**: Executing policies with support for multiple backends (ExecuTorch, OpenVINO, ONNX, LLMs).
- **Action execution**: Chunking, smoothing, and queueing actions for hardware control.
- **Remote execution**: Running policies on edge servers via gRPC with bidirectional streaming.
- **Observability**: Logging, validation, and metrics collection.

## Documents

### [policy_runtime_design.md](./policy_runtime_design.md)
The primary design specification. Covers:
- §1–5: Motivation, glossary, scope, problem statement
- §6–8: Core abstractions (`PolicyRuntime`, `Robot`, `Execution`)
- §9–13: Action management (`ActionChunking`, `ActionQueue`, smoothing, merging)
- §14: CLI integration (`physicalai run` subcommand via jsonargparse entry points)
- §15–19: Implementation phases, consumer mapping, strategy composition
- §20–22: Open questions, file layout, references

**Read this first.** It defines the architecture, APIs, and design rationale.

### [policy_server_design.md](./policy_server_design.md)
Remote execution design (Phase 5). Specifies:
- §1–3: Motivation, why remote execution is needed, deployment scenarios
- §4–7: `PolicyServer` component, request/response types, gRPC bidirectional streaming
- §8–10: CLI integration, warmup/shutdown, multi-tenancy and auth considerations
- §11–13: Observability, client libraries, acceptance criteria

**Read after the main design.** Extends `RemoteExecution` from the runtime design with server-side details.

### [design_review_summary.md](./design_review_summary.md)
One-page executive summary for stakeholders. Maps:
- PhysicalAI subsystems → runtime design sections
- Problem space (inference, action handling, remote deployment) → solution shape
- Implementation phases with concrete deliverables

**Use this for pre-read before design reviews.**

### [design_review_deck.md](./design_review_deck.md)
Marp-formatted presentation deck (13 main + 8 backup slides). Covers:
- Motivation, problem statement, solution sketch
- Core abstractions and their roles
- Action handling flow and strategy composition
- CLI, phases, and key decisions

**Render with:**
```bash
npx @marp-team/marp-cli design_review_deck.md --pdf
# or use VS Code Marp extension
```

### [inference_comparison_report.md](./inference_comparison_report.md)
Gap analysis: existing inference systems (LeRobot, InferenceKit, Viper) vs. runtime design requirements.

**Reference only.** Shows the baseline and what the new design adds.

## Key Design Decisions

1. **CLI in runtime distribution**: `physicalai run` and `physicalai serve` live in the `physicalai/` distribution (no Torch/Lightning), not in the training distribution. Training subcommands plug in via entry points.

2. **Entry-point based plugin architecture**: One `physicalai` console script; subcommands discovered via `[project.entry-points."physicalai.cli.subcommands"]`.

3. **Sibling execution models**: `AsyncExecution` and `RemoteExecution` both inherit from `Execution`; chosen at runtime by config, not compile-time.

4. **Action smoothing at multiple levels**: `Guidance` (denoising), `TemporalEnsemble` (runner), `ActionQueue.smoother` (queue-time). All coexist and compose.

5. **Strategy composition in consumers**: Strategies (sentry, highlight, HIL, DAgger) are implemented as thin consumer code on `PolicyRuntime` + `Callback`, not baked into the runtime.

## File Layout

```
physicalai/
  src/physicalai/
    runtime/
      __init__.py
      policy_runtime.py       # PolicyRuntime, Robot, Execution
      async_execution.py      # AsyncExecution (local, process, async)
      remote_execution.py     # RemoteExecution + client
      action_queue.py         # ActionQueue, LerpChunkSmoother, etc.
      validators.py           # RuntimeValidator, callbacks
    execution/
      remote/
        server.py             # PolicyServer, gRPC service (Phase 5)
    cli/
      main.py                 # jsonargparse dispatcher
      run.py                  # 'physicalai run' subcommand
      serve.py                # 'physicalai serve' subcommand (Phase 5)
  docs/design/components/runtime-system/  # (this directory)
```

## Next Steps

- **Phase 4 (now)**: Hardening and API stabilization.
- **Phase 5**: Remote execution (gRPC, PolicyServer, warmup/shutdown).
- **Phase 6**: Deferred components (advanced smoothing, custom runners, backends).

## Questions?

See **[Open Questions](./policy_runtime_design.md#20-open-questions)** in the main design document.
