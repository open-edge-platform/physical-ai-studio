---
status: accepted
---

# Add library-first policy orchestration without replacing Lightning

Independent policy repositories have different training lifecycles and incompatible dependency stacks. Add orchestration in the Studio library, consumed by Python, CLI, and GUI, while preserving `Policy`, `Trainer`, and the core Lightning pipeline as essential, directly usable interfaces. Permit separate Python environments and validate capabilities independently rather than forcing all policies into one training implementation or promising uniform feature parity.

## Consequences

- The GUI consumes library workflow behavior; it does not implement a separate model-training path. Application scheduling, authorization, and persistence retain their existing ownership.
- Upstream-supported hardware can establish the initial baseline. Intel training, inference, export, and quantization remain explicit, independently validated capability targets rather than universal admission requirements.
- Contribute reusable capability and hardware changes upstream to pursue off-the-shelf and day-zero support. Upstream availability does not replace validation of Studio's integration.
- Preserve existing direct Lightning usage. An upstream execution handle must not silently replace an object returned by the existing `Policy`/`Trainer` interfaces.
- Provide a common orchestration facade with a proper programmatic Python interface accepting model, dataset, and configuration inputs. YAML is an additional serialization/input route, not a mandatory intermediate or the only public interface. [ADR 0003](0003-upstream-apis-as-source-of-truth.md) defines how official upstream objects are passed.
- Allow first-party workflows to use isolated execution without changing direct in-process Lightning behavior. Reuse compatible prepared environments across policies; do not require one environment per model or manual shell activation between jobs. Select an explicitly requested environment first; otherwise reuse the configured compatible default. Ambiguity is an error, and an active environment is never modified automatically. Name the public selector `python_env`: Runtime already uses `execution` for inference execution strategies, and Studio uses "environment" for robot-and-camera setups and simulations.
- Provide explicit curated setup through existing tools and an attach-existing-environment option. Training must not silently install or upgrade packages.
- Set up and select a Python environment on the machine that runs the job, including remote and SSH training targets, and record both with the job. Before copying data, claiming accelerators, or downloading checkpoints, run a fast check inside that environment: plugin imports, key package versions, upstream revision and downstream patches, and declared system dependencies. Setup runs the full check once; each job repeats the fast check. Failures explain how to fix the problem, and nothing is installed or modified automatically.
- This decision accepts the architectural direction, not a final provider protocol or the existing Engine proposal wholesale. [ADR 0002](0002-upstream-first-policy-integration.md), [ADR 0003](0003-upstream-apis-as-source-of-truth.md), and [ADR 0004](0004-official-checkpoints-and-runtime-inference.md) record later decisions. Detailed package layout, export and quantization, and a lightweight orchestration package for upstream environments are designed during the first proof.

See the [research and decision tree](policy-integration-research.md) for evidence and deferred design work.
