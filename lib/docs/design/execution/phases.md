# Phases

Splitting app development into phases helps manage complexity, reduce risks, and improve planning. Each phase focuses on specific goals, allowing for incremental progress, easier troubleshooting, and better resource allocation.
After completing each phase, a round of analysis and planning should be done to keep the implemantaiton aligned with the global goals.

## Phase 0

Preparations to unblock implementation:

- Set up high-level project structure: organize installable package & tests ✅
- Develop an interface of a data module & wrap leRobot dataset ✅
- PoC of configuration system (CLI, config files): CLI should be isolated in a standalone file, no global config, as many as possible parameters are exposed via API. When required, a dataclass can be used as a configuration storage

## Phase 1

Initial training implementation:

- Model interface
- Training engine
- Models: Diffusion Policy (simple pusht), ACT

## Phase 2

- Reiterate on model configuration process
- Look at RL datasets
