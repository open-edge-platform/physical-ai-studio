# Physical AI LeRobot Policy Plugin

This plugin connects LeRobot's native trainer and policy format to Physical AI. It provides a `LeRobotEngine` under the `physicalai.engines` entry-point group and a Runtime adapter under `physicalai.inference.adapters`. The adapter owns LeRobot preprocessing, stateful action selection, postprocessing, and episode reset; robot installations do not need `physicalai-train`.

## Install

For library development, install the host torch variant that matches the machine and the LeRobot engine extra from `library/`:

```bash
uv sync --extra cu132 --extra lerobot
```

Use `--extra cpu` or `--extra xpu` instead when appropriate. This host extra controls host-side PyTorch and first-party training; it does not select LeRobot's managed worker. The worker has its own locked CPU, CUDA 13.2 (`cu132`), and XPU torch variants and follows LeRobot's native device selection.

## Native CLI

`pai train lerobot` passes native arguments to LeRobot's `lerobot-train` command. The worker's standard output, standard error, and exit code are passed through unchanged:

```bash
pai train lerobot --policy.type=act --policy.device=cuda --dataset.repo_id=lerobot/pusht --steps=20000
pai train lerobot --policy.type=act --policy.device=xpu --dataset.repo_id=lerobot/pusht
pai train --interpreter /opt/lerobot/bin/python lerobot --help
```

LeRobot's native `--policy.device` selects the policy device and matching worker torch extra (`cuda` → `cu132`, `xpu` → `xpu`, `cpu` → `cpu`). If the native config does not specify a device, the engine detects available XPU, then CUDA, then CPU without relying on the host torch build. Native arguments pass through unchanged. `--interpreter` skips managed provisioning; the supplied Python must already have the torch build required by the native device.

Prepare a managed environment ahead of time (or while building an image), inspect its lock and size, and remove stale environments with:

```bash
pai engine prepare lerobot --accelerator xpu
pai engine list
pai engine prune
```

Managed environments are prepared by Runtime's `physicalai.isolate` `UvProvider` with the `uv` installed alongside Studio (the `uv` package), falling back to `uv` on `PATH`. `pai engine prepare --accelerator` selects which worker profile to pre-provision; a later training call still follows its native device setting or automatic hardware detection. Python config objects passed to `LeRobotEngine.train()` must match the worker's exact LeRobot version; native CLI flags and config files only need to satisfy the plugin's supported version range.

## Python API

```python
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from physicalai.engine import Engine
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

config = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id="lerobot/pusht"),
    policy=ACTConfig(push_to_hub=False, device="cuda"),
    steps=20_000,
)

engine = LeRobotEngine()
run = engine.train(config)  # policy.device selects the matching worker torch extra
package = engine.export(run, "exports/act")
model = Engine.load(package)
```

`Engine.get("lerobot")` returns the class, and `Engine.create("lerobot")` creates an engine instance. `LeRobotEngine.run()` accepts native settings and overrides `dataset.repo_id`, `dataset.root`, `output_dir`, and `policy.device` with its standard inputs. Its `device` argument selects both the LeRobot device and matching worker torch extra; use `device="auto"` for hardware-aware detection.

LeRobot's worker project and lock ship as plugin package data; policy families with their own LeRobot extras (`smolvla`, `pi`, `molmoact2`) are extras of that one project. Runtime isolate synchronizes the project into a cache keyed by its source and Python/extras profile. Engine checks the worker's version handshake (the worker starts training only after the host accepts it), forwards JSON-lines progress, and propagates cancellation; tagged protocol events share stderr while native stdout/stderr are forwarded to the host logger. The managed cancellation listener currently requires POSIX pipe semantics; native CLI passthrough remains platform-native.

## Runtime Packages

Exporting a native LeRobot checkpoint creates a Runtime package containing `manifest.json` and the unchanged `pretrained_model/` files under a `.lerobot` artifact directory. The plugin Runtime adapter loads LeRobot's saved pre/postprocessor pipelines, calls `select_action()`, and forwards resets to the policy and processors. It does not delegate LeRobot preprocessing or postprocessing to Studio or Runtime components. Its scope is policies whose inference fits this step-by-step loop and that are on the supported list below; policies that need a different inference loop are out of scope until checked.

The plugin supports only LeRobot policy types whose training, packaging and Runtime inference have been checked: ACT, Diffusion, SmolVLA and Pi0.5 (`policy_support.py`). Training, `cli()`, export and the Runtime adapter reject other types before LeRobot builds a model. Training and managed `cli()` runs check the arguments first, and the worker checks again once LeRobot has resolved them, for example a `policy.path` set in a config file. Reward-model training is not supported: LeRobot trains a configured `reward_model` instead of the policy. Several LeRobot 0.7 policies (`fineart_vla`, `gaussian_actor`, `groot`, `pi0_fast`, `wall_x`) and reward models (`reward_classifier`, `robometer`, `topreward`) load remote code without a pinned revision. Before Runtime loads a package, and before export, the adapter checks `policy_preprocessor.json` and `policy_postprocessor.json`: each step must be one of the processor steps the supported policies use, and each state file a `.safetensors` file in the checkpoint. That guard stays until the remote-code path is reviewed and pinned. Optimized export enablements and downstream patch materialization are not included; upstream support comes first, and patch application remains SDL-gated.

## Development Checks

From the repository root, run the focused contract tests with the library dependencies installed:

```bash
uv run --project library pytest library/tests/unit/engine
uv run --project plugins/policies/lerobot --extra train --extra tests pytest plugins/policies/lerobot/tests
```

Optional public-dataset training checks are gated because they download data and start the isolated worker:

```bash
PHYSICALAI_RUN_LEROBOT_INTEGRATION=1 \
PHYSICALAI_LEROBOT_DEVICE=auto \
PHYSICALAI_LEROBOT_TEST_STEPS=5 \
HF_LEROBOT_HOME=/path/to/lerobot-cache \
uv run --project library --extra cu132 --extra tests --extra lerobot \
    pytest library/tests/integration/test_engine_lerobot_hf_training.py::test_native_lerobot_training_matches_engine_on_pinned_pusht -s
```
