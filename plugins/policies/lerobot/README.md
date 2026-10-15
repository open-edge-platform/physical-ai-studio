# Physical AI LeRobot Policy Plugin

This plugin connects LeRobot's native trainer and policy format to Physical AI. It provides a `LeRobotEngine` under the `physicalai.engines` entry-point group and a Runtime adapter under `physicalai.inference.adapters`. The adapter owns LeRobot preprocessing, stateful action selection, postprocessing, and episode reset; robot installations do not need `physicalai-train`.

## Install

For library development, install the host torch variant that matches the machine and the LeRobot engine extra from `library/`:

```bash
uv sync --extra cu128 --extra lerobot
```

Use `--extra cpu` or `--extra xpu` instead when appropriate. This host extra controls host-side PyTorch and first-party training; it does not select LeRobot's managed worker. The worker has its own locked CPU, CUDA (`cu128`), and XPU torch variants and follows LeRobot's native device selection.

## Native CLI

`pai train lerobot` passes native arguments to LeRobot's `lerobot-train` command. The worker's standard output, standard error, and exit code are passed through unchanged:

```bash
pai train lerobot --policy.type=act --policy.device=cuda --dataset.repo_id=lerobot/pusht --steps=20000
pai train lerobot --policy.type=act --policy.device=xpu --dataset.repo_id=lerobot/pusht
pai train --interpreter /opt/lerobot/bin/python lerobot --help
```

LeRobot's native `--policy.device` selects the policy device and matching worker torch extra (`cuda` → `cu128`, `xpu` → `xpu`, `cpu` → `cpu`). If the native config does not specify a device, the engine detects available XPU, then CUDA, then CPU without relying on the host torch build. Native arguments pass through unchanged. `--interpreter` skips managed provisioning; the supplied Python must already have the torch build required by the native device.

Prepare a managed environment ahead of time (or while building an image), inspect its lock and size, and remove stale environments with:

```bash
pai engine prepare lerobot --accelerator xpu
pai engine list
pai engine prune
```

The default interpreter is managed by `physicalai-train` and provisioned with the pinned `uv` package. `pai engine prepare --accelerator` selects only which worker environment to pre-provision; a later training call still follows its native device setting or automatic hardware detection. Python config objects passed to `LeRobotEngine.train()` must match the worker's exact LeRobot version; native CLI flags and config files only need to satisfy the plugin's supported version range.

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

LeRobot's worker project and lock ship as plugin package data. The core runner synchronizes that project into a cache keyed by framework, accelerator, and lock hash, checks the worker's version handshake, forwards JSON-lines progress, and propagates cancellation without redirecting native CLI output. The managed progress/cancellation runner currently requires POSIX inherited file descriptors; native CLI passthrough remains platform-native.

## Runtime Packages

Exporting a native LeRobot checkpoint creates a Runtime package containing `manifest.json` and the unchanged `pretrained_model/` files under a `.lerobot` artifact directory. The plugin Runtime adapter loads LeRobot's saved pre/postprocessor pipelines, calls `select_action()`, and forwards resets to the policy and processors. It does not delegate LeRobot preprocessing or postprocessing to Studio or Runtime components.

The Engine omits Groot from its policy list, and the Runtime adapter rejects Groot artifacts before loading a policy class because LeRobot enables `trust_remote_code=True` for Groot. That guard stays until the remote-code path is reviewed and pinned. Optimized export enablements and downstream patch materialization are not included; upstream support comes first, and patch application remains SDL-gated.

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
uv run --project library --extra cu128 --extra tests --extra lerobot \
    pytest library/tests/integration/test_engine_lerobot_hf_training.py::test_native_lerobot_training_matches_engine_on_pinned_pusht -s
```
