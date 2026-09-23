# Policy integration: code-first design

**Status:** Agreed design direction; API sketches below are not implemented. See [research](policy-integration-research.md) for evidence and references, and [decisions](README.md#decisions) for ADRs.

## One picture

```text
Python / CLI / GUI
        │ same request
        ▼
      Engine                         physicalai-train
        ├─ resolve provider + python_env
        ├─ preflight on the machine running the job
        ├─ prepare/convert dataset
        ├─ launch, report, and cancel
        └─ validate result + provenance
                 │
                 ▼
      Provider plugin                lightweight integration
                 │ request
                 ▼
      Official framework worker      isolated python_env
                 │
                 ▼
      Official checkpoint ─────────► Studio model catalog
                 │
         ┌───────┴──────────┐
         ▼                  ▼
  Runtime export     Runtime adapter → upstream policy server
         └──────────┬───────┘
                    ▼
  InferenceModel → PolicySource → RobotRuntime → robot
```

A **plugin** is how an integration is packaged and discovered. An **adapter** is code that translates at a seam. A plugin may provide an upstream-native workflow or a Lightning adapter. These are different decisions, not “plugin versus adapter.”

## The Python interface

Existing direct Lightning training remains first-class:

```python
policy = ACT()
data = LeRobotDataModule(repo_id="lerobot/pusht")
Trainer(max_epochs=10).fit(policy, datamodule=data)
```

The proposed Engine adds managed workflows without replacing `Policy` or `Trainer`:

```python
# Studio/Lightning route: the caller supplies live Python objects.
result = Engine().fit(model=policy, datamodule=data)

# Upstream route: Config refers to the official upstream class.
config = Config(
    "gr00t.configs.finetune_config.FinetuneConfig",
    {
        "base_model_path": "nvidia/GR00T-N1.7-3B",
        "embodiment_tag": "NEW_EMBODIMENT",
        "max_steps": 10_000,
        "global_batch_size": 32,
    },
)
result = Engine().fit(
    config=config,
    dataset=studio_lerobot_dataset,
    python_env="gr00t-n17",
)
```

`Config(class_path, init_args)` is Runtime's existing reference to the official class. No Studio `GrootConfig` or copy of an upstream type is added. A dataset and output directory supplied by Engine bind to the official config fields; specifying the same field twice is an error.

Conceptually, Engine does this:

```python
def fit(request):
    provider = providers.for_official_config(request.config)
    target = resolve_target(request.target)
    env = target.python_env(request.python_env or provider.default_python_env)
    provider.preflight(env)                       # before transfer / GPU / weights
    dataset = provider.prepare_dataset(request.dataset)
    run = executor.launch(provider.invocation(request, dataset, env))
    result = provider.collect_result(run)          # native bundle + provenance
    model_catalog.register(result)
    return result
```

The Engine owns shared orchestration. It does **not** implement the framework's optimizer, model, or training loop.

## Provider: official API when usable, official CLI otherwise

A provider is a small adapter around one upstream project's configuration, dataset contract, invocation, and artifacts:

```python
class LeRobotProvider:
    def prepare_dataset(self, studio_dataset): ...
    def preflight(self, python_env): ...
    def invocation(self, official_config, dataset, python_env): ...
    def collect_result(self, completed_run): ...
```

Use a stable, supported Python function when upstream exposes one. Otherwise call the upstream's documented CLI from the selected interpreter using an argv list (`shell=False`). Both routes preserve upstream ownership. Logs are for display; structured status and artifacts determine success.

```python
# Conceptual native LeRobot invocation, in the LeRobot training python_env.
argv = [
    "<resolved-python-env>/bin/lerobot-train",
    f"--dataset.repo_id={dataset.repo_id}",
    "--policy.type=act",
    "--steps=10000",
    f"--output_dir={output_dir}",
]
run_process(argv, shell=False)
```

GR00T's documented entry point is `gr00t/experiment/launch_finetune.py`; use that CLI initially because it builds the internal training config before calling the training code. Contribute a reusable public training function upstream later. LeRobot 0.6.0 publishes `lerobot-train` in its [console scripts](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/pyproject.toml). XR1 has its own Hydra/`torchrun` launcher. Do not substitute Studio's Trainer merely because an upstream implementation internally uses Lightning.

## LeRobot: compare the two routes, don't assume either wins

### Existing Lightning wrapper

```python
from physicalai.policies.lerobot import ACT

wrapped_policy = ACT.from_dataset(repo_id)
result = Engine().fit(model=wrapped_policy, datamodule=studio_datamodule)
```

The wrapper delegates model/config/processors to LeRobot while adapting the policy to Studio's Lightning lifecycle. It also owns feature setup, optimizer configuration, checkpoint reconstruction, and action selection.

### Native LeRobot provider

```python
result = Engine().fit(
    config=official_lerobot_config,
    dataset=dataset_path,
    python_env="lerobot-training",
)
```

The provider stages the data, invokes official `lerobot-train`, and returns LeRobot's original checkpoint bundle. It does not translate LeRobot's training loop into Lightning.

Studio pins LeRobot 0.6.0 for dataset support, so a provider does not remove LeRobot from Studio's data environment. The potential benefit is isolating policy-training dependencies and preserving LeRobot's full training lifecycle.

After the release, compare both routes on ACT and SmolVLA using the same LeRobot revision, dataset, recipe/settings, and hardware:

```text
Lightning wrapper: loss/grad/weight parity + train/save/reload/infer/task evaluation
Native provider:   official lerobot-train + save/reload/infer/task evaluation
```

The existing [wrapper/native replay tests](../../../../library/tests/integration/test_lerobot_wrapper_equivalence.py) are useful evidence for selected update steps; they do not prove parity with the complete [`lerobot-train` workflow](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/scripts/lerobot_train.py). The current code declares ACT, Diffusion, and SmolVLA as equivalence-validated. Other wrappers remain experimental until their workflows have end-to-end evidence.

Choose a route per supported policy/capability based on behavior, checkpoint interoperability, Studio features retained, dependency/upstream-upgrade burden, and task results. Do not expose two routes for one policy as equally supported before that evidence exists.

## Result: preserve upstream checkpoints

```python
TrainingResult(
    status="completed",
    artifact=ArtifactRef(format="lerobot", path="runs/123/checkpoint/"),
    metrics={"train_loss": ..., "step": ...},  # upstream names
    provenance={
        "provider": "huggingface-lerobot",
        "upstream_revision": "...",
        "python_env": "lerobot-training",
        "dataset_conversion": "...",
    },
)
```

Studio records the official checkpoint unchanged, along with provenance and independently validated capabilities. It must not rename every artifact to `model.ckpt`. A successful training run does not automatically mean the model can be loaded by Runtime.

## From checkpoint to robot

### Preferred when validated: export to Runtime's contract

Studio/provider export creates backend model files and a `manifest.json`. Runtime owns loading, preprocessing/postprocessing, inference execution, action chunking, and robot control; Studio owns export and export validation.

```python
from physicalai.inference import InferenceModel
from physicalai.runtime import PolicySource, RobotRuntime, SyncExecution

model = InferenceModel("./models/123/exports/openvino")
runtime = RobotRuntime(
    fps=30,
    robot=robot,
    cameras=cameras,
    action_source=PolicySource(model=model, execution=SyncExecution()),
)
with runtime:
    runtime.run(duration_s=60)
```

Export and quantization are separate capabilities, not consequences of supporting training.

### If export is unavailable: Runtime adapter to upstream serving

A provider can offer a `RuntimeAdapter` that starts/connects to the official server in a compatible Python environment:

```python
# Conceptual adapter; must be registered and selected by a Runtime manifest.
class GrootServingAdapter(RuntimeAdapter):
    def load(self, artifact_path):
        self.client = launch_or_connect_to_groot_server(artifact_path)

    def predict(self, runtime_inputs):
        upstream_inputs = to_groot_observation(runtime_inputs)
        upstream_action = self.client.get_action(upstream_inputs)
        return {"action": to_runtime_action(upstream_action)}
```

`InferenceModel` and `PolicySource` remain the Runtime-facing contract. Validate observation/action names, order, units, normalization, action-chunk semantics, latency, timeouts, stale responses, and safe failure behavior. Exactly one layer owns normalization and action queues.

## Proof order

1. **GR00T + first-party:** convert a Studio LeRobot dataset; fine-tune GR00T N1.7 with the official workflow; reload, infer, and run one first-party policy through Engine.
2. **XR1:** add Xiaomi-Robotics-1 as a second upstream provider to validate that the design generalizes beyond GR00T.
3. **LeRobot A/B after release:** compare the Lightning wrapper and official `lerobot-train` routes on ACT and SmolVLA. Do not assume the existing wrapper tests prove full workflow parity.
4. **Physical deployment:** claim it only after testing on the named robot/scenario.

For evidence and pinned source references, see [the research appendix](policy-integration-research.md).
