# Physical AI GR00T Policy Plugin

This plugin connects NVIDIA [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.7 to the Physical AI engine and Runtime:

- **Train** with GR00T's own `launch_finetune.py` through `Engine.create("gr00t")` or `physicalai train gr00t`.
- **Infer** through Runtime's `InferenceModel` with a native PyTorch package.
- **Export** an OpenVINO package by applying the [Robotics AI Suite `gr00t-n1d7-ov`](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines/gr00t-n1d7-ov) patch series to the same upstream commit.

## How it is built

The plugin declares each project source and `physicalai.isolate` sync profile in `environments.py`. Runtime can prepare these environments with only `physicalai` and the plugin installed; it does not import `physicalai.engine` or require `physicalai-train`. The optional `engine.py` adapts the same declarations to Engine `WorkerEnv` and `Enablement` metadata for training and export:

| Environment                  | Source (`environments.py`)                                                             | Used for                                |
| ---------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------- |
| `default`                    | `GitProject`: Isaac-GR00T at the pinned commit, its own `uv.lock`                      | Training and PyTorch inference (CUDA)   |
| `robotics-ai-suite-openvino` | Same commit plus the Suite's 8 OpenVINO patches, declared by the OpenVINO `Enablement` | OpenVINO export and inference (CPU/GPU) |

GR00T pins `transformers==4.57.3`, `numpy==1.26.4`, `torch==2.7.1`, and Python `<3.13`; Physical AI Runtime requires `transformers>=5.15`. GR00T is therefore never imported in the host: training and export run through Engine's `run_worker`, and Runtime inference runs in a worker of the package's environment.

To materialize the `robotics-ai-suite-openvino` environment, Runtime isolate fetches the pinned upstream commit and checks `HEAD`. It then reads each patch, verifies its SHA-256, applies the series with `git am`, checks the resulting Git tree, and runs `uv sync --frozen` on the checkout's lock.

Runtime isolate caches environments under `~/.cache/physicalai/isolate/<key>` (override with `PHYSICALAI_ISOLATE_CACHE`). The key hashes the source and sync profile, including the Python version; a changed commit, reviewed patched tree, or profile builds a new environment. Engine deployments can still set `PHYSICALAI_ENGINE_CACHE` as an Engine-only cache override.

## Where the patches live

The Robotics AI Suite team owns the OpenVINO patches today, so `environments.py` declares them as `PublishedPatches` pinned to the Suite revision. Runtime isolate downloads them, or reads a local mirror in `src/physicalai_gr00t_policy_plugin/patches/robotics-ai-suite-gr00t-n1d7-ov/`, and verifies the per-file digests.

If the patch maintainers contribute here instead, the patches live in the plugin, and only the declaration changes:

```python
OPENVINO_PATCHES = BundledPatches(_PACKAGE / "patches" / "openvino", tree="d657bcb...", license="Apache-2.0")
OPENVINO = WorkerEnv(..., project=GitProject(UPSTREAM.repository, UPSTREAM.revision, patches=OPENVINO_PATCHES))
```

Their workflow stays Git-native:

1. Develop on the pinned Isaac-GR00T commit, then run `git format-patch <commit>..HEAD -o src/physicalai_gr00t_policy_plugin/patches/openvino/`.
2. Run `physicalai engine prepare gr00t --enablement robotics-ai-suite-openvino`. With a stale tree pin, the error reports the new tree; pin it after review.
3. Run the end-to-end test below and open a pull request reviewed by the patch owners.

Bundled digests are computed from the files, so adding or editing a patch builds a new environment without editing any checksum; the tree pin is the reviewed result. Moving the Suite's eight patches unchanged into the plugin reproduces tree `d657bcb` and keeps the environment key, so existing environments and packages stay valid. Layers allow a gradual move:

```python
patches=(
    PublishedPatches(..., files=suite_files[:4], tree=intermediate_tree),
    BundledPatches(local, files=own_files, tree=final_tree),
)
```

## Workflow

```python
from physicalai.engine import Engine
from physicalai_gr00t_policy_plugin.config import Gr00tTrainConfig

engine = Engine.create("gr00t")
run = engine.train(
    Gr00tTrainConfig(
        dataset_path="/data/libero_demo",  # LeRobot v2.1 with meta/modality.json
        embodiment_tag="LIBERO_PANDA",
        output_dir="runs/gr00t-libero",
        max_steps=2000,
        global_batch_size=32,
    ),
)

native = engine.export(run, "exports/gr00t-native")  # PyTorch package
openvino = engine.export(run, "exports/gr00t-openvino", backend="openvino")  # Suite patches

with Engine.load(openvino) as model:
    actions = model.predict_action_chunk(
        {
            "images": {"image": front_rgb, "wrist_image": wrist_rgb},  # HWC uint8
            "state": state_vector,  # flat, per meta/modality.json
            "task": "put the white mug on the plate",
        },
    )
```

The native CLI passes flags unchanged to `launch_finetune.py`:

```bash
physicalai train gr00t --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /data/libero_demo --embodiment-tag LIBERO_PANDA --output-dir runs/gr00t-libero
physicalai engine prepare gr00t                                          # optional pre-provisioning
physicalai engine prepare gr00t --enablement robotics-ai-suite-openvino
```

`Gr00tTrainConfig` is a plugin-owned host-side convenience type; it is **not** NVIDIA's `FinetuneConfig`. The official class is `gr00t.configs.finetune_config.FinetuneConfig`, which upstream's launcher constructs inside the isolated GR00T worker. We do not import GR00T into the host because its Python/Torch/Transformers dependencies conflict with Runtime. For the exact official interface, use the native CLI above; its flags are forwarded unchanged. The Python settings type exposes common fields for Engine/Studio and `extra_args` for other native flags.

Upstream-released checkpoints package the same way; pass the dataset that defines the I/O mapping and the embodiment:

```python
Gr00tEngine.export_openvino("GR00T-N1.7-LIBERO/libero_10", "exports/libero",
                            dataset_path="/data/libero_demo", embodiment_tag="LIBERO_PANDA")
```

## Runtime packages

```text
<package>/
├── manifest.json            backend "gr00t", reset callback
└── gr00t_n1d7.gr00t/
    ├── package.json         format, embodiment, I/O mapping, pins, export verification
    ├── model/               native checkpoint (safetensors and JSON only)
    └── openvino/            Suite single IR plus config/
```

`package.json` records the environment name, its stable package identity key, and its pins (upstream commit, patch digests, patched tree). Runtime refuses a package whose environment key differs from the installed plugin's. The I/O mapping comes from the dataset's `meta/modality.json`: camera names, state slices, and action slices.

`Engine.load(package, seed=0)` draws GR00T's flow-matching noise from a seeded CPU generator, so native and OpenVINO packages receive identical noise and their actions can be compared directly.

## Model access

`nvidia/GR00T-N1.7-3B` and `nvidia/GR00T-N1.7-LIBERO` are public. Their backbone repository, `nvidia/Cosmos-Reason2-2B`, is gated: accept its license on Hugging Face before training or inference. GR00T builds the backbone from that repository's configuration and processor files, then overwrites every backbone weight with the checkpoint's.

For development without that access, `Gr00tTrainConfig(backbone_source="Qwen/Qwen3-VL-2B-Instruct")` (or `PHYSICALAI_GR00T_BACKBONE_SOURCE`) builds the identically shaped public Qwen3-VL-2B architecture instead. Tokenization and image preprocessing then differ from NVIDIA's model, so packages record the substitution and Runtime warns when loading them. Do not deploy such packages.

## Security

- No pickle crosses the process boundary: frames carry JSON plus raw buffers of numeric dtypes.
- Workers refuse to execute code from model repositories, even though upstream requests `trust_remote_code=True`.
- Packages copy only safetensors, JSON, and YAML; checkpoint pickles such as `training_args.bin` are excluded.
- GR00T's ZMQ `PolicyServer` is not used: it binds `0.0.0.0` and decodes msgpack-numpy (which unpickles object arrays) before checking its token.

## Validation

`library/tests/integration/test_engine_gr00t_e2e.py` runs the whole workflow. It is opt-in because it downloads about 20 GB and needs a CUDA GPU:

```bash
cd library
PHYSICALAI_RUN_GR00T_E2E=1 PHYSICALAI_GR00T_E2E_OUTPUT=/tmp/gr00t-e2e \
  uv run --no-sync pytest tests/integration/test_engine_gr00t_e2e.py -s
```

It provisions both environments through core and runs two scenarios with `nvidia/Cosmos-Reason2-2B`, writing `report.json`. Results on an RTX A6000 host with 16 CPU cores:

| Check                                      | Fine-tuned (20 steps, batch 8)                   | Released `GR00T-N1.7-LIBERO`                       |
| ------------------------------------------ | ------------------------------------------------ | -------------------------------------------------- |
| Native package and Runtime (CUDA)          | 12.6 GB; loads in 11 s; 0.13 s per 16-step chunk | 6.9 GB; loads in 12 s; 0.15 s per chunk            |
| OpenVINO export (Suite patches)            | 105 s; 6.3 GB fp16 IR; cosine 1.0, MAE 2.0e-5    | 98 s; 6.3 GB fp16 IR; cosine 0.9999994, MAE 8.2e-5 |
| OpenVINO Runtime (CPU)                     | loads in 25 s; 1.2 s per chunk                   | loads in 25 s; 1.4 s per chunk                     |
| Native vs OpenVINO, same seeded noise      | mean 5.1e-4, max 4.2e-3                          | mean 9.8e-4, max 6.3e-3                            |
| Chunk MAE vs dataset actions (zero: 0.216) | 0.36 (20 steps is not a trained policy)          | 0.0133 native, 0.0134 OpenVINO                     |
| Reset and replay; package after inference  | identical; unchanged                             | identical; unchanged                               |

The released checkpoint's agreement with recorded actions checks the Runtime I/O mapping end to end: a swapped camera or a misaligned state or action slice would not reach 0.013.

Provisioning from an empty uv cache takes about 4 minutes for `native` (15 GB, including TensorRT and flash-attn) and under a minute for `openvino` (2 GB). Unit tests run without network, GPU, or weights:

```bash
cd plugins/policies/gr00t && uv run --project ../../../library pytest tests
```

## Limitations

- Training requires CUDA (upstream flash-attn and DeepSpeed build). XPU training is not enabled.
- Datasets must be LeRobot v2.1 with `meta/modality.json`; Studio's LeRobot v3.0 datasets need conversion.
- Observations must have a single frame per modality (`delta_indices == [0]`).
- Deployment needs the backbone repository in the Hugging Face cache: GR00T constructs the backbone before loading checkpoint weights.
