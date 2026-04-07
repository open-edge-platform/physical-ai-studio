# Inference Core Design Document

> **Scope note:** This document describes the design of the **domain‑agnostic inference core** — the layer that provides backend execution, metadata IO, and the base `InferenceModel`. In our proposed architecture, this layer lives **inside physicalai** as `physicalai.inference`, not as a separate package. References to "inferencekit" in this document describe the module's design; the module path is `physicalai.inference.*`. This layer can be silently extracted as a standalone package later if other domains (e.g., vision via model_api) need it independently.

**Base inference framework providing unified model loading, prediction, and extensibility across backends and domains.**

---

## Table of Contents

- [Inference Core Design Document](#inference-core-design-document)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Purpose](#purpose)
    - [Architecture Position](#architecture-position)
    - [Design Goals](#design-goals)
    - [Non-Goals](#non-goals)
  - [Architecture](#architecture)
    - [Package Structure](#package-structure)
    - [Design Principles](#design-principles)
  - [Core Components](#core-components)
    - [InferenceModel](#inferencemodel)
    - [RuntimeAdapter](#runtimeadapter)
    - [InferenceRunner](#inferencerunner)
    - [Callback System](#callback-system)
    - [Preprocessors and Postprocessors](#preprocessors-and-postprocessors)
    - [Manifest Format](#manifest-format)
  - [Extension \& Plugin System](#extension--plugin-system)
    - [Backend Registry](#backend-registry)
    - [Building a Custom Domain Layer](#building-a-custom-domain-layer)
    - [Publishing to HuggingFace](#publishing-to-huggingface)
  - [Runners (Domain-Provided)](#runners-domain-provided)
    - [Contrib Runners](#contrib-runners)
  - [Supported Backends](#supported-backends)
  - [Domain Layer Examples](#domain-layer-examples)
    - [Example 1: Vision (model_api)](#example-1-vision-model_api)
    - [Example 2: Physical‑AI Plugins](#example-2-physicalai-plugins)
    - [Example 3: Custom Domain](#example-3-custom-domain)
  - [Usage Examples](#usage-examples)
    - [Basic usage](#basic-usage)
    - [With explicit backend](#with-explicit-backend)
    - [With callbacks](#with-callbacks)
    - [Context manager for resource cleanup](#context-manager-for-resource-cleanup)
  - [API Reference](#api-reference)
    - [Main Entry Point](#main-entry-point)
    - [Runners](#runners)
    - [Adapters](#adapters)
    - [Callbacks](#callbacks)
    - [Plugins](#plugins)
    - [Extension Points](#extension-points)
  - [Appendix: Design Rationale](#appendix-design-rationale)
    - [Why a separate inference package?](#why-a-separate-inference-package)
    - [Why inferencekit is a base layer, not a model_api replacement](#why-inferencekit-is-a-base-layer-not-a-model_api-replacement)
    - [Migration path for model_api](#migration-path-for-model_api)
    - [Why runners are separate from adapters?](#why-runners-are-separate-from-adapters)
    - [Why callbacks instead of inheritance?](#why-callbacks-instead-of-inheritance)
    - [Why a plugin system?](#why-a-plugin-system)
  - [Related Documents](#related-documents)

---

## Overview

### Purpose

**inferencekit** is the base execution engine for the Geti ecosystem. It standardizes backend execution and metadata IO. It provides:

- Backend abstraction (OpenVINO, ONNX, TensorRT, Torch)
- Manifest loading (`manifest.json`)
- Minimal `InferenceModel(path)` + `model(inputs)` API

**inferencekit knows nothing about vision, robotics, or any specific domain.** Domain plugins live above it (physical‑ai‑framework, model_api, custom layers).

### Architecture Position

inferencekit is the **foundation layer** in a layered architecture. Domain-specific systems build on top of it, each adding their own preprocessing, postprocessing, runners, and model types:

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                       Domain Layers                                           │
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────────────────┐  ┌─────────────────┐  │
│  │    model_api     │  │  physical‑ai‑framework       │  │  custom-xyz     │  │
│  │  (vision)        │  │  (physical‑AI)               │  │  (your domain)  │  │
│  │                  │  │                              │  │                 │  │
│  │  YOLO, SAM,      │  │  Policy plugins:             │  │  Your models,   │  │
│  │  Anomaly, OTX,   │  │  physicalai-train, LeRobot,  │  │  your runners,  │  │
│  │  Ultralytics,    │  │  custom frameworks           │  │  publishable    │  │
│  │  Roboflow        │  │                              │  │  on HuggingFace │  │
│  └────────┬─────────┘  └────────┬─────────────────────┘  └───────┬─────────┘  │
│           │                     │                     │                       │
│           └─────────────────────┼─────────────────────┘                       │
│                                 │                                             │
│                          depends on                                           │
│                                 │                                             │
│                                 ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐             │
│  │                       inferencekit                           │             │
│  │                    (base framework)                          │             │
│  │                                                              │             │
│  │  InferenceModel  │  RuntimeAdapter  │  InferenceRunner       │             │
│  │  Callbacks       │  Pre/Post ABCs   │  Plugin Registry       │             │
│  │  OpenVINO, ONNX, TensorRT, Torch backends                    │             │
│  └──────────────────────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Key principle:** Domain layers depend on inferencekit. inferencekit depends on nothing domain-specific. Dependencies flow upward only.

| Layer                     | Owns                                                                                                                                                      | Does NOT own                          |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **inferencekit**          | Backend adapters, manifest IO, base InferenceModel                                                                                                        | Vision models, robotics, domain logic |
| **model_api**             | Vision preprocessing, task wrappers (YOLO, SAM), result types                                                                                             | Backend execution, robotics           |
| **physical‑ai‑framework** | Policy plugins, unified APIs, orchestration, observation pipeline, safety runtime, episode orchestration, device management, camera/robot interfaces, CLI | Backend execution, training           |
| **custom layers**         | Domain-specific models, runners, pre/postprocessors                                                                                                       | Backend execution, core infra         |

### Design Goals

| Goal                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| **G1: Execution Engine**     | Provide backend execution and manifest IO                    |
| **G2: Minimal API**          | `InferenceModel(path)` + `model(inputs)` across backends     |
| **G3: Backend Agnostic**     | Support OpenVINO, ONNX, TensorRT, Torch without code changes |
| **G4: Minimal Dependencies** | Core has few requirements; optional extras per backend       |
| **G5: Domain Agnostic**      | No vision, robotics, or domain-specific code                 |

### Non-Goals

| Non-Goal                                        | Rationale                        |
| ----------------------------------------------- | -------------------------------- |
| Vision preprocessing/postprocessing             | Belongs in model_api             |
| Physical‑AI orchestration                       | Belongs in physical‑ai‑framework |
| Training infrastructure                         | Separate concern                 |
| Result types (DetectionResult, etc.)            | Domain-layer concern             |
| Framework-specific wrappers (Ultralytics, etc.) | Domain-layer concern             |

---

## Architecture

### Package Structure

```text
inferencekit/
├── __init__.py                              # Public API: InferenceModel
├── model.py                                 # InferenceModel - main entry point
├── runners/
│   ├── __init__.py
│   ├── base.py                              # InferenceRunner ABC
│   ├── single_pass.py                       # SinglePassRunner (default)
│   ├── batch.py                             # BatchRunner
│   └── streaming.py                         # StreamingRunner
├── adapters/
│   ├── __init__.py                          # get_adapter() factory
│   ├── base.py                              # RuntimeAdapter ABC
│   ├── openvino.py                          # OpenVINO backend
│   ├── onnx.py                              # ONNX Runtime backend
│   ├── tensorrt.py                          # TensorRT backend
│   └── torch_export.py                      # Torch Export IR / ExecuTorch
├── callbacks/
│   ├── __init__.py
│   ├── base.py                              # Callback ABC
│   ├── timing.py                            # TimingCallback
│   └── logging.py                           # LoggingCallback
├── preprocessors/
│   ├── __init__.py
│   └── base.py                              # Preprocessor ABC
├── postprocessors/
│   ├── __init__.py
│   └── base.py                              # Postprocessor ABC
├── io/
│   ├── __init__.py
│   ├── manifest.py                          # Manifest loading (JSON)
│   └── instantiate.py                       # class_path + init_args
├── plugins/
│   ├── __init__.py                          # Plugin registry + entry points
│   ├── base.py                              # Plugin ABC
│   └── registry.py                          # BackendRegistry, RunnerRegistry
└── contrib/
    ├── __init__.py
    ├── iterative.py                         # IterativeRunner (flow-matching)
    └── tiled.py                             # TiledRunner (large inputs)
```

### Design Principles

| Principle                        | Description                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| **Foundation, Not Application**  | inferencekit provides ABCs and infrastructure; domain layers provide implementations |
| **Composition over Inheritance** | Runners, callbacks, adapters are composable building blocks                          |
| **Progressive Disclosure**       | Simple API for 90% of users, full control for power users                            |
| **Plugin-First Extensibility**   | New backends, runners, formats via registry + entry points                           |
| **Minimal Dependencies**         | Core has few requirements; backends and contrib are optional extras                  |

---

## Core Components

### InferenceModel

The main entry point for inference. Orchestrates runners, adapters, and callbacks.

**Design Philosophy:**

90% of users should only need:

```python
from inferencekit import InferenceModel

model = InferenceModel("./exports/my_model")
outputs = model(inputs)
```

Progressive customization for advanced users:

```python
# Tier 2: Override parameters
model = InferenceModel(
    "./exports/my_model",
    backend="onnx",
    device="cuda",
)

# Tier 3: Explicit components
from inferencekit.callbacks import TimingCallback

model = InferenceModel(
    "./exports/my_model",
    callbacks=[TimingCallback()],
)

# Tier 4: Full control (domain layers use this)
from inferencekit.adapters import ONNXAdapter
from inferencekit.runners import SinglePassRunner

adapter = ONNXAdapter(device="cuda")
adapter.load(Path("./model.onnx"))
runner = SinglePassRunner()
model = InferenceModel(adapter=adapter, runner=runner)
```

**API:**

```python
class InferenceModel:
    """Unified inference interface for exported models.

    Automatically detects backend, device, and configuration from
    export directory metadata. Domain layers can subclass or compose
    this to add domain-specific behavior.

    Callable: use model(inputs) to run inference.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        backend: str | None = None,
        device: str = "auto",
        callbacks: list[Callback] | None = None,
        *,
        adapter: RuntimeAdapter | None = None,
        runner: InferenceRunner | None = None,
        **kwargs,
    ):
        """Initialize and load model.

        Args:
            path: Directory containing exported model and metadata,
                  or HuggingFace URI (hf://user/repo)
            backend: Backend to use (auto-detected from metadata if None)
            device: Device for inference ("auto", "cpu", "cuda", "CPU", "GPU")
            callbacks: Optional callbacks for instrumentation
            adapter: Explicit RuntimeAdapter (advanced; skips auto-detection)
            runner: Explicit InferenceRunner (advanced; skips metadata lookup)
            **kwargs: Additional arguments passed to runner/adapter
        """
        ...

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference on inputs.

        Args:
            inputs: Dictionary mapping input names to arrays/tensors

        Returns:
            Dictionary mapping output names to arrays/tensors
        """
        ...

    def reset(self) -> None:
        """Reset model state (for stateful runners)."""
        ...

    def __enter__(self) -> "InferenceModel":
        """Context manager entry."""
        ...

    def __exit__(self, *args) -> None:
        """Context manager exit - cleanup resources."""
        ...
```

### RuntimeAdapter

Adapters execute **one forward pass** on a specific backend. Intentionally stateless.

```python
class RuntimeAdapter(ABC):
    """Abstract base class for backend-specific inference.

    Each backend (OpenVINO, ONNX, TensorRT, Torch) implements this
    interface. Domain layers should NOT need to subclass this — they
    compose adapters via runners and callbacks instead.
    """

    def __init__(self, device: str = "cpu", **kwargs):
        self.device = device
        self.config = kwargs

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load model from disk."""
        ...

    @abstractmethod
    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run single forward pass."""
        ...

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """Model input names."""
        ...

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Model output names."""
        ...
```

### InferenceRunner

Runners define **how inference runs** — the algorithm, not what happens to outputs.

Runners are implemented in domain layers (physical‑ai plugins, model_api, custom). inferencekit provides only the interface.

```python
class InferenceRunner(ABC):
    """Abstract base class for inference execution patterns.

    Runners control the inference algorithm: single pass, iterative
    denoising, tiled execution, streaming, etc. Domain layers should
    subclass InferenceRunner to implement domain-specific patterns.
    """

    @abstractmethod
    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        """Execute the inference pattern.

        Args:
            adapter: Backend adapter for forward passes
            inputs: Model inputs

        Returns:
            Model outputs
        """
        ...

    def reset(self) -> None:
        """Reset runner state between episodes/sequences."""
        pass
```

### Callback System

Lightning-compatible hooks for cross-cutting concerns:

```python
class Callback:
    """Base callback class. Override hooks as needed.

    Callbacks are the preferred way to add instrumentation,
    safety checks, logging, and other cross-cutting concerns
    without modifying model or runner code.
    """

    def on_predict_start(self, inputs: dict) -> dict | None:
        """Called before prediction. Can modify inputs."""
        pass

    def on_predict_end(self, outputs: dict) -> dict | None:
        """Called after prediction. Can modify outputs."""
        pass

    def on_reset(self) -> None:
        """Called when model state is reset."""
        pass

    def on_load(self, model: "InferenceModel") -> None:
        """Called after model is loaded."""
        pass
```

Callbacks are domain‑provided. inferencekit defines the interface; domain layers supply implementations.

### Preprocessors and Postprocessors

Transform inputs before inference and outputs after:

```python
class Preprocessor(ABC):
    """Transform inputs before inference.

    Domain layers implement concrete preprocessors:
    - model_api: ImageResize, ImageNormalize, LayoutTransform
    - physicalai-train: ObservationNormalizer, ActionUnnormalizer
    """

    @abstractmethod
    def __call__(self, inputs: dict) -> dict:
        ...

class Postprocessor(ABC):
    """Transform outputs after inference.

    Domain layers implement concrete postprocessors:
    - model_api: NMS, BoxDecoder, MaskDecoder
    - physicalai-train: ActionChunker, ActionClamp
    """

    @abstractmethod
    def __call__(self, outputs: dict) -> dict:
        ...
```

### Manifest Format

All exported models use a unified `manifest.json` format. The manifest uses `class_path` + `init_args` (following `jsonargparse` conventions) for component specification:

```json
{
  "format": "policy_package",
  "version": "1.0",
  "robots": [
    {
      "name": "main",
      "type": "Koch v1.1",
      "state": { "shape": [14], "dtype": "float32" },
      "action": { "shape": [14], "dtype": "float32" }
    }
  ],
  "cameras": [
    {
      "name": "top",
      "shape": [3, 480, 640],
      "dtype": "uint8"
    },
    {
      "name": "wrist",
      "shape": [3, 480, 640],
      "dtype": "uint8"
    }
  ],
  "policy": {
    "name": "my_model",
    "kind": "single_pass"
  },
  "artifacts": {
    "onnx": "model.onnx"
  },
  "runner": {
    "class_path": "inferencekit.runners.SinglePassRunner",
    "init_args": {}
  },
  "adapter": {
    "class_path": "inferencekit.adapters.ONNXAdapter",
    "init_args": {
      "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
    }
  },
  "preprocessors": [
    {
      "class_path": "mypackage.preprocessors.ImageResize",
      "init_args": {
        "target_size": [640, 640]
      }
    }
  ],
  "postprocessors": [
    {
      "class_path": "mypackage.postprocessors.NMS",
      "init_args": {
        "confidence_threshold": 0.5
      }
    }
  ]
}
```

**How models are loaded:**

The framework reads `manifest.json` and resolves the model configuration:

1. **Built‑in models** (physicalai-train, LeRobot): `policy.kind` maps to a built‑in runner. No `class_path` needed for the runner — the `kind` field is sufficient.
2. **Custom/exotic models**: `runner.class_path` points to the user's runner class. The framework instantiates it dynamically.
3. **Hardware validation**: `robots` and `cameras` sections declare expected shapes. The runtime validates observations against these on first contact.

The `class_path` + `init_args` pattern allows domain layers to specify their own components in the manifest without inferencekit needing to know about them.

---

## Extension & Plugin System

inferencekit only supports **backend adapters** as extensions. All domain plugins live above it (physical‑ai‑framework, model_api, custom layers).

### Backend Registry

inferencekit exposes a backend registry for RuntimeAdapters. Domain plugins are not registered here.

### Building a Custom Domain Layer

Anyone can create a domain-specific inference layer on top of inferencekit. Here's the pattern:

**Step 1: Define your domain model**

```python
# my_domain_inference/model.py
from inferencekit import InferenceModel

class MyDomainModel(InferenceModel):
    """Domain-specific inference model.

    Extends InferenceModel with domain-specific methods,
    preprocessing, and postprocessing.
    """

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        # Attach domain preprocessors/postprocessors
        self.preprocessors = self._load_preprocessors(path)
        self.postprocessors = self._load_postprocessors(path)

    def domain_predict(self, domain_inputs):
        """Domain-specific prediction method."""
        # Preprocess domain inputs -> generic inputs
        inputs = self._preprocess(domain_inputs)
        # Run generic inference
        outputs = self(inputs)
        # Postprocess generic outputs -> domain outputs
        return self._postprocess(outputs)
```

**Step 2: Define domain-specific runners (if needed)**

```python
# my_domain_inference/runners.py
from inferencekit.runners import InferenceRunner

class MyDomainRunner(InferenceRunner):
    """Runner for domain-specific inference patterns."""

    def run(self, adapter, inputs):
        # Implement domain-specific execution logic
        ...
```

**Step 3: Register via entry points**

```toml
# my_domain_inference/pyproject.toml
[project.entry-points."inferencekit.runners"]
my_domain_runner = "my_domain_inference.runners:MyDomainRunner"
```

**Step 4: Package and distribute**

```bash
# Publish to PyPI
pip install my-domain-inference

# Or publish to HuggingFace (see below)
```

### Publishing to HuggingFace

Domain layers can publish model packages to HuggingFace that include:

1. **Exported model artifacts** (ONNX, OpenVINO, etc.)
2. **Manifest** (`manifest.json`) specifying the inferencekit runner, preprocessors, etc.
3. **Domain package dependency** declared in the manifest

```json
{
  "format": "policy_package",
  "version": "1.0",
  "robots": [...],
  "cameras":[...],
  "policy": {
    "name": "my_model",
    "kind": "custom"
  },
  "domain_package": "my-domain-inference",
  "artifacts": {
    "onnx": "model.onnx"
  },
  "runner": {
    "class_path": "my_domain_inference.runners.MyDomainRunner",
    "init_args": {
      "param1": "value1"
    }
  },
  "preprocessors": [
    {
      "class_path": "my_domain_inference.preprocessors.MyPreprocessor",
      "init_args": {}
    }
  ]
}
```

**Loading from HuggingFace:**

```python
from inferencekit import InferenceModel

# Auto-downloads model + resolves domain package
model = InferenceModel("hf://username/my-model")
outputs = model(inputs)
```

---

## Runners (Domain-Provided)

inferencekit defines the `InferenceRunner` interface. Domain layers implement concrete runners.

```python
class SinglePassRunner(InferenceRunner):
    """Default runner. Covers 90% of use cases."""

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        return adapter.predict(inputs)

    def reset(self) -> None:
        pass  # No state


class BatchRunner(InferenceRunner):
    """Batched inference for throughput optimization."""

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        # Split inputs into batches, run, merge results
        ...


class StreamingRunner(InferenceRunner):
    """Streaming inference for real-time applications."""

    def __init__(self, buffer_size: int = 1):
        self.buffer_size = buffer_size

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        # Process streaming inputs with buffering
        ...
```

### Contrib Runners

If desired, inferencekit can host a small `contrib` module for reference implementations, but it does not own domain logic.

```python
# inferencekit/contrib/iterative.py
class IterativeRunner(InferenceRunner):
    """Runner for iterative/flow-matching inference.

    Performs multiple forward passes with denoising steps.
    Used by diffusion models, flow-matching policies, etc.
    """

    def __init__(
        self,
        num_steps: int = 10,
        scheduler: str = "euler",
        timestep_spacing: str = "linear",
    ):
        self.num_steps = num_steps
        self.scheduler = scheduler
        self.timestep_spacing = timestep_spacing

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        x_t = np.random.randn(*self._infer_shape(inputs)).astype(np.float32)
        timesteps = self._generate_timesteps()
        dt = -1.0 / self.num_steps

        for t in timesteps:
            step_inputs = {**inputs, "x_t": x_t, "timestep": np.array([t])}
            v_t = adapter.predict(step_inputs)["v_t"]
            x_t = self._step(x_t, v_t, dt)

        return {"output": x_t}
```

```python
# inferencekit/contrib/tiled.py
class TiledRunner(InferenceRunner):
    """Runner for tile-based inference on large inputs.

    Splits large inputs into overlapping tiles, runs inference
    on each tile, and merges results. Useful for high-resolution
    images, satellite imagery, medical imaging, etc.
    """

    def __init__(
        self,
        tile_size: tuple[int, int] = (640, 640),
        overlap: float = 0.25,
        merge_strategy: str = "average",
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.merge_strategy = merge_strategy

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        tiles = self._split_into_tiles(inputs)
        tile_results = [adapter.predict(tile) for tile in tiles]
        return self._merge_results(tile_results)
```

Domain layers can also contribute runners back to `inferencekit.contrib` via pull request, or ship them in their own packages.

---

## Supported Backends

| Backend             | Hardware             | Status      | Installation                         |
| ------------------- | -------------------- | ----------- | ------------------------------------ |
| **OpenVINO**        | Intel CPU/GPU/NPU    | Implemented | `pip install inferencekit[openvino]` |
| **ONNX Runtime**    | Cross-platform, CUDA | Implemented | `pip install inferencekit[onnx]`     |
| **TensorRT**        | NVIDIA GPU           | Planned     | `pip install inferencekit[tensorrt]` |
| **Torch Export IR** | CPU/CUDA, ExecuTorch | Implemented | Built-in                             |

Third-party backends can be added via the backend registry without modifying inferencekit.

---

## Domain Layer Examples

These examples show how domain-specific libraries build on inferencekit's interfaces. Each example demonstrates the pattern; full implementations live in their respective packages.

### Example 1: Vision (model_api)

[model_api](https://github.com/open-edge-platform/model_api) provides vision-specific inference on top of inferencekit. It adds image preprocessing, task-specific model wrappers, and structured result types.

```python
# model_api wrapping inferencekit for vision inference
from inferencekit import InferenceModel
from inferencekit.runners import InferenceRunner, SinglePassRunner
from inferencekit.preprocessors import Preprocessor
from inferencekit.postprocessors import Postprocessor


# Vision-specific preprocessor
class ImagePreprocessor(Preprocessor):
    """Resize, normalize, and layout-transform images."""

    def __init__(self, target_size, mean, std, layout="NCHW"):
        self.target_size = target_size
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.layout = layout

    def __call__(self, inputs: dict) -> dict:
        image = inputs["image"]
        image = cv2.resize(image, self.target_size)
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        if self.layout == "NCHW":
            image = image.transpose(2, 0, 1)
        inputs["image"] = image[np.newaxis, ...]
        return inputs


# Vision-specific postprocessor (e.g., NMS for detection)
class DetectionPostprocessor(Postprocessor):
    """Decode detection outputs and apply NMS."""

    def __init__(self, confidence_threshold=0.5, nms_threshold=0.45):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def __call__(self, outputs: dict) -> dict:
        boxes, scores, labels = self._decode(outputs)
        keep = self._nms(boxes, scores)
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
        }


# Vision model built on top of InferenceModel
class DetectionModel(InferenceModel):
    """YOLO/SSD/etc. detection model."""

    def __init__(self, path, confidence=0.5, **kwargs):
        super().__init__(path, **kwargs)
        self.preprocessors = [
            ImagePreprocessor(
                target_size=(640, 640),
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]
        self.postprocessors = [
            DetectionPostprocessor(confidence_threshold=confidence)
        ]

    def detect(self, image: np.ndarray) -> dict:
        """Convenience method for vision users."""
        return self({"image": image})
```

**Usage:**

```python
from model_api import DetectionModel

model = DetectionModel("./exports/yolo_v8", backend="openvino")
detections = model.detect(image)
print(detections["boxes"], detections["scores"])
```

### Example 2: Physical‑AI Plugins

physicalai hosts policy plugins for physicalai-train, LeRobot, and custom frameworks. Each plugin supplies preprocessors, runners, and optional wrappers.

```python
# physical‑ai‑framework plugin example (policy-specific)
from inferencekit import InferenceModel
from inferencekit.runners import InferenceRunner


# Policy-specific runner with action chunking
class ActionChunkingRunner(InferenceRunner):
    """Runner that manages action chunk queues.

    Policies output action chunks (multiple future actions).
    This runner queues them and dispenses one action per call.
    """

    def __init__(self, chunk_size: int = 16, n_action_steps: int = 1):
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self._action_queue = []

    def run(self, adapter, inputs):
        if not self._action_queue:
            outputs = adapter.predict(inputs)
            chunk = outputs["action"]  # shape: (chunk_size, action_dim)
            self._action_queue = list(chunk[:self.n_action_steps])

        action = self._action_queue.pop(0)
        return {"action": action}

    def reset(self):
        self._action_queue = []


```

Policy‑specific behavior (e.g., `select_action`, episode reset) is implemented in physical‑ai‑framework’s `InferenceModel` wrapper, which subclasses inferencekit’s base `InferenceModel`.

### Example 3: Custom Domain

Anyone can build a domain layer. Here's a minimal example for audio inference:

```python
# audio_inference/model.py
from inferencekit import InferenceModel
from inferencekit.preprocessors import Preprocessor


class AudioPreprocessor(Preprocessor):
    """Convert audio to mel spectrogram."""

    def __init__(self, sample_rate=16000, n_mels=80):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def __call__(self, inputs):
        audio = inputs["audio"]
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels
        )
        inputs["mel_spectrogram"] = mel
        return inputs


class AudioClassificationModel(InferenceModel):
    """Audio classification on top of inferencekit."""

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.preprocessors = [AudioPreprocessor()]

    def classify(self, audio: np.ndarray) -> dict:
        return self({"audio": audio})
```

**Package and publish:**

```toml
# audio_inference/pyproject.toml
[project]
name = "audio-inference-kit"
dependencies = ["inferencekit", "librosa"]

[project.entry-points."inferencekit.runners"]
audio_streaming = "audio_inference.runners:AudioStreamingRunner"
```

```bash
pip install audio-inference-kit
# or publish to HuggingFace with model artifacts + metadata
```

---

## Usage Examples

### Basic usage

```python
from inferencekit import InferenceModel

# Load model (auto-detects backend)
model = InferenceModel("./exports/my_model")

# Run inference
inputs = {"input": data_array}
outputs = model(inputs)
```

### With explicit backend

```python
model = InferenceModel(
    "./exports/my_model",
    backend="openvino",
    device="CPU",
)
```

### With callbacks

```python
from inferencekit.callbacks import TimingCallback, LoggingCallback

model = InferenceModel(
    "./exports/my_model",
    callbacks=[TimingCallback(), LoggingCallback()],
)

# Callbacks fire automatically
outputs = model(inputs)
```

### Context manager for resource cleanup

```python
with InferenceModel("./exports/my_model") as model:
    outputs = model(inputs)
# Resources automatically cleaned up
```

---

## API Reference

### Main Entry Point

```python
from inferencekit import InferenceModel

model = InferenceModel("./exports/my_model")
outputs = model(inputs)
```

### Runners

```python
from inferencekit.runners import (
    InferenceRunner,      # ABC - subclass for custom runners
    SinglePassRunner,     # Default - covers 90% of models
    BatchRunner,          # Throughput-optimized batching
    StreamingRunner,      # Real-time streaming
)

# Contrib runners (install with inferencekit[contrib])
from inferencekit.contrib import (
    IterativeRunner,      # Multi-step denoising / flow matching
    TiledRunner,          # Tile-based for large inputs
)
```

### Adapters

```python
from inferencekit.adapters import (
    RuntimeAdapter,       # ABC
    OpenVINOAdapter,      # Intel devices
    ONNXAdapter,          # Cross-platform
    TorchExportAdapter,   # PyTorch
    get_adapter,          # Factory function
)
```

### Callbacks

```python
from inferencekit.callbacks import (
    Callback,             # ABC
    TimingCallback,       # Performance profiling
    LoggingCallback,      # Prediction logging
)
```

### Plugins

```python
from inferencekit.plugins import registry

# List available backends
print(registry.backends.list())

# Register custom backend
registry.backends.register("my_backend", MyBackend)

# Get a backend by name
adapter = registry.backends.get("onnx", device="cuda")
```

### Extension Points

| Extension        | How to Extend               | Registration                          |
| ---------------- | --------------------------- | ------------------------------------- |
| New backend      | Implement `RuntimeAdapter`  | Entry point: `inferencekit.backends`  |
| New runner       | Implement `InferenceRunner` | Entry point: `inferencekit.runners`   |
| New model format | Implement format plugin     | Entry point: `inferencekit.formats`   |
| New callback     | Subclass `Callback`         | Entry point: `inferencekit.callbacks` |
| Preprocessing    | Implement `Preprocessor`    | Via metadata `class_path`             |
| Postprocessing   | Implement `Postprocessor`   | Via metadata `class_path`             |

---

## Appendix: Design Rationale

### Why a separate inference package?

1. **Reusability**: Same core across vision (model_api), robotics (physicalai-train), audio, NLP, and custom domains
2. **Clear boundaries**: Generic concerns (backends, metadata, plugins) separated from domain concerns (images, robots, audio)
3. **Easier testing**: Domain-agnostic package has fewer dependencies
4. **Ecosystem growth**: Anyone can build and publish domain layers without modifying inferencekit

### Why inferencekit is a base layer, not a model_api replacement

model_api provides rich vision-specific functionality: image preprocessing embedded in model graphs, task-specific wrappers (YOLO, SSD, SAM), result types, parameter validation, and tiling. These are vision concerns that don't belong in a generic inference framework.

Instead of replacing model_api, inferencekit provides the **foundation** that model_api can build on:

| Concern           | inferencekit provides                  | model_api adds                           |
| ----------------- | -------------------------------------- | ---------------------------------------- |
| Backend execution | RuntimeAdapter (OV, ONNX, TRT)         | Wraps RuntimeAdapter in InferenceAdapter |
| Model loading     | Manifest-driven `InferenceModel(path)` | Vision-specific `Model.create_model()`   |
| Preprocessing     | Preprocessor ABC                       | ImageResize, Normalize, LayoutTransform  |
| Postprocessing    | Postprocessor ABC                      | NMS, BoxDecoder, MaskDecoder             |
| Runners           | SinglePassRunner, BatchRunner          | TiledRunner (via contrib or own impl)    |
| Result types      | `dict[str, Any]`                       | DetectionResult, ClassificationResult    |

### Migration path for model_api

1. **Phase 1 (compatibility)**: model_api wraps inferencekit's RuntimeAdapter inside its existing InferenceAdapter. No public API change.
2. **Phase 2 (adoption)**: model_api adopts RuntimeAdapter directly, deprecates its own adapter layer.
3. **Phase 3 (simplification)**: model_api becomes a pure domain layer on top of inferencekit.

### Why runners are separate from adapters?

- **Adapters** handle backend-specific execution (ONNX vs OpenVINO)
- **Runners** handle algorithm-specific patterns (single-pass vs iterative)
- This separation allows N backends × M inference patterns without N×M implementations

### Why callbacks instead of inheritance?

- **Composability**: Mix and match (timing + logging + safety)
- **Reusability**: Same callback works across all models and domains
- **Maintainability**: Add cross-cutting concerns without changing core code
- **Familiarity**: Lightning users already understand this pattern

### Why a plugin system?

- **Ecosystem growth**: Third parties can extend without forking
- **Clean dependencies**: inferencekit doesn't depend on domain packages
- **Discoverability**: Entry points make extensions automatically available
- **Publishability**: Domain layers can be packaged and shared independently

---

## Related Documents

- **[Strategy](../architecture/strategy.md)** — Big-picture architecture and layering decisions
- **[Architecture](../architecture/architecture.md)** — physicalai runtime CLI and packaging
- **[LeRobot Integration](../integrations/lerobot.md)** — LeRobot integration for physicalai (built‑in, reads manifest.json)

---

_Document Version: 3.0_
_Last Updated: 2026-02-16_
