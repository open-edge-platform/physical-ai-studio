# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GR00T engine workflows with the Robotics AI Suite OpenVINO enablement.

Two scenarios share the pinned LIBERO demo data:

1. ``test_train_package_and_export``: fine-tune ``nvidia/GR00T-N1.7-3B`` through
   the engine, run the native package through Runtime, export through the Suite
   patches, run the OpenVINO package, and compare both backends.
2. ``test_released_checkpoint_predicts_dataset_actions``: package NVIDIA's
   released ``GR00T-N1.7-LIBERO`` checkpoint natively and through OpenVINO, and
   check that both predict the dataset's recorded actions. This catches I/O
   mapping errors (camera, state, or action slices) that backend parity cannot.

The tests are opt-in: they download about 25 GB, need a CUDA GPU for training,
and need access to ``nvidia/Cosmos-Reason2-2B`` on Hugging Face::

    PHYSICALAI_RUN_GR00T_E2E=1 uv run --no-sync pytest tests/integration/test_engine_gr00t_e2e.py -s

Optional variables: ``PHYSICALAI_GR00T_E2E_STEPS`` (default 20),
``PHYSICALAI_GR00T_E2E_BATCH`` (default 8), and ``PHYSICALAI_GR00T_E2E_OUTPUT``
(keeps runs, packages, and ``report.json``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PHYSICALAI_RUN_GR00T_E2E") != "1",
    reason="Set PHYSICALAI_RUN_GR00T_E2E=1 to run the GR00T end-to-end workflows.",
)

_SEED = 1234
_EPISODE = 0
_FRAMES = (0, 60, 120)
_HORIZON = 16
_DEMO = "demo_data/libero_demo"
_OPENVINO = "robotics-ai-suite-openvino"
_RELEASED_REPOSITORY = "nvidia/GR00T-N1.7-LIBERO"
_RELEASED_REVISION = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
_RELEASED_FILES = (
    "libero_10/config.json",
    "libero_10/embodiment_id.json",
    "libero_10/model-*.safetensors",
    "libero_10/model.safetensors.index.json",
    "libero_10/processor_config.json",
    "libero_10/statistics.json",
)
_PARITY_FRACTION = 0.05
_BASELINE_FRACTION = 0.5


class _Progress:
    def __init__(self) -> None:
        self.steps: list[tuple[int, int | None, dict[str, float]]] = []

    def on_step(self, step: int, total: int | None, metrics: dict[str, float]) -> None:
        self.steps.append((step, total, dict(metrics)))

    @staticmethod
    def should_stop() -> bool:
        return False


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    configured = os.environ.get("PHYSICALAI_GR00T_E2E_OUTPUT")
    path = Path(configured).expanduser() if configured else tmp_path_factory.mktemp("gr00t-e2e")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def environments_ready(workspace: Path) -> dict[str, float]:
    """Prepare both pinned environments through core and time them.

    Returns:
        Seconds spent preparing each environment.
    """
    from physicalai.engine import Engine, prepare_environment  # noqa: PLC0415

    engine = Engine.get("gr00t")
    timings = {}
    for name in (None, _OPENVINO):
        started = time.monotonic()
        prepare_environment(engine.environment(name))
        timings[f"prepare_{name or 'default'}"] = time.monotonic() - started
    _update_report(workspace, {"environments": timings})
    return timings


@pytest.fixture(scope="module")
def dataset(workspace: Path, environments_ready: dict[str, float]) -> Path:
    del environments_ready
    return _libero_demo(workspace / "data" / "libero_demo")


@pytest.fixture(scope="module")
def observations(dataset: Path) -> list[dict[str, Any]]:
    return _observations(dataset)


def test_train_package_and_export(workspace: Path, dataset: Path, observations: list[dict[str, Any]]) -> None:
    from physicalai.engine import Engine  # noqa: PLC0415
    from physicalai_gr00t_policy_plugin.config import Gr00tTrainConfig  # noqa: PLC0415

    steps = int(os.environ.get("PHYSICALAI_GR00T_E2E_STEPS", "20"))
    batch = int(os.environ.get("PHYSICALAI_GR00T_E2E_BATCH", "8"))
    engine_class = Engine.get("gr00t")
    timings: dict[str, float] = {}
    progress = _Progress()

    started = time.monotonic()
    run = Engine.create("gr00t").train(
        Gr00tTrainConfig(
            dataset_path=str(dataset),
            embodiment_tag="LIBERO_PANDA",
            output_dir=str(workspace / "trained" / "run"),
            max_steps=steps,
            global_batch_size=batch,
            save_steps=steps + 1,
            save_only_model=True,
        ),
        hooks=progress,
    )
    timings["train"] = time.monotonic() - started
    assert run.checkpoint is not None
    run_info = json.loads((run.checkpoint / "physicalai_run.json").read_text(encoding="utf-8"))
    assert run_info["backbone_source"] == "nvidia/Cosmos-Reason2-2B"
    assert not run_info["backbone_override"]
    assert run_info["environment"]["name"] == "default"
    assert [step for step, _, _ in progress.steps] == list(range(1, steps + 1))
    losses = [metrics["loss"] for _, _, metrics in progress.steps if "loss" in metrics]
    assert losses
    assert all(math.isfinite(loss) for loss in losses)

    report = _scenario(
        workspace / "trained",
        observations,
        native=lambda output: engine_class.export(run, output),
        openvino=lambda output: engine_class.export(run, output, backend="openvino"),
    )
    report["train"] = {"steps": steps, "global_batch_size": batch, "losses": losses, "seconds": timings["train"]}
    _update_report(workspace, {"trained": report})

    _assert_backends_agree(report)


def test_released_checkpoint_predicts_dataset_actions(
    workspace: Path,
    dataset: Path,
    observations: list[dict[str, Any]],
) -> None:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from physicalai_gr00t_policy_plugin.engine import Gr00tEngine  # noqa: PLC0415

    started = time.monotonic()
    # A local directory holds real files; Hugging Face cache snapshots are symlinks, which packaging rejects.
    snapshot = Path(
        snapshot_download(
            _RELEASED_REPOSITORY,
            revision=_RELEASED_REVISION,
            allow_patterns=list(_RELEASED_FILES),
            local_dir=workspace / "released" / "checkpoint",
        ),
    )
    download_seconds = time.monotonic() - started
    checkpoint = snapshot / "libero_10"
    options = {"dataset_path": dataset, "embodiment_tag": "LIBERO_PANDA"}

    report = _scenario(
        workspace / "released",
        observations,
        native=lambda output: Gr00tEngine.export_native(checkpoint, output, **options),
        openvino=lambda output: Gr00tEngine.export_openvino(checkpoint, output, **options),
    )
    report["checkpoint"] = {
        "repository": _RELEASED_REPOSITORY,
        "revision": _RELEASED_REVISION,
        "download_seconds": download_seconds,
    }
    _update_report(workspace, {"released": report})

    _assert_backends_agree(report)
    quality = report["quality"]
    for backend in ("native", "openvino"):
        assert quality[f"{backend}_chunk_mae"] < _BASELINE_FRACTION * quality["zero_action_chunk_mae"]


def _scenario(
    root: Path,
    observations: list[dict[str, Any]],
    *,
    native: Any,  # noqa: ANN401
    openvino: Any,  # noqa: ANN401
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    started = time.monotonic()
    native_package = native(root / "packages" / "native")
    timings["export_native"] = time.monotonic() - started
    native_actions, native_info = _infer(native_package, observations)

    started = time.monotonic()
    openvino_package = openvino(root / "packages" / "openvino")
    timings["export_openvino"] = time.monotonic() - started
    descriptor = json.loads((openvino_package / "gr00t_n1d7.gr00t" / "package.json").read_text(encoding="utf-8"))
    openvino_actions, openvino_info = _infer(openvino_package, observations)

    targets = np.stack([observation["_target"] for observation in observations])
    differences = np.abs(native_actions - openvino_actions)
    return {
        "timings_s": {
            **timings,
            **{f"native_{key}": value for key, value in native_info["timings"].items()},
            **{f"openvino_{key}": value for key, value in openvino_info["timings"].items()},
        },
        "export": descriptor["export"],
        "environment": descriptor["environment"]["name"],
        "backbone_source": descriptor["backbone_source"],
        "parity": {
            "shape": list(native_actions.shape),
            "max_abs_diff": float(differences.max()),
            "mean_abs_diff": float(differences.mean()),
            "action_abs_mean": float(np.abs(native_actions).mean()),
            "native_reset_replay_max_abs_diff": native_info["replay_max_abs_diff"],
            "openvino_reset_replay_max_abs_diff": openvino_info["replay_max_abs_diff"],
        },
        "quality": {
            "native_chunk_mae": float(np.abs(native_actions - targets).mean()),
            "openvino_chunk_mae": float(np.abs(openvino_actions - targets).mean()),
            "zero_action_chunk_mae": float(np.abs(targets).mean()),
        },
        "packages_bytes": {"native": _size(native_package), "openvino": _size(openvino_package)},
    }


def _assert_backends_agree(report: dict[str, Any]) -> None:
    parity = report["parity"]
    assert parity["shape"] == [len(_FRAMES), _HORIZON, 7]
    assert report["export"]["verification"]["cosine"] > 0.999  # noqa: PLR2004
    assert parity["native_reset_replay_max_abs_diff"] == 0.0
    assert parity["openvino_reset_replay_max_abs_diff"] == 0.0
    assert parity["mean_abs_diff"] < _PARITY_FRACTION * max(parity["action_abs_mean"], 1e-3)
    assert report["backbone_source"] == "nvidia/Cosmos-Reason2-2B"
    assert report["environment"] == _OPENVINO


def _infer(package: Path, observations: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    from physicalai.engine import Engine  # noqa: PLC0415

    before = _listing(package)
    started = time.monotonic()
    model = Engine.load(package, seed=_SEED)
    load_seconds = time.monotonic() - started
    with model:
        chunks = []
        latencies = []
        for observation in observations:
            started = time.monotonic()
            chunks.append(model.predict_action_chunk(_runtime_inputs(observation)))
            latencies.append(time.monotonic() - started)
        model.reset()
        replay = model.predict_action_chunk(_runtime_inputs(observations[0]))
    assert _listing(package) == before, "Inference must not modify the Runtime package."
    actions = np.stack(chunks)
    assert np.isfinite(actions).all()
    return actions, {
        "timings": {"load": load_seconds, "predict_mean": float(np.mean(latencies[1:] or latencies))},
        "replay_max_abs_diff": float(np.abs(replay - actions[0]).max()),
    }


def _runtime_inputs(observation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in observation.items() if not key.startswith("_")}


def _observations(dataset: Path) -> list[dict[str, Any]]:
    import av  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(dataset / "data" / "chunk-000" / f"episode_{_EPISODE:06d}.parquet").to_pydict()
    tasks = {
        record["task_index"]: record["task"]
        for record in map(json.loads, (dataset / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines())
    }
    frames: dict[str, list[np.ndarray]] = {}
    for camera in ("image", "wrist_image"):
        path = dataset / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{_EPISODE:06d}.mp4"
        with av.open(str(path)) as container:
            frames[camera] = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    actions = np.asarray(table["action"], dtype=np.float32)
    return [
        {
            "images": {camera: frames[camera][index] for camera in frames},
            "state": np.asarray(table["observation.state"][index], dtype=np.float32),
            "task": tasks[table["task_index"][index]],
            "_target": actions[index : index + _HORIZON],
        }
        for index in _FRAMES
    ]


def _libero_demo(destination: Path) -> Path:
    """Materialize the LIBERO demo dataset from the pinned upstream checkout.

    The checkout holds Git LFS pointers; each object is downloaded from GitHub's
    LFS media endpoint at the pinned commit and verified against its pointer.

    Returns:
        The dataset root.
    """
    from physicalai.engine import Engine, prepare_environment  # noqa: PLC0415
    from physicalai_gr00t_policy_plugin.environments import UPSTREAM  # noqa: PLC0415

    if (destination / "meta" / "modality.json").is_file():
        return destination
    source = prepare_environment(Engine.get("gr00t").environment()).project / _DEMO
    slug = UPSTREAM.repository.removeprefix("https://github.com/").removesuffix(".git")
    for pointer in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = pointer.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = pointer.read_bytes()
        if not text.startswith(b"version https://git-lfs"):
            target.write_bytes(text)
            continue
        fields = dict(line.split(" ", 1) for line in text.decode("utf-8").splitlines() if " " in line)
        expected = fields["oid"].removeprefix("sha256:")
        url = f"https://media.githubusercontent.com/media/{slug}/{UPSTREAM.revision}/{_DEMO}/{relative.as_posix()}"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310  # Pinned HTTPS URL.
            data = response.read()
        assert hashlib.sha256(data).hexdigest() == expected, f"LFS object mismatch for {relative}"
        target.write_bytes(data)
    return destination


def _update_report(workspace: Path, section: dict[str, Any]) -> None:
    path = workspace / "report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    report.update(section)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(section, indent=2))  # noqa: T201


def _size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _listing(path: Path) -> dict[str, int]:
    return {item.relative_to(path).as_posix(): item.stat().st_size for item in path.rglob("*")}
