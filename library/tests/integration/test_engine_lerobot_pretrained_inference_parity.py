# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in action parity checks for real, revision-pinned LeRobot Hub policies."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.integration import _device

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    import numpy as np

_MODEL_DATASETS = [
    pytest.param(
        "lerobot/act_aloha_sim_transfer_cube_human",
        "ba73b2766f1371cdc133ca4efb97eb090d744625",
        "lerobot/aloha_sim_transfer_cube_human",
        "6a43d500f101255823a9d2b9dc244eeb01a2cd31",
        id="pretrained-act-aloha-transfer-cube",
    ),
    pytest.param(
        "lerobot/diffusion_pusht",
        "84a7c23178445c6bbf7e1a884ff497017910f653",
        "lerobot/pusht",
        "b1c3ecbae7f244acc039a3dbc255a00dad1372b9",
        id="pretrained-diffusion-pusht",
    ),
]
_MODEL_DOWNLOAD_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.md"]
_MODEL_FILES = ("config.json", "train_config.json", "model.safetensors")
_SEQUENCE_LENGTH = 3


@pytest.mark.integration
@pytest.mark.requires_download
@pytest.mark.parametrize(("model_repo", "model_revision", "dataset_repo", "dataset_revision"), _MODEL_DATASETS)
def test_runtime_actions_match_pinned_pretrained_lerobot_policy_after_reset(
    model_repo: str,
    model_revision: str,
    dataset_repo: str,
    dataset_revision: str,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    """Compare actual pretrained ACT/Diffusion outputs through LeRobot and Runtime.

    Set ``PHYSICALAI_RUN_LEROBOT_PRETRAINED_INFERENCE=1`` to permit Hub access.
    Device selection defaults to ``PHYSICALAI_LEROBOT_DEVICE=auto``; override it
    only when intended. Detected GPU hardware that host PyTorch cannot use is an
    error, not a reason to run the parity test on CPU. Checkpoints and datasets
    are pinned to immutable Hub commit SHAs. The policy Hub repos currently publish
    native configs and weights but not the processor
    JSON files; when absent, LeRobot's native factory builds them from the pinned
    dataset statistics and saves them beside the local native checkpoint before
    both inference paths load those saved pipelines.
    """
    if os.environ.get("PHYSICALAI_RUN_LEROBOT_PRETRAINED_INFERENCE") != "1":
        pytest.skip("Set PHYSICALAI_RUN_LEROBOT_PRETRAINED_INFERENCE=1 to run pretrained Hub parity tests.")

    import gc
    import json
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    from httpx import RequestError
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import get_policy_class
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
    from lerobot.utils.import_utils import register_third_party_plugins
    from physicalai.engine import Engine
    from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

    device = _requested_device(torch)
    native_dir = tmp_path / "source-native" / "pretrained_model"
    native_dir.mkdir(parents=True)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model_repo,
                repo_type="model",
                revision=model_revision,
                allow_patterns=_MODEL_DOWNLOAD_ALLOW_PATTERNS,
            )
        )
    except (HfHubHTTPError, LocalEntryNotFoundError, RequestError) as error:
        pytest.skip(f"Pinned model {model_repo}@{model_revision} is unavailable: {type(error).__name__}: {error}")

    for filename in _MODEL_FILES:
        source_file = snapshot / filename
        if not source_file.is_file():
            pytest.skip(f"Pinned model {model_repo}@{model_revision} does not publish {filename}.")
        shutil.copyfile(source_file, native_dir / filename)

    try:
        dataset = LeRobotDataset(dataset_repo, revision=dataset_revision, episodes=[0])
    except (FileNotFoundError, HfHubHTTPError, LocalEntryNotFoundError, RequestError) as error:
        pytest.skip(f"Pinned dataset {dataset_repo}@{dataset_revision} is unavailable: {type(error).__name__}: {error}")
    if len(dataset) < _SEQUENCE_LENGTH:
        pytest.skip(f"Pinned dataset {dataset_repo}@{dataset_revision} has fewer than {_SEQUENCE_LENGTH} frames.")

    policy_config = PreTrainedConfig.from_pretrained(native_dir, cli_overrides=[f"--device={device}"])
    rows = [dataset[index] for index in range(_SEQUENCE_LENGTH)]
    for key, feature in policy_config.input_features.items():
        assert key in rows[0], f"Dataset {dataset_repo} is missing model input {key}."
        assert tuple(rows[0][key].shape) == tuple(feature.shape), (
            f"Dataset {dataset_repo} feature {key} has shape {tuple(rows[0][key].shape)}, "
            f"checkpoint expects {tuple(feature.shape)}."
        )

    preprocessor_path = native_dir / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    postprocessor_path = native_dir / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
    hub_processors_present = preprocessor_path.is_file() and postprocessor_path.is_file()
    if hub_processors_present:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=str(native_dir),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            dataset_stats=dataset.meta.stats,
        )
        preprocessor.save_pretrained(native_dir)
        postprocessor.save_pretrained(native_dir)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=str(native_dir),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
    assert preprocessor_path.is_file()
    assert postprocessor_path.is_file()

    register_third_party_plugins()
    policy = get_policy_class(policy_config.type).from_pretrained(native_dir, config=policy_config)
    policy.to(device).eval()
    native_first, native_after_reset = _native_action_sequences(
        policy,
        preprocessor,
        postprocessor,
        rows,
        policy_config.input_features,
        device,
    )
    del policy, preprocessor, postprocessor
    gc.collect()

    package = LeRobotEngine.export(native_dir, output_path=tmp_path / "runtime-package")
    runtime_model = Engine.load(package, device=device)
    runtime_first, runtime_after_reset = _runtime_action_sequences(
        runtime_model, rows, policy_config.input_features, device
    )

    rtol, atol = _action_tolerances(device)
    np.testing.assert_allclose(native_first, native_after_reset, rtol=rtol, atol=atol)
    np.testing.assert_allclose(runtime_first, runtime_after_reset, rtol=rtol, atol=atol)
    np.testing.assert_allclose(native_first, runtime_first, rtol=rtol, atol=atol)
    np.testing.assert_allclose(native_after_reset, runtime_after_reset, rtol=rtol, atol=atol)
    assert np.isfinite(native_first).all()
    assert np.isfinite(runtime_first).all()

    summary = {
        "model": model_repo,
        "model_revision": model_revision,
        "dataset": dataset_repo,
        "dataset_revision": dataset_revision,
        "device": device,
        "steps": _SEQUENCE_LENGTH,
        "hub_processors_present": hub_processors_present,
        "rtol": rtol,
        "atol": atol,
        "max_abs_difference": float(np.max(np.abs(native_first - runtime_first))),
        "native_actions": native_first.tolist(),
        "runtime_actions": runtime_first.tolist(),
    }
    record_property("pretrained_inference_parity", json.dumps(summary, sort_keys=True))
    print(f"Pretrained inference parity: {json.dumps(summary, sort_keys=True)}")


def _requested_device(torch: Any) -> str:
    """Select the hardware-aware default and reject an unusable GPU runtime.

    Returns:
        LeRobot's native CPU, CUDA, or XPU device string.

    Raises:
        RuntimeError: If selected GPU hardware is unavailable to host PyTorch.
        ValueError: If the environment selects an unsupported device.
    """
    return _device.select_lerobot_test_device(
        os.environ.get("PHYSICALAI_LEROBOT_DEVICE", "auto"),
        torch_module=torch,
    )


def _native_action_sequences(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    rows: Sequence[Mapping[str, Any]],
    input_features: Mapping[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one identical native observation sequence twice with an explicit reset.

    Returns:
        Actions from the initial and post-reset native-policy runs.
    """
    import numpy as np
    import torch

    def run_sequence() -> np.ndarray:
        _reset_native_inference(policy, preprocessor, postprocessor)
        _seed_inference_rng(torch, device)
        actions = []
        with torch.inference_mode():
            for row in rows:
                batch = {key: row[key].clone() for key in input_features}
                action = postprocessor(policy.select_action(preprocessor(batch)))
                actions.append(action.detach().float().cpu().numpy().reshape(-1))
        return np.stack(actions)

    return run_sequence(), run_sequence()


def _runtime_action_sequences(
    runtime_model: Any,
    rows: Sequence[Mapping[str, Any]],
    input_features: Mapping[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one identical Runtime observation sequence twice with an explicit reset.

    Returns:
        Actions from the initial and post-reset Runtime runs.
    """
    import numpy as np
    import torch

    def run_sequence() -> np.ndarray:
        runtime_model.reset()
        _seed_inference_rng(torch, device)
        actions = []
        for row in rows:
            observation: dict[str, object] = {}
            images: dict[str, object] = {}
            for key in input_features:
                value = row[key].detach().cpu().numpy().copy()
                if key == "observation.state":
                    observation["state"] = value
                elif key == "observation.image":
                    observation["image"] = value
                elif key.startswith("observation.images."):
                    images[key.removeprefix("observation.images.")] = value
                else:
                    observation[key] = value
            if images:
                observation["images"] = images
            action = runtime_model.select_action(observation)
            actions.append(np.asarray(action).reshape(-1))
        return np.stack(actions)

    return run_sequence(), run_sequence()


def _reset_native_inference(policy: Any, preprocessor: Any, postprocessor: Any) -> None:
    for component in (policy, preprocessor, postprocessor):
        reset = getattr(component, "reset", None)
        if callable(reset):
            reset()


def _seed_inference_rng(torch: Any, device: str) -> None:
    import random

    import numpy as np

    seed = 2718
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore[attr-defined]
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    elif device == "xpu":
        torch.xpu.manual_seed_all(seed)  # type: ignore[attr-defined]


def _action_tolerances(device: str) -> tuple[float, float]:
    """Use tight CPU and modest accelerator tolerances for float32 parity.

    Returns:
        Relative and absolute tolerances. Identical CPU paths should be exact;
        accelerator kernels allow a small amount of float32 operation-order drift.
    """
    return (1e-5, 1e-6) if device == "cpu" else (1e-3, 1e-4)
