# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker-side helpers that run inside the GR00T environments."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_WORKER_DIR = Path(__file__).resolve().parents[1] / "src" / "physicalai_gr00t_policy_plugin" / "_worker"


@pytest.fixture
def workers(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    """Import worker scripts the way the worker interpreter does (as top-level modules).

    Returns:
        The imported worker modules by name.
    """
    monkeypatch.syspath_prepend(str(_WORKER_DIR))
    names = ("gr00t_worker_common", "gr00t_infer_worker", "gr00t_train_worker")
    return {name: importlib.import_module(name) for name in names}


def test_event_channel_tags_protocol_messages_on_stderr(
    workers: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = workers["gr00t_worker_common"]
    writes: list[tuple[int, bytes]] = []

    def write(file_descriptor: int, payload: bytes) -> int:
        writes.append((file_descriptor, payload))
        return len(payload)

    monkeypatch.setenv("PHYSICALAI_WORKER_PROTOCOL_FD", "2")
    monkeypatch.setattr(common.os, "write", write)
    common.EventChannel().send({"type": "hello", "protocol": 1})

    expected = bytes([30]) + b'physicalai-worker {"type":"hello","protocol":1}' + bytes([10])
    assert writes == [(2, expected)]


def test_seeded_noise_is_identical_across_call_styles_devices_and_dtypes(workers: dict[str, ModuleType]) -> None:
    torch = pytest.importorskip("torch")
    seeded_noise = workers["gr00t_infer_worker"]._seeded_noise
    with seeded_noise(7):
        native_style = torch.randn(size=(1, 4, 3), dtype=torch.bfloat16, device="cpu")
    with seeded_noise(7):
        openvino_style = torch.randn(1, 4, 3, dtype=torch.float32)
    with seeded_noise(8):
        other_seed = torch.randn(1, 4, 3)
    assert native_style.dtype == torch.bfloat16
    torch.testing.assert_close(native_style.float(), openvino_style.bfloat16().float())
    assert not torch.equal(openvino_style, other_seed)
    with seeded_noise(None):
        assert torch.randn(2).shape == (2,)


def test_remote_code_is_refused_after_hardening(workers: dict[str, ModuleType]) -> None:
    pytest.importorskip("transformers")
    dynamic = importlib.import_module("transformers.dynamic_module_utils")
    configuration_auto = importlib.import_module("transformers.models.auto.configuration_auto")
    saved = {
        module: module.__dict__.get("get_class_from_dynamic_module")
        for module in list(sys.modules.values())
        if getattr(module, "__name__", "").startswith("transformers")
        and "get_class_from_dynamic_module" in getattr(module, "__dict__", {})
    }
    try:
        workers["gr00t_worker_common"].disable_remote_code()
        with pytest.raises(RuntimeError, match="disabled by the Physical AI GR00T plugin"):
            dynamic.get_class_from_dynamic_module("x.Y", "repo")
        with pytest.raises(RuntimeError, match="disabled"):
            configuration_auto.get_class_from_dynamic_module("x.Y", "repo")
    finally:
        for module, function in saved.items():
            module.__dict__["get_class_from_dynamic_module"] = function


class _Channel:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def send(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


def test_training_progress_reports_steps_metrics_and_cancellation(workers: dict[str, ModuleType]) -> None:
    class Callback:
        pass

    class Trainer:
        def __init__(self) -> None:
            self.callbacks: list[Any] = []

        def add_callback(self, callback: Any) -> None:  # ruff: ignore[any-type]
            self.callbacks.append(callback)

    channel = _Channel()
    cancel = SimpleNamespace(is_set=lambda: False)
    state: dict[str, Any] = {}
    trainer = workers["gr00t_train_worker"]._with_progress(Trainer, Callback, channel, cancel, state)()
    callback = trainer.callbacks[0]
    control = SimpleNamespace(should_training_stop=False)

    # Transformers calls on_log after on_step_end for the same step.
    callback.on_step_end(None, SimpleNamespace(global_step=1, max_steps=3), control)
    assert channel.messages == []
    logs = {"loss": 0.5, "epoch": 0.1, "flag": True, "bad": float("nan")}
    callback.on_log(None, SimpleNamespace(global_step=1, max_steps=3), control, logs=logs)
    assert channel.messages == [{"type": "progress", "step": 1, "total": 3, "metrics": {"loss": 0.5, "epoch": 0.1}}]

    callback.on_step_end(None, SimpleNamespace(global_step=2, max_steps=3), control)
    cancel.is_set = lambda: True
    callback.on_step_end(None, SimpleNamespace(global_step=3, max_steps=3), control)
    callback.on_train_end(None, SimpleNamespace(global_step=3, max_steps=3), control)
    assert [message["step"] for message in channel.messages] == [1, 2, 3]
    assert control.should_training_stop is True
    assert state == {"global_step": 3, "cancelled_step": 3}


def test_openvino_cache_view_keeps_the_package_unchanged(workers: dict[str, ModuleType], tmp_path: Path) -> None:
    package_dir = tmp_path / "package" / "openvino"
    package_dir.mkdir(parents=True)
    (package_dir / "model.xml").write_text("<net/>", encoding="utf-8")
    cache_view = workers["gr00t_infer_worker"]._cache_view
    link = cache_view(package_dir, tmp_path / "cache" / "view")
    assert link.parent == tmp_path / "cache" / "view"
    assert (link / "model.xml").read_text(encoding="utf-8") == "<net/>"
    assert link.parent / ".ov_cache" != package_dir.parent / ".ov_cache"
    assert cache_view(package_dir, tmp_path / "cache" / "view") == link
    assert sorted(path.name for path in (tmp_path / "package").iterdir()) == ["openvino"]


def test_run_info_records_the_dataset_embodiment_and_backbone(
    workers: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "0.1.0")
    config = SimpleNamespace(
        training=SimpleNamespace(
            output_dir=str(tmp_path),
            experiment_name="exp",
            start_from_checkpoint="base",
            max_steps=3,
        ),
        model=SimpleNamespace(model_name="Qwen/Qwen3-VL-2B-Instruct"),
        data=SimpleNamespace(
            datasets=[
                {"dataset_paths": [str(tmp_path / "data")], "embodiment_tag": SimpleNamespace(value="libero_sim")},
            ],
        ),
    )
    train_worker = workers["gr00t_train_worker"]
    output_dir = train_worker._output_dir(config)
    assert output_dir == (tmp_path / "exp").resolve()
    train_worker._write_run_info(output_dir, {"config": config, "global_step": 3}, "Qwen/Qwen3-VL-2B-Instruct")
    info = (output_dir / train_worker.RUN_INFO_NAME).read_text(encoding="utf-8")
    assert '"embodiment_tag": "libero_sim"' in info
    assert '"backbone_override": true' in info
    assert '"global_step": 3' in info


def test_worker_identity_and_checkout_come_from_core_variables(
    workers: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = workers["gr00t_worker_common"]
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "0.1.0")
    monkeypatch.setenv("PHYSICALAI_WORKER_FRAMEWORK", "gr00t")
    assert common.identity() == {"framework": "gr00t", "package": "gr00t", "version": "0.1.0"}

    monkeypatch.delenv("PHYSICALAI_WORKER_PROJECT", raising=False)
    with pytest.raises(RuntimeError, match="PHYSICALAI_WORKER_PROJECT"):
        common.source_dir()
    monkeypatch.setenv("PHYSICALAI_WORKER_PROJECT", "/cache/gr00t/project")
    assert common.source_dir() == Path("/cache/gr00t/project")
