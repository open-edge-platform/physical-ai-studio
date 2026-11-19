# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run upstream GR00T fine-tuning inside the pinned native environment.

The worker executes Isaac-GR00T's own ``launch_finetune.py`` with the native
command-line flags. It adds only Physical AI policy around the native run:

- remote model code is refused (upstream defaults to ``trust_remote_code``);
- training progress and cancellation flow over the host protocol descriptor;
- an explicit backbone-source override can replace the gated default
  backbone repository for development (``PHYSICALAI_GR00T_BACKBONE_SOURCE``).

It also records ``physicalai_run.json`` beside the native checkpoint so a later
export knows the dataset, embodiment, and pins that produced the checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import runpy
import sys
import traceback
from pathlib import Path
from typing import Any

import gr00t_worker_common as common

RUN_INFO_NAME = "physicalai_run.json"
_BACKBONE_ENV = "PHYSICALAI_GR00T_BACKBONE_SOURCE"


def main(argv: list[str]) -> int:
    """Train with native GR00T arguments and report the result to the host.

    Returns:
        Process exit status.
    """
    channel = common.EventChannel()
    channel.send({"type": "hello", "protocol": common.PROTOCOL_VERSION, **common.identity()})
    cancel = common.start_cancel_listener() if channel.enabled else None
    try:
        result = _train(argv, channel, cancel)
    except common.CancelledError as exc:
        channel.send({"type": "cancelled", "step": exc.step})
        return 130
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code == 0:
            channel.send({"type": "result", "output_dir": None, "checkpoint": None})
        else:
            channel.send({"type": "error", "error": "SystemExit", "message": str(exc.code)})
        return code
    except Exception as exc:  # ruff: ignore[blind-except]  # Report any native failure over the protocol.
        channel.send({"type": "error", "error": type(exc).__name__, "message": str(exc)})
        traceback.print_exc(file=sys.stderr)
        return 1
    channel.send({"type": "result", **result})
    return 0


def _train(argv: list[str], channel: common.EventChannel, cancel: Any) -> dict[str, str]:  # ruff: ignore[any-type]
    common.disable_remote_code()
    script = common.source_dir() / "gr00t" / "experiment" / "launch_finetune.py"
    backbone = os.environ.get(_BACKBONE_ENV) or None
    state: dict[str, Any] = {}

    from gr00t.experiment import experiment
    from transformers import AutoModel, TrainerCallback

    native_run = experiment.run

    def run(config: Any) -> Any:  # ruff: ignore[any-type]
        config.training.transformers_trust_remote_code = False
        if backbone:
            config.model.model_name = backbone
        state["config"] = config
        state["output_dir"] = _output_dir(config)
        return native_run(config)

    experiment.run = run
    if backbone:
        _override_checkpoint_backbone(AutoModel, backbone)
    experiment.Gr00tTrainer = _with_progress(experiment.Gr00tTrainer, TrainerCallback, channel, cancel, state)

    original_argv = sys.argv
    sys.argv = [str(script), *argv]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv

    if "output_dir" not in state:
        msg = "Native fine-tuning finished without building a training configuration."
        raise RuntimeError(msg)
    output_dir: Path = state["output_dir"]
    _write_run_info(output_dir, state, backbone)
    if state.get("cancelled_step") is not None:
        raise common.CancelledError(state["cancelled_step"])
    return {"output_dir": str(output_dir), "checkpoint": str(output_dir)}


def _override_checkpoint_backbone(auto_model: Any, backbone: str) -> None:  # ruff: ignore[any-type]
    """Make checkpoint loading build the backbone from *backbone*.

    The base checkpoint's ``config.json`` names the backbone repository used to
    construct the backbone before checkpoint weights overwrite it. Passing
    ``model_name`` updates that configuration attribute at load time.
    """
    native_from_pretrained = auto_model.from_pretrained

    def from_pretrained(*args: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type]
        kwargs.setdefault("model_name", backbone)
        return native_from_pretrained(*args, **kwargs)

    auto_model.from_pretrained = from_pretrained


def _with_progress(trainer_class: type, callback_class: type, channel: Any, cancel: Any, state: dict) -> type:  # ruff: ignore[any-type]
    """Add a callback that reports one progress event per optimizer step.

    Transformers calls ``on_log`` after ``on_step_end`` for the same step, so a
    step's event is held until its metrics arrive: at its own log, at the next
    step, or when training ends.

    Returns:
        A trainer subclass that installs the progress callback.
    """

    class ProgressCallback(callback_class):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {}
            self.pending: tuple[int, int | None] | None = None

        def flush(self) -> None:
            if self.pending is not None:
                step, total = self.pending
                channel.send({"type": "progress", "step": step, "total": total, "metrics": dict(self.metrics)})
                self.pending = None

        def on_log(self, args: Any, train_state: Any, control: Any, logs: Any = None, **kwargs: Any) -> Any:  # ruff: ignore[any-type, unused-method-argument]
            for name, value in (logs or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    self.metrics[name] = float(value)
            self.flush()
            return control

        def on_step_end(self, args: Any, train_state: Any, control: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type, unused-method-argument]
            self.flush()
            step = int(train_state.global_step)
            state["global_step"] = step
            self.pending = (step, int(train_state.max_steps) if train_state.max_steps else None)
            if cancel is not None and cancel.is_set():
                state["cancelled_step"] = step
                control.should_training_stop = True
            return control

        def on_train_end(self, args: Any, train_state: Any, control: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type, unused-method-argument]
            self.flush()
            return control

    class ProgressTrainer(trainer_class):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # ruff: ignore[any-type]  # Forwarded unchanged to the native trainer.
            super().__init__(*args, **kwargs)
            self.add_callback(ProgressCallback())

    return ProgressTrainer


def _output_dir(config: Any) -> Path:  # ruff: ignore[any-type]
    output_dir = Path(config.training.output_dir)
    if config.training.experiment_name is not None:
        output_dir /= config.training.experiment_name
    return output_dir.resolve()


def _dataset(config: Any) -> tuple[str | None, str | None]:  # ruff: ignore[any-type]
    datasets = list(getattr(config.data, "datasets", None) or [])
    if not datasets:
        return None, None
    first = datasets[0]
    paths = first.get("dataset_paths") if isinstance(first, dict) else getattr(first, "dataset_paths", None)
    tag = first.get("embodiment_tag") if isinstance(first, dict) else getattr(first, "embodiment_tag", None)
    path = str(Path(paths[0]).resolve()) if paths else None
    return path, (str(getattr(tag, "value", tag)) if tag is not None else None)


def _write_run_info(output_dir: Path, state: dict, backbone: str | None) -> None:
    config = state["config"]
    dataset_path, embodiment_tag = _dataset(config)
    info = {
        "format": 1,
        "framework": "gr00t",
        "policy": "gr00t_n1d7",
        "worker": common.identity(),
        "base_model_path": config.training.start_from_checkpoint,
        "dataset_path": dataset_path,
        "embodiment_tag": embodiment_tag,
        "backbone_source": config.model.model_name,
        "backbone_override": backbone is not None,
        "max_steps": config.training.max_steps,
        "global_step": state.get("global_step"),
        "cancelled_step": state.get("cancelled_step"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RUN_INFO_NAME).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
