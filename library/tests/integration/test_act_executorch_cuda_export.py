# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration test: ExecuTorch export of a CUDA-resident ACT policy.

Regression test for a bug where ``to_executorch()`` traced the model and its
input sample directly on their live device instead of moving a copy to CPU
first. ExecuTorch's delegates (``"portable"``, ``"xnnpack"``) and its
EValue/flatbuffer serialization format are CPU/edge-only, with no CUDA/XPU
device case; tracing and lowering a CUDA-resident graph through
``to_edge_transform_and_lower()`` -> ``to_executorch()`` reached that missing
case in native code and segfaulted the whole process -- a crash, not a
catchable Python exception -- instead of exporting or raising cleanly.

This mirrors the real code path that regressed: `training.job.run_training_job`
(``application/backend/src/training/job.py``) trains a policy and exports it
in the same process without ever moving it back to CPU, whether training runs
locally in Studio or in the standalone remote trainer service.

This class of bug is invisible on CPU: the model and its export path never
leave the CPU device there, so every existing CPU-only export test (including
the rest of this policy's export suite) passes whether or not the fix is
present. Catching it requires a real CUDA device. Run explicitly with:

    pytest -m slow tests/integration/test_act_executorch_cuda_export.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from physicalai.data import LeRobotDataModule
from physicalai.policies import get_policy
from physicalai.policies.act.policy import ACT
from physicalai.train import Trainer

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="regression test for a CUDA-only crash; requires a real GPU",
    ),
]


@pytest.fixture(scope="module")
def datamodule() -> LeRobotDataModule:
    """A small real dataset, enough for one training step."""
    return LeRobotDataModule(
        repo_id="lerobot/pusht",
        train_batch_size=8,
        episodes=list(range(2)),
    )


@pytest.fixture(scope="module")
def cuda_trained_act(datamodule: LeRobotDataModule) -> ACT:
    """Train ACT for one step on CUDA and leave it resident there for export.

    No ``.cpu()`` call anywhere here, deliberately: that is exactly what the
    trainer does not do either between training and export (see
    ``training.job.run_training_job`` -> ``_export``).
    """
    policy = get_policy("act", source="physicalai")
    trainer = Trainer(
        accelerator="cuda",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(policy, datamodule=datamodule)
    # Lightning's strategy teardown moves the module back to CPU once `fit()`
    # returns, to free GPU memory for whatever runs next -- expected
    # framework behavior, not the bug this test guards against. The crash
    # this regresses against reproduces whenever the *policy* is resident on
    # an accelerator at export time (e.g. `training.job.run_training_job`
    # exports immediately after training, before any such reset would apply
    # in every Trainer/strategy combination). Force CUDA residency
    # explicitly so the test exercises that condition deterministically,
    # independent of Lightning's own post-fit teardown timing.
    return policy.to("cuda")


class TestACTExecuTorchExportFromCUDA:
    """``to_executorch()`` must succeed -- without crashing the process -- from a CUDA-resident policy."""

    def test_policy_is_actually_cuda_resident_before_export(self, cuda_trained_act: ACT) -> None:
        """Sanity check: if this fails, the regression below would pass for the wrong reason."""
        assert next(cuda_trained_act.model.parameters()).device.type == "cuda"

    def test_export_succeeds_and_produces_a_pte_file(self, cuda_trained_act: ACT, tmp_path: Path) -> None:
        export_dir = tmp_path / "act_executorch_cuda"

        cuda_trained_act.export(export_dir, backend="executorch")

        assert (export_dir / "manifest.json").is_file()
        assert any(export_dir.glob("*.pte"))

    def test_export_does_not_move_the_live_policy_off_cuda(self, cuda_trained_act: ACT, tmp_path: Path) -> None:
        """``to_executorch()`` must trace a copy, not mutate the live, CUDA-resident policy in place.

        A later export backend in the same job (OpenVINO, ONNX) relies on the
        policy still being where training left it.
        """
        cuda_trained_act.export(tmp_path / "act_executorch_cuda_2", backend="executorch")

        assert next(cuda_trained_act.model.parameters()).device.type == "cuda"
