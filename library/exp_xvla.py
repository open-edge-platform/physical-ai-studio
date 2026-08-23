# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scratch experiment: benchmark the published XVLA LIBERO checkpoint on LIBERO-10."""

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.policies.xvla.libero import XVLALiberoPolicy

# Load the published checkpoint. Architecture fields (action space, domain id, soft
# prompts) and the normalization statistics come from the repo's config/preprocessor.
# XVLALiberoPolicy bridges this checkpoint's 20-dim bimanual ee6d action space to
# LiberoGym's 7-dim single-arm interface; see physicalai.policies.xvla.libero.
policy = XVLALiberoPolicy(pretrained_name_or_path="lerobot/xvla-libero")
policy.eval()

# control_mode="absolute" is the one environment setting this checkpoint needs: it predicts
# an absolute target end-effector pose per step, not a delta. Leaving it at LiberoGym's
# "relative" default scores 0/5 on the first five tasks -- silently, without raising.
# Everything else stays at default: render resolution and a longer step budget were both
# measured as immaterial (see the ablation table in physicalai.policies.xvla.libero).
benchmark = LiberoBenchmark(
    task_suite="libero_10",
    num_episodes=1,
    max_steps=520,
    control_mode="absolute",
)
results = benchmark.evaluate(policy)

# View results
print(results.summary())  # noqa: T201
results.to_json("results_xvla.json")
