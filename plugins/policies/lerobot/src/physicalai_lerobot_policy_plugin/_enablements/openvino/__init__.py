# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Direct OpenVINO IR export backend and torch-free bundle runtime for LeRobot.

This package intentionally keeps ``__init__.py`` free of eager imports so that
importing ``.runtime`` during bundle verification or rollout never loads PyTorch.
``lerobot.utils.import_utils.load_plugin`` imports all submodules automatically.
"""
