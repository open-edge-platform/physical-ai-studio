# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Types for gym environments."""

import numpy as np

type ScalarVec[T] = T | list[T] | np.ndarray
