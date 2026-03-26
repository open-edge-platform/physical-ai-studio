# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PushT benchmark for evaluating policies on the PushT task.

Example:
    >>> from physicalai.benchmark.pusht import PushTBenchmark

    >>> benchmark = PushTBenchmark(num_envs=1)
    >>> results = benchmark.evaluate(policy)
    >>> print(results.overall_success_rate)
"""

from physicalai.benchmark.pusht.pusht import PushTBenchmark

__all__ = [
    "PushTBenchmark",
]
