# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""JobStatus is generated into the UI's openapi-spec.d.ts and consumed widely.

Adding a member is a breaking change for every generated consumer, so this
pins the exact member set explicitly rather than relying on an implicit
"it still imports" check -- a future change (e.g. a GPU-busy "waiting" status)
must fail this test and be a deliberate decision, not an incidental one.
"""

from schemas.base_job import JobStatus


def test_job_status_has_exactly_the_expected_members() -> None:
    assert {status.value for status in JobStatus} == {
        "pending",
        "running",
        "completed",
        "failed",
        "canceled",
    }
