from uuid import uuid4

import pytest
from botocore.exceptions import SSOTokenLoadError
from fastapi import FastAPI
from fastapi.testclient import TestClient

from schemas.base_job import JobStatus
from schemas.job import RemoteTrainJobPayload, TrainJob
from schemas.job_response import JobResponse, job_response
from services.aws_auth_service import authentication_issue, operational_authentication_issue


@pytest.mark.parametrize("status", list(JobStatus))
def test_public_snapshot_preserves_storage_and_matches_http(status: JobStatus) -> None:
    issue = authentication_issue(SSOTokenLoadError(error_msg="expired"), "studio")
    assert issue is not None
    job = TrainJob(
        project_id=uuid4(),
        status=status,
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            remote_trainer_id=uuid4(),
            policy="act",
            model_name="model",
            authentication_issue=issue,
        ),
    )
    stored = job.model_dump_json()
    restored = TrainJob.model_validate_json(stored)
    snapshot = job_response(restored).model_dump(mode="json")
    assert "authentication_issue" not in snapshot["payload"]
    assert snapshot["issues"] == (
        [operational_authentication_issue(issue).model_dump()]
        if status in {JobStatus.PENDING, JobStatus.RUNNING}
        else []
    )
    assert job.model_dump_json() == stored
    app = FastAPI()

    @app.get("/job", response_model=JobResponse)
    def get_job() -> TrainJob:
        return restored

    assert TestClient(app).get("/job").json() == snapshot


def test_issue_identity_is_shared_across_observers_and_distinct_between_profiles() -> None:
    first = authentication_issue(SSOTokenLoadError(error_msg="first"), "studio")
    repeated = authentication_issue(SSOTokenLoadError(error_msg="second"), "studio")
    other = authentication_issue(SSOTokenLoadError(error_msg="first"), "other")
    assert first is not None and repeated is not None and other is not None
    assert operational_authentication_issue(first).id == operational_authentication_issue(repeated).id
    assert operational_authentication_issue(first).id != operational_authentication_issue(other).id
