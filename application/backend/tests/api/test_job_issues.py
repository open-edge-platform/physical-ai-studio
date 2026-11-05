from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from botocore.exceptions import SSOTokenLoadError
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient

from api.dependencies import get_job_service
from api.job import jobs_websocket, router
from schemas.base_job import JobStatus
from schemas.job import RemoteTrainJobPayload, TrainJob
from services.aws_auth_service import authentication_issue
from services.event_processor import EventType


async def test_existing_issue_and_recovery_match_http_and_websocket() -> None:
    job = TrainJob(
        project_id=uuid4(),
        status=JobStatus.RUNNING,
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            remote_trainer_id=uuid4(),
            policy="act",
            model_name="model",
            authentication_issue=authentication_issue(SSOTokenLoadError(error_msg="expired"), "studio"),
        ),
    )
    service = MagicMock(get_job_list=AsyncMock(return_value=[job]), get_job_by_id=AsyncMock(return_value=job))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_job_service] = lambda: service
    client = TestClient(app)
    snapshot = client.get("/api/jobs").json()[0]
    assert snapshot["issues"]
    assert client.get(f"/api/jobs/{job.id}").json() == snapshot

    events = MagicMock()
    websocket = MagicMock(accept=AsyncMock(), send_json=AsyncMock())

    async def receive(_mode: str) -> None:
        callback = events.subscribe.call_args.args[1]
        await callback(EventType.JOB_UPDATE, job)
        assert isinstance(job.payload, RemoteTrainJobPayload)
        job.payload.authentication_issue = None
        await callback(EventType.JOB_UPDATE, job)
        raise WebSocketDisconnect

    websocket.receive_json = receive
    await jobs_websocket(websocket, events)
    updates = websocket.send_json.call_args_list
    assert updates[0].args[0] == {"event": EventType.JOB_UPDATE, "data": snapshot}
    assert updates[1].args[0]["data"] == client.get("/api/jobs").json()[0]
    assert updates[1].args[0]["data"]["issues"] == []
    events.unsubscribe.assert_called_once()
