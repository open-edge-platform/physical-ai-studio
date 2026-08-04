from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from schemas.job import TrainingTarget, TrainJobPayload
from schemas.remote_trainer import RemoteTrainer
from services.job_service import JobService

MODULE = "services.job_service"


def _session_context() -> AsyncMock:
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.mark.anyio
async def test_submit_remote_job_pins_configured_url_and_ignores_client_url() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.is_job_duplicate = AsyncMock(return_value=False)
    repository.save = AsyncMock(side_effect=lambda job: job)
    remote_trainer_id = uuid4()
    configured_trainer = RemoteTrainer(
        id=remote_trainer_id,
        name="trainer",
        url="https://configured-trainer.test",
    )
    payload = TrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        training_target=TrainingTarget.REMOTE,
        remote_trainer_id=remote_trainer_id,
        remote_trainer_url="https://client-supplied.test",
    )

    remote_trainer_service = MagicMock()
    remote_trainer_service.get_remote_trainer = AsyncMock(return_value=configured_trainer)

    with patch(f"{MODULE}.JobRepository", return_value=repository):
        job = await JobService(session, remote_trainer_service).submit_train_job(payload)

    assert str(job.payload.remote_trainer_url) == "https://configured-trainer.test/"
    assert job.payload.remote_trainer_id == remote_trainer_id
    # Pinned from the configured trainer's record, for display in job logs.
    assert job.payload.remote_trainer_name == "trainer"
    repository.save.assert_awaited_once()
