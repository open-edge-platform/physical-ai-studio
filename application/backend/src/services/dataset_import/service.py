import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import InvalidJobStateError, ResourceNotFoundError, ResourceType
from repositories import JobRepository
from schemas import Job
from schemas.base_job import JobStatus, JobType
from schemas.dataset_import_job import DatasetImportFinalizeInput, DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob
from services.dataset_import.staging import generate_staging_id, resolve_payload_archive_path
from services.staged_archive import cleanup_staged_archive


class DatasetImportService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = JobRepository(session)

    @staticmethod
    async def _load_validated_dataset_import_job(
        repo: JobRepository,
        *,
        project_id: UUID,
        job_id: UUID,
    ) -> tuple[Job, DatasetImportJobPayload]:
        job = await repo.get_by_id(job_id)
        if job is None:
            raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
        if job.project_id != project_id:
            raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
        if job.type != JobType.DATASET_IMPORT:
            raise InvalidJobStateError("Job is not a dataset import job")
        if not isinstance(job.payload, DatasetImportJobPayload):
            raise InvalidJobStateError("Dataset import job payload is invalid")

        return job, job.payload

    async def prepare_dataset_import_job(
        self,
        project_id: UUID,
        format_hint: str = "auto",
        dataset_name: str = "",
    ) -> Job:
        payload = DatasetImportJobPayload(
            archive_staging_id=generate_staging_id(),
            format_hint=format_hint,
            dataset_name=dataset_name or None,
            step=ImportStep.AWAITING_ARCHIVE_UPLOAD,
        )
        job = DatasetImportJob(
            project_id=project_id,
            payload=payload,
            message="Dataset import job prepared, awaiting upload",
        )
        try:
            return await self.repo.save(job)
        except IntegrityError:
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=project_id)

    async def attach_dataset_import_archive(
        self,
        project_id: UUID,
        job_id: UUID,
        uploaded_archive_name: str,
    ) -> Job:
        job, payload = await self._load_validated_dataset_import_job(self.repo, project_id=project_id, job_id=job_id)
        if payload.step != ImportStep.AWAITING_ARCHIVE_UPLOAD:
            raise InvalidJobStateError(
                f"Archive can only be attached when job is in '{ImportStep.AWAITING_ARCHIVE_UPLOAD}' step"
            )

        payload.uploaded_archive_name = uploaded_archive_name
        payload.step = ImportStep.QUEUED_FOR_DETECTION

        updates = {
            "payload": payload.model_dump(mode="json"),
            "status": JobStatus.PENDING,
            "message": "Dataset queued for importing",
            "progress": 5,
        }
        return await self.repo.update(job, updates)

    async def claim_pending_dataset_import_job(self) -> Job | None:
        return await self.repo.claim_pending_dataset_import_job()

    async def finalize_dataset_import_job(
        self,
        project_id: UUID,
        job_id: UUID,
        finalize_input: DatasetImportFinalizeInput,
    ) -> Job:
        job, payload = await self._load_validated_dataset_import_job(self.repo, project_id=project_id, job_id=job_id)
        if payload.step != ImportStep.AWAITING_USER_REVIEW:
            raise InvalidJobStateError(
                f"Dataset import can only be finalized from '{ImportStep.AWAITING_USER_REVIEW}' step"
            )

        payload.finalize_input = finalize_input
        payload.step = ImportStep.QUEUED_FOR_IMPORT

        updates = {
            "payload": payload.model_dump(mode="json"),
            "status": JobStatus.PENDING,
            "message": "Dataset import finalized and queued",
            "progress": 45,
        }

        return await self.repo.update(job, updates)

    async def cancel_dataset_import_job(self, project_id: UUID, job_id: UUID) -> Job:
        job, payload = await self._load_validated_dataset_import_job(self.repo, project_id=project_id, job_id=job_id)
        if payload.step != ImportStep.AWAITING_USER_REVIEW:
            raise InvalidJobStateError(
                f"Dataset import can only be canceled from '{ImportStep.AWAITING_USER_REVIEW}' step"
            )

        updates = {
            "status": JobStatus.CANCELED,
            "message": "Dataset import canceled",
            "end_time": datetime.datetime.now(tz=datetime.UTC),
        }
        updated_job = await self.repo.update(job, updates)
        cleanup_staged_archive(resolve_payload_archive_path(payload))
        return updated_job
