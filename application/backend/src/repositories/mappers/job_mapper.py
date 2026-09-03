from pydantic import TypeAdapter

from db.schema import JobDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Job
from schemas.base_job import JobType
from schemas.job import TrainingTarget

JOB_ADAPTER = TypeAdapter(Job)

_LOCAL_ONLY_EXTRA_KEYS = ("remote_trainer_id", "remote_trainer_url", "remote_trainer_name")


class JobMapper(IBaseMapper):
    @staticmethod
    def to_schema(db_schema: Job) -> JobDB:
        job_data = db_schema.model_dump()
        job_data["payload"] = db_schema.payload.model_dump(mode="json")
        return JobDB(**job_data)

    @staticmethod
    def from_schema(model: JobDB) -> Job:
        # Backward compatibility: jobs persisted before `training_target` was
        # added to `TrainJobPayload` have no such key, which trips the
        # discriminated union with `union_tag_not_found` instead of a normal
        # validation error. Infer it: a legacy payload carrying
        # `remote_trainer_id` was remote, everything else was local.
        payload = model.payload
        if model.type == JobType.TRAINING and isinstance(payload, dict):
            if "training_target" not in payload:
                is_remote = bool(payload.get("remote_trainer_id"))
                payload["training_target"] = TrainingTarget.REMOTE if is_remote else TrainingTarget.LOCAL
            if payload["training_target"] == TrainingTarget.LOCAL:
                # Legacy local payloads also carried these (unused) remote-only
                # fields, which `LocalTrainJobPayload` now forbids as extras.
                for key in _LOCAL_ONLY_EXTRA_KEYS:
                    payload.pop(key, None)
        return JOB_ADAPTER.validate_python(model, from_attributes=True)
