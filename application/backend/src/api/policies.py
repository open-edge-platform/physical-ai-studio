import asyncio
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from loguru import logger
from pydantic import BaseModel

from api.dependencies import get_remote_trainer_service
from services.remote_trainer_service import RemoteTrainerService
from services.training_backends.local import resolve_hf_token
from training.policy_catalog import AvailablePolicy, available_policy_classes, policy_catalog
from training.policy_catalog import missing_policy_dependencies as _missing_policy_dependencies

router = APIRouter(prefix="/api/policies", tags=["Policies"])

_AccessStatus = Literal["granted", "missing_token", "denied", "unavailable", "not_required"]


_HUGGINGFACE_REQUIREMENTS = {
    # Hub dependencies are lazy and can vary with a policy's configuration, so
    # there is no library API that can discover them safely without starting a
    # download. When adding/changing a policy, inspect its `from_pretrained`,
    # `hf_hub_download`, and `Auto*from_pretrained` calls and list every default
    # repository that Studio training needs here.
    "molmoact2": (("allenai/MolmoAct2", False),),
    "pi05": (
        ("lerobot/pi05_base", True),
        ("google/paligemma-3b-pt-224", True),
    ),
    "smolvla": (("lerobot/smolvla_base", False),),
    "rldx1": (("RLWRLD/RLDX-1-PT", False),),
    "xr0": (
        ("XiaomiRobotics/Xiaomi-Robotics-0-Pretrain", False),
        ("Qwen/Qwen3-VL-4B-Instruct", False),
    ),
}


class HuggingFaceRequirementAccess(BaseModel):
    """Access status for one Hub repository a policy needs."""

    repository: str
    status: _AccessStatus
    access_url: str
    required: bool


class HuggingFaceAccessResponse(BaseModel):
    """Hub access statuses for every repository a policy requires."""

    requirements: list[HuggingFaceRequirementAccess]


def _available_policy_classes() -> dict[str, tuple[type, list[str]]]:
    """Resolve installed first-party policies and their export backends."""
    return available_policy_classes(_missing_policy_dependencies, logger.warning)


@router.get("/available")
async def get_available_policies(
    service: Annotated[RemoteTrainerService, Depends(get_remote_trainer_service)],
    remote_trainer_id: UUID | None = None,
) -> list[AvailablePolicy]:
    """Return policies available on the selected training target."""
    if remote_trainer_id is not None:
        return (await service.get_policy_catalog(remote_trainer_id)).available
    return policy_catalog(_available_policy_classes()).available


@router.get("/backends")
async def get_supported_backends_per_policy(
    service: Annotated[RemoteTrainerService, Depends(get_remote_trainer_service)],
    remote_trainer_id: UUID | None = None,
) -> dict[str, list[str]]:
    """Return export backends on the selected training target."""
    if remote_trainer_id is not None:
        return (await service.get_policy_catalog(remote_trainer_id)).backends
    return policy_catalog(_available_policy_classes()).backends


@router.get("/{policy}/huggingface-access")
async def check_huggingface_access(policy: str) -> HuggingFaceAccessResponse:
    """Check whether the configured token can read a policy's required Hub model."""
    from physicalai.policies import list_physicalai_policies

    if policy not in list_physicalai_policies():
        raise HTTPException(status_code=404, detail="Unknown policy")

    requirements = _HUGGINGFACE_REQUIREMENTS.get(policy, ())
    if not requirements:
        return HuggingFaceAccessResponse(requirements=[])
    token = resolve_hf_token()
    token_value = token.get_secret_value() if token is not None else ""
    if not token_value:
        return HuggingFaceAccessResponse(
            requirements=[
                HuggingFaceRequirementAccess(
                    repository=repository,
                    status="missing_token",
                    access_url=f"https://huggingface.co/{repository}",
                    required=required,
                )
                for repository, required in requirements
            ]
        )

    async def check_requirement(repository: str, required: bool) -> HuggingFaceRequirementAccess:
        access_url = f"https://huggingface.co/{repository}"
        status: _AccessStatus
        try:
            await asyncio.to_thread(HfApi(token=token_value).auth_check, repository)
        except (GatedRepoError, RepositoryNotFoundError):
            status = "denied"
        except Exception:
            status = "unavailable"
        else:
            status = "granted"
        return HuggingFaceRequirementAccess(
            repository=repository,
            status=status,
            access_url=access_url,
            required=required,
        )

    checked_requirements = await asyncio.gather(
        *(check_requirement(repository, required) for repository, required in requirements)
    )
    return HuggingFaceAccessResponse(requirements=checked_requirements)
