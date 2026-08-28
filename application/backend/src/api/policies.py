import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from physicalai.policies import ACT, Pi0, Pi05, SmolVLA
from pydantic import BaseModel

from settings import get_settings

router = APIRouter(prefix="/api/policies", tags=["Policies"])

_AccessStatus = Literal["granted", "missing_token", "denied", "unavailable", "not_required"]

_POLICY_CLASSES = {
    "act": ACT,
    "pi0": Pi0,
    "pi05": Pi05,
    "smolvla": SmolVLA,
}

_HUGGINGFACE_REQUIREMENTS = {
    # Hub dependencies are lazy and can vary with a policy's configuration, so
    # there is no library API that can discover them safely without starting a
    # download. When adding/changing a policy, inspect its `from_pretrained`,
    # `hf_hub_download`, and `Auto*from_pretrained` calls and list every default
    # repository that Studio training needs here.
    "pi05": (
        ("lerobot/pi05_base", True),
        ("google/paligemma-3b-pt-224", True),
    ),
    "smolvla": (("lerobot/smolvla_base", False),),
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


@router.get("/backends")
def get_supported_backends_per_policy() -> dict[str, list[str]]:
    """Return the supported export backends for each policy."""
    return {
        name: [str(b) for b in cls.get_supported_export_backends()]
        if hasattr(cls, "get_supported_export_backends")
        else []
        for name, cls in _POLICY_CLASSES.items()
    }


@router.get("/{policy}/huggingface-access")
async def check_huggingface_access(policy: str) -> HuggingFaceAccessResponse:
    """Check whether the configured token can read a policy's required Hub model."""
    if policy not in _POLICY_CLASSES:
        raise HTTPException(status_code=404, detail="Unknown policy")

    requirements = _HUGGINGFACE_REQUIREMENTS.get(policy, ())
    if not requirements:
        return HuggingFaceAccessResponse(requirements=[])
    token = get_settings().huggingface.hf_token
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
