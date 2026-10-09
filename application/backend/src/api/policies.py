import asyncio
from importlib import metadata
from typing import Literal

from fastapi import APIRouter, HTTPException
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from loguru import logger
from packaging.requirements import Requirement
from pydantic import BaseModel

from services.training_backends.local import resolve_hf_token

router = APIRouter(prefix="/api/policies", tags=["Policies"])

_AccessStatus = Literal["granted", "missing_token", "denied", "unavailable", "not_required"]


def _missing_policy_dependencies(name: str) -> list[str]:
    """Check declared extras without importing deferred model dependencies."""
    missing = []
    for raw in metadata.requires("physicalai-train") or []:
        requirement = Requirement(raw)
        marker = requirement.marker
        if marker is None or not marker.evaluate({"extra": name}) or marker.evaluate({"extra": ""}):
            continue
        try:
            installed_version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            missing.append(requirement.name)
        else:
            if installed_version not in requirement.specifier:
                missing.append(f"{requirement.name}{requirement.specifier} (installed {installed_version})")
    return missing


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


class AvailablePolicy(BaseModel):
    """Policy metadata for the training UI."""

    id: str
    name: str
    description: str
    min_vram: int | None


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
    from physicalai.policies import get_physicalai_policy_class, list_physicalai_policies

    classes = {}
    for name in list_physicalai_policies():
        if missing := _missing_policy_dependencies(name):
            logger.warning("Policy {} is unavailable (missing optional dependencies): {}", name, ", ".join(missing))
            continue
        try:
            cls = get_physicalai_policy_class(name)
            if not getattr(cls, "STUDIO_TRAINABLE", False):
                continue
            backends = (
                [str(b) for b in cls.get_supported_export_backends()]
                if hasattr(cls, "get_supported_export_backends")
                else []
            )
            classes[name] = (cls, backends)
        except ImportError as exc:
            logger.warning("Policy {} is unavailable: {}", name, exc)
    return classes


@router.get("/available")
def get_available_policies() -> list[AvailablePolicy]:
    """Return trainable policies with their display and hardware metadata."""
    policies = []
    for name, (cls, _) in _available_policy_classes().items():
        vram_gb = getattr(cls, "MIN_TRAINING_VRAM_GB", None)
        policies.append(
            AvailablePolicy(
                id=name,
                name=getattr(cls, "DISPLAY_NAME", cls.__name__),
                description=getattr(cls, "TRAINING_DESCRIPTION", (cls.__doc__ or cls.__name__).strip().splitlines()[0]),
                min_vram=vram_gb * 1024**3 if vram_gb is not None else None,
            )
        )
    return policies


@router.get("/backends")
def get_supported_backends_per_policy() -> dict[str, list[str]]:
    """Return the supported export backends for each policy."""
    return {name: backends for name, (_, backends) in _available_policy_classes().items()}


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
