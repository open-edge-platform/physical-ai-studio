from fastapi import APIRouter
from physicalai.policies import ACT, Pi0, Pi05, SmolVLA

router = APIRouter(prefix="/api/policies", tags=["Policies"])

_POLICY_CLASSES = {
    "act": ACT,
    "pi0": Pi0,
    "pi05": Pi05,
    "smolvla": SmolVLA,
}


@router.get("/backends")
def get_supported_backends_per_policy() -> dict[str, list[str]]:
    """Return the supported export backends for each policy."""
    return {
        name: [str(b) for b in cls.get_supported_export_backends()]
        if hasattr(cls, "get_supported_export_backends")
        else []
        for name, cls in _POLICY_CLASSES.items()
    }
