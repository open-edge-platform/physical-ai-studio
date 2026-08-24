from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, create_model

from robots.catalog.registry import RobotCatalogRegistry
from robots.catalog.so101 import SO101Robot
from robots.catalog.widowxai import TrossenBimanualRobot, TrossenSingleArmRobot
from schemas.robot_type import BaseRobot, RobotType

__all__ = [
    "ReadableRobot",
    "Robot",
    "RobotAdapter",
    "RobotType",
    "RobotWithConnectionState",
    "RobotWithConnectionStateAdapter",
    "SO101Robot",
    "TrossenBimanualRobot",
    "TrossenSingleArmRobot",
    "UnavailableRobot",
    "UnavailableRobotWithConnectionState",
]

# ============================================================================
# Discriminated union of all robot types — built dynamically from registry
# ============================================================================

_registry = RobotCatalogRegistry()
Robot = _registry.make_robot_type()
RobotAdapter: TypeAdapter = _registry.get_robot_adapter()


class UnavailableRobot(BaseRobot):
    """A persisted robot whose catalog plugin is not currently installed."""

    type: str
    payload: dict[str, Any]
    unavailable: Literal[True] = True


ReadableRobot = Robot | UnavailableRobot


# ============================================================================
# RobotWithConnectionState variants
# ============================================================================

_ConnectionStatus = Literal["online", "offline", "unknown"]


def _build_union(models: list[type]) -> Any:
    result: Any = models[0]
    for model in models[1:]:
        result = result | model
    return result


def _with_connection_status(robot_model: type) -> type:
    return create_model(
        f"{robot_model.__name__}WithConnectionState",
        __base__=robot_model,
        connection_status=(_ConnectionStatus, "unknown"),
    )


_connection_state_models = [_with_connection_status(model) for model in _registry.get_robot_types()]
RobotWithConnectionState: Any = Annotated[_build_union(_connection_state_models), Field(discriminator="type")]
RobotWithConnectionStateAdapter: TypeAdapter[Any] = TypeAdapter(RobotWithConnectionState)


class UnavailableRobotWithConnectionState(UnavailableRobot):
    connection_status: Literal["unknown"] = "unknown"
