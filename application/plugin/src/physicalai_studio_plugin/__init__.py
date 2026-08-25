"""Public API for the Physical AI Studio plugin."""

from .assets import RobotAsset
from .catalog import (
    BuildRobotCallable,
    CatalogRobot,
    PayloadContainer,
    RobotAdapterOptions,
    RobotCatalogDefinition,
    RobotCatalogRegistry,
)
from .factory import CatalogRobotFactory
from .probe import PortScanner, RobotProbe
from .schemas import SerialPortInfo
from .transport import shared_robot_name
from .ui_schema import (
    RobotFieldUiOptions,
    RobotPayloadUiOptions,
    RobotUiConnectionBinding,
    RobotUiConnectionItem,
    RobotUiFieldItem,
    RobotUiInfoItem,
    RobotUiItem,
    RobotUiSectionOptions,
    robot_field_ui,
    robot_payload_ui,
    validate_robot_payload_ui,
)

__all__ = [
    "BuildRobotCallable",
    "CatalogRobot",
    "CatalogRobotFactory",
    "PayloadContainer",
    "PortScanner",
    "RobotAdapterOptions",
    "RobotAsset",
    "RobotCatalogDefinition",
    "RobotCatalogRegistry",
    "RobotFieldUiOptions",
    "RobotPayloadUiOptions",
    "RobotProbe",
    "RobotUiConnectionBinding",
    "RobotUiConnectionItem",
    "RobotUiFieldItem",
    "RobotUiInfoItem",
    "RobotUiItem",
    "RobotUiSectionOptions",
    "SerialPortInfo",
    "robot_field_ui",
    "robot_payload_ui",
    "shared_robot_name",
    "validate_robot_payload_ui",
]
