from __future__ import annotations


def test_imports() -> None:
    from physicalai_studio_plugin import (
        BuildRobotCallable,
        CatalogRobot,
        CatalogRobotFactory,
        PayloadContainer,
        PortScanner,
        RobotAdapterOptions,
        RobotAsset,
        RobotCatalogDefinition,
        RobotProbe,
        SerialPortInfo,
        shared_robot_name,
    )

    exports = (
        BuildRobotCallable,
        CatalogRobot,
        CatalogRobotFactory,
        PayloadContainer,
        PortScanner,
        RobotAdapterOptions,
        RobotAsset,
        RobotCatalogDefinition,
        RobotProbe,
        SerialPortInfo,
        shared_robot_name,
    )
    assert len(exports) == 11
