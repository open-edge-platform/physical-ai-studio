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
    )
    assert len(exports) == 10
