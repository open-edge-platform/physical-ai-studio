from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from physicalai_studio_plugin import RobotAsset, RobotCatalogDefinition
from physicalai_studio_plugin import RobotCatalogRegistry as RobotCatalogRegistryProtocol
from pydantic import BaseModel, Field, TypeAdapter, create_model

from schemas.robot_type import BaseRobot

from . import so101, widowxai

CATALOG_PLUGIN_ENTRYPOINT_GROUP = "physicalai.studio.catalog_plugins"

RegisterPluginCallable = Callable[["RobotCatalogRegistry"], None]


def _build_union(types: list[type]) -> Any:
    """Dynamically build Union[T1, T2, ...] from a list of types."""
    if not types:
        return BaseRobot
    result: Any = types[0]
    for t in types[1:]:
        result = result | t
    return result


class RobotCatalogRegistry(RobotCatalogRegistryProtocol):
    def __init__(self) -> None:
        self._definitions: dict[str, RobotCatalogDefinition] = {}
        self._robot_models: dict[str, type[BaseRobot]] = {}
        self._robot_adapter: TypeAdapter | None = None

        for definition in so101.get_definitions() + widowxai.get_definitions():
            self.register_robot(definition)

        self._load_external_plugins()

    def list_definitions(self) -> list[RobotCatalogDefinition]:
        return list(self._definitions.values())

    def get_definition(self, robot_type: str) -> RobotCatalogDefinition | None:
        return self._definitions.get(robot_type)

    def register_robot(self, definition: RobotCatalogDefinition | Any) -> None:
        definition = self._coerce_definition(definition)
        if definition.type in self._definitions:
            raise ValueError(f"Duplicate robot catalog registration for type: {definition.type}")

        self._definitions[definition.type] = definition
        self._robot_models.pop(definition.type, None)
        self._robot_adapter = None

    @staticmethod
    def _coerce_definition(definition: RobotCatalogDefinition | Any) -> RobotCatalogDefinition:
        """Adapt legacy plugin definitions until plugins adopt ``RobotAsset``."""
        if isinstance(definition, RobotCatalogDefinition):
            return definition

        plugin_asset = getattr(definition, "asset", None)
        if plugin_asset is not None:
            asset = RobotAsset(
                urdf_relative_path=Path(plugin_asset.urdf_relative_path),
                packages={package: Path(path) for package, path in plugin_asset.packages.items()},
                joint_map=plugin_asset.joint_map,
                root_resolver=plugin_asset.root_resolver,
            )
        else:
            asset = None

        urdf_relative_path = getattr(definition, "urdf_relative_path", None)
        if asset is None and urdf_relative_path is not None:
            relative_path = Path(urdf_relative_path)
            package_paths = getattr(definition, "asset_packages", None)
            if package_paths is None:
                package_paths = dict.fromkeys(getattr(definition, "package_map", {}), relative_path.parts[0])
            asset = RobotAsset(
                urdf_relative_path=relative_path,
                packages={package: Path(path) for package, path in package_paths.items()},
                joint_map=getattr(definition, "joint_map", {}),
                root_resolver=getattr(definition, "asset_root_resolver", None),
            )

        robot_payload = getattr(definition, "robot_payload", None)

        return RobotCatalogDefinition(
            type=definition.type,
            display_name=definition.display_name,
            role=definition.role,
            robot_builder=definition.robot_builder,
            robot_payload=robot_payload,
            asset=asset,
            adapter_options=definition.adapter_options,
            probe=definition.probe,
        )

    def get_robot_types(self) -> list[type[BaseModel]]:
        models: list[type[BaseModel]] = []
        for d in self._definitions.values():
            model = self._robot_models.get(d.type)
            if model is None:
                payload_annotation = dict[str, Any] if d.robot_payload is None else d.robot_payload
                payload_default: Any = Field(default_factory=dict) if d.robot_payload is None else ...

                model = create_model(
                    f"{d.type}Robot",
                    __base__=BaseRobot,
                    type=Literal[d.type],  # pyrefly: ignore[invalid-literal]
                    payload=(payload_annotation, payload_default),
                )
                self._robot_models[d.type] = model
            models.append(model)
        return models

    def make_robot_type(self) -> Any:
        models = self.get_robot_types()
        return Annotated[_build_union(models), Field(discriminator="type")]

    def _load_external_plugins(self) -> None:
        discovered_entry_points = list(entry_points(group=CATALOG_PLUGIN_ENTRYPOINT_GROUP))

        for discovered_entry_point in discovered_entry_points:
            register_plugin = discovered_entry_point.load()
            if not callable(register_plugin):
                raise ValueError(
                    f"Catalog plugin entry point '{discovered_entry_point.name}' must load a callable "
                    "register_physicalai_studio_plugin(registry)"
                )

            plugin_callable: RegisterPluginCallable = cast("RegisterPluginCallable", register_plugin)
            plugin_callable(self)

    def get_robot_adapter(self) -> TypeAdapter:
        if self._robot_adapter is None:
            self._robot_adapter = TypeAdapter(self.make_robot_type())
        return self._robot_adapter
