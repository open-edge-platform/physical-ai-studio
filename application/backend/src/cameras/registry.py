"""Registry-backed construction of Studio's built-in cameras."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, cast

from physicalai.config import Config, instantiate

from cameras.catalog import CameraCatalogDefinition, CatalogCamera
from schemas.project_camera import BaslerCameraPayload, IPCameraPayload, RealsenseCameraPayload, USBCameraPayload

if TYPE_CHECKING:
    from physicalai.capture import Camera as PhysicalAICamera

    from cameras.catalog import CatalogCameraFactory
    from schemas.project_camera import Camera


async def _build_builtin(
    camera: CatalogCamera[Any], _factory: CatalogCameraFactory, *, class_path: str
) -> PhysicalAICamera:
    # Keep the existing constructor arguments until the per-camera catalogs move.
    fields = {"width", "height", "fps", "url"} if camera.type == "ipcam" else {"width", "height", "fps"}
    kwargs = {key: value for key, value in camera.payload.model_dump().items() if key in fields and value is not None}
    if camera.type == "realsense":
        kwargs["enable_depth"] = camera.payload.model_dump()["output_type"] in {"depth", "both"}
    if camera.type == "usb_camera":
        kwargs["device"] = camera.fingerprint
    elif camera.type != "ipcam":
        serial = camera.fingerprint.get("serial")
        if not isinstance(serial, str) or not serial:
            raise ValueError(f"Camera fingerprint for {camera.type!r} must include a non-empty 'serial'")
        kwargs["serial_number"] = serial
    return cast("PhysicalAICamera", instantiate(Config(class_path, kwargs)))


class CameraCatalogRegistry:
    """One definition per persisted driver; registration never opens hardware."""

    def __init__(self) -> None:
        self._definitions: dict[str, CameraCatalogDefinition[Any]] = {}
        for driver, display_name, payload, class_path in (
            ("usb_camera", "USB Camera", USBCameraPayload, "physicalai.capture.UVCCamera"),
            ("realsense", "Intel RealSense Camera", RealsenseCameraPayload, "physicalai.capture.RealSenseCamera"),
            ("basler", "Basler Camera", BaslerCameraPayload, "physicalai.capture.BaslerCamera"),
            ("ipcam", "IP Camera", IPCameraPayload, "physicalai.capture.IPCamera"),
        ):
            self.register_camera(
                CameraCatalogDefinition(
                    type=driver,
                    display_name=display_name,
                    camera_payload=payload,
                    camera_builder=partial(_build_builtin, class_path=class_path),
                )
            )

    def register_camera(self, definition: CameraCatalogDefinition[Any]) -> None:
        """Register a definition, rejecting duplicate driver names."""
        if definition.type in self._definitions:
            raise ValueError(f"Duplicate camera catalog registration for type: {definition.type}")
        self._definitions[definition.type] = definition

    def get_definition(self, driver: str) -> CameraCatalogDefinition[Any] | None:
        """Look up a persisted camera driver."""
        return self._definitions.get(driver)

    def list_definitions(self) -> list[CameraCatalogDefinition[Any]]:
        """List definitions in registration order."""
        return list(self._definitions.values())

    def supports(self, driver: str) -> bool:
        """Whether this driver has a registered builder."""
        definition = self.get_definition(driver)
        return definition is not None and definition.camera_builder is not None

    async def build_camera(self, camera: Camera) -> PhysicalAICamera:
        """Validate and construct a camera without connecting it."""
        definition = self.get_definition(camera.driver)
        if definition is None or definition.camera_builder is None:
            raise ValueError(f"Unsupported camera driver {camera.driver!r}")
        if not camera.fingerprint:
            raise ValueError("Camera must be reselected")
        descriptor = CatalogCamera(
            type=camera.driver,
            fingerprint=camera.fingerprint,
            payload=definition.camera_payload.model_validate(camera.payload.model_dump()),
        )
        return await definition.camera_builder(descriptor, self)

    def build_camera_config(self, camera: Camera) -> Config:
        """Bridge synchronous workers/exports to async builders in one place."""

        def build() -> Config:
            return Config.from_instance(asyncio.run(self.build_camera(camera)))

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return build()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(build).result()


@lru_cache
def get_camera_catalog_registry() -> CameraCatalogRegistry:
    """Return the process-wide built-in camera registry."""
    return CameraCatalogRegistry()
