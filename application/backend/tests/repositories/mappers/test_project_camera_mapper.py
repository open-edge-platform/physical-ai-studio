from uuid import uuid4

from db.schema import ProjectCameraDB
from repositories.mappers.project_camera_mapper import ProjectCameraMapper


def _camera(fingerprint: str) -> ProjectCameraDB:
    return ProjectCameraDB(
        id=str(uuid4()),
        project_id=str(uuid4()),
        name="Front camera",
        driver="usb_camera",
        fingerprint=fingerprint,
        hardware_name="Camera",
        payload={"width": 640, "height": 480, "fps": 30},
    )


def test_from_schema_parses_a_structured_fingerprint() -> None:
    camera = ProjectCameraMapper.from_schema(_camera('{"serial":"abc"}'))

    assert camera.fingerprint == {"serial": "abc"}


def test_from_schema_marks_a_legacy_fingerprint_for_reselection() -> None:
    camera = ProjectCameraMapper.from_schema(_camera("/dev/video0"))

    assert camera.fingerprint is None
