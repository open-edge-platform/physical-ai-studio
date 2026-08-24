from collections.abc import Callable
from unittest.mock import MagicMock

from fastapi.dependencies.utils import get_dependant
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import (
    get_dataset_service,
    get_job_service,
    get_log_service,
    get_model_metrics_service,
    get_model_service,
    get_robot_client_factory,
)
from api.record import robot_control_websocket
from api.robot_control import robot_websocket
from robots.robot_client_factory import RobotClientFactory
from services.job_service import JobService
from services.robot_catalog_service import RobotCatalogService
from settings import Settings
from utils.serial_robot_tools import RobotConnectionManager


def _dependency_calls(endpoint: Callable[..., object]) -> set[Callable[..., object] | None]:
    """Return the direct FastAPI dependencies declared by an endpoint."""
    dependant = get_dependant(path="/", call=endpoint)
    return {dependency.call for dependency in dependant.dependencies}


def test_robot_client_factory_uses_provided_manager() -> None:
    robot_manager = MagicMock(spec=RobotConnectionManager)
    catalog_service = MagicMock(spec=RobotCatalogService)

    factory = get_robot_client_factory(robot_manager, catalog_service)

    assert isinstance(factory, RobotClientFactory)
    assert factory.robot_manager is robot_manager
    assert factory.catalog_registry is catalog_service.registry


def test_model_metrics_service_uses_injected_settings() -> None:
    settings = MagicMock(spec=Settings)

    service = get_model_metrics_service(settings)

    assert service.settings is settings


def test_log_service_uses_injected_collaborators() -> None:
    settings = MagicMock(spec=Settings)
    job_service = MagicMock(spec=JobService)

    service = get_log_service(settings, job_service)

    assert service.settings is settings
    assert service.job_service is job_service


def test_job_service_uses_dependency_provided_session() -> None:
    session = MagicMock(spec=AsyncSession)

    service = get_job_service(session)

    assert service.session is session


def test_database_services_share_dependency_provided_session() -> None:
    session = MagicMock(spec=AsyncSession)

    services = [get_job_service(session), get_model_service(session), get_dataset_service(session)]

    assert all(service.session is session for service in services)


def test_record_websocket_depends_on_robot_client_factory() -> None:
    assert get_robot_client_factory in _dependency_calls(robot_control_websocket)


def test_teleoperation_websocket_depends_on_robot_client_factory() -> None:
    assert get_robot_client_factory in _dependency_calls(robot_websocket)
