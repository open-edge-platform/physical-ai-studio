from fastapi import BackgroundTasks

from api.system import _stop_process, restart_server
from services.health_service import HealthService


async def test_restart_server_marks_restart_required_and_schedules_shutdown() -> None:
    background_tasks = BackgroundTasks()
    health_service = HealthService()

    response = await restart_server(background_tasks, health_service)

    assert response == {"status": "restarting"}
    assert health_service.plugin_restart_required is True
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].func is _stop_process
