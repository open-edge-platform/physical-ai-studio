import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger

from api.dependencies import (
    RecordingLockedCamerasDep,
    RobotCalibrationServiceDep,
    RobotConnectionManagerDep,
    get_scheduler_ws,
)
from core.scheduler import Scheduler
from robots.robot_client_factory import RobotClientFactory
from schemas import Dataset, InferenceDevice, Model
from schemas.environment import EnvironmentWithRelations
from workers.base import run_at_frequency
from workers.robot_control_orchestrator_worker import RobotControlOrchestrator

router = APIRouter(prefix="/api/record")


@router.get("/robot_control/ws", tags=["WebSocket"], summary="Robot Control (WebSocket)", status_code=426)
async def robot_control_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


async def handle_incoming(  # noqa: PLR0912
    websocket: WebSocket,
    process: RobotControlOrchestrator,
    locked_camera_fingerprints: set[str],
) -> None:
    """Handle incoming messages for robot control."""
    try:
        while True:
            data = await websocket.receive_json("text")
            payload = data.get("data", {})
            match data["event"]:
                case "load_environment":
                    environment = EnvironmentWithRelations.model_validate(payload["environment"])
                    locked_camera_fingerprints.clear()
                    locked_camera_fingerprints.update(camera.fingerprint for camera in environment.cameras)
                    process.load_environment(environment)
                case "load_model":
                    process.load_model(
                        Model.model_validate(payload["model"]),
                        InferenceDevice.model_validate(payload["inference_device"]),
                    )
                case "load_dataset":
                    process.load_dataset(Dataset.model_validate(payload["dataset"]))
                case "set_follower_source":
                    process.set_follower_source(payload["follower_source"])
                case "start_recording":
                    process.start_recording(payload["task"])
                case "save_episode":
                    process.save_episode()
                case "discard_episode":
                    process.discard_episode()
                case "start_task":
                    process.start_task(payload["task"])
                case "stop_task":
                    process.stop_task()
                case "disconnect":
                    process.disconnect()
                    break
            await asyncio.sleep(0.005)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Incoming task stopped: {e}")
        logger.info("Except: disconnected!")


async def handle_outgoing(websocket: WebSocket, queue: asyncio.Queue) -> None:
    """Handle outgoing messages for robot control."""
    try:
        while True:
            data = await queue.get()
            if data is None:
                break
            await websocket.send_json(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Outgoing task stopped: {e}")


async def observation_update_loop(websocket: WebSocket, robot_control: RobotControlOrchestrator) -> None:
    """Handle outgoing messages for robot control."""
    try:
        while True:
            async with run_at_frequency(30):
                try:
                    observation = robot_control.get_observation_report()
                    if observation:
                        await websocket.send_json({"event": "observations", "data": observation})
                except Exception as e:
                    logger.error(f"Observation update error: {e}")
    except WebSocketDisconnect:
        pass


@router.websocket("/robot_control/ws")
async def robot_control_websocket(
    websocket: WebSocket,
    robot_manager: RobotConnectionManagerDep,
    calibration_service: RobotCalibrationServiceDep,
    scheduler: Annotated[Scheduler, Depends(get_scheduler_ws)],
    locked_camera_fingerprints: RecordingLockedCamerasDep,
) -> None:
    """Robot control websocket."""
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    robot_control = RobotControlOrchestrator(
        message_queue=queue,
        robot_client_factory=RobotClientFactory(robot_manager=robot_manager, calibration_service=calibration_service),
        mp_terminate_event=scheduler.mp_stop_event,
    )
    robot_control.start()
    try:
        incoming_task = asyncio.create_task(handle_incoming(websocket, robot_control, locked_camera_fingerprints))
        outgoing_task = asyncio.create_task(handle_outgoing(websocket, queue))
        observation_update_task = asyncio.create_task(observation_update_loop(websocket, robot_control))

        _, pending = await asyncio.wait(
            {incoming_task, outgoing_task, observation_update_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
    finally:
        robot_control.stop()
        locked_camera_fingerprints.clear()
