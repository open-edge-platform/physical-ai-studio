import asyncio
import multiprocessing as mp
from queue import Empty
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket
from fastapi.responses import Response
from loguru import logger

from api.dependencies import ModelRegistryDep, RobotCalibrationServiceDep, RobotConnectionManagerDep, get_scheduler_ws
from core.scheduler import Scheduler
from robots.robot_client_factory import RobotClientFactory
from schemas import Dataset, Model
from schemas.environment import EnvironmentWithRelations
from workers.robot_control_worker import RobotControlWorker

router = APIRouter(prefix="/api/record")


@router.get("/robot_control/ws", tags=["WebSocket"], summary="Robot Control (WebSocket)", status_code=426)
async def robot_control_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


async def handle_incoming(websocket: WebSocket, process: RobotControlWorker) -> None:
    """Handle incoming messages for robot control."""
    try:
        while True:
            data = await websocket.receive_json("text")
            payload = data.get("data", {})
            match data["event"]:
                case "load_environment":
                    process.load_environment(EnvironmentWithRelations.model_validate(payload["environment"]))
                case "load_model":
                    process.load_model(Model.model_validate(payload["model"]), payload["backend"])
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
                    process.stop()
                case "disconnect":
                    process.disconnect()
                    break
    except Exception as e:
        logger.error(f"Incoming task stopped: {e}")
        logger.info("Except: disconnected!")


async def handle_outgoing(websocket: WebSocket, queue: mp.Queue) -> None:
    """Handle outgoing messages for robot control."""
    try:
        while True:
            try:
                loop = asyncio.get_running_loop()

                message = await loop.run_in_executor(None, queue.get)
                await websocket.send_json(message)
            except Empty:
                await asyncio.sleep(0.05)
    except Exception as e:
        logger.error(f"Outgoing task stopped: {e}")


@router.websocket("/robot_control/ws")
async def robot_control_websocket(
    websocket: WebSocket,
    robot_manager: RobotConnectionManagerDep,
    calibration_service: RobotCalibrationServiceDep,
    scheduler: Annotated[Scheduler, Depends(get_scheduler_ws)],
    model_registry: ModelRegistryDep,
) -> None:
    """Robot control websocket."""
    await websocket.accept()
    queue: mp.Queue = mp.Queue()
    process = RobotControlWorker(
        stop_event=scheduler.mp_stop_event,
        robot_client_factory=RobotClientFactory(
            robot_manager=robot_manager,
            calibration_service=calibration_service,
        ),
        queue=queue,
        model_worker_registry=model_registry,
    )
    process.start()

    incoming_task = asyncio.create_task(handle_incoming(websocket, process))
    outgoing_task = asyncio.create_task(handle_outgoing(websocket, queue))

    _, pending = await asyncio.wait(
        {incoming_task, outgoing_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

    if process is not None:
        process.disconnect()
        process.join(10)

    queue.close()
    logger.info("websocket handling done...")
