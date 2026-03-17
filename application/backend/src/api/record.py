import asyncio
import multiprocessing as mp
from queue import Empty
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket
from fastapi.responses import Response
from loguru import logger

from api.dependencies import (
    RobotCalibrationServiceDep,
    RobotConnectionManagerDep,
    get_dataset_service,
    get_scheduler_ws,
)
from core.scheduler import Scheduler
from exceptions import ResourceNotFoundError
from robots.robot_client_factory import RobotClientFactory
from schemas import Model, TeleoperationConfig, Dataset
from schemas.environment import EnvironmentWithRelations
from services import DatasetService
from utils.serialize_utils import to_python_primitive
from workers.inference_worker import InferenceWorker
from workers.teleoperate_worker import TeleoperateWorker

router = APIRouter(prefix="/api/record")


@router.get("/teleoperate/ws", tags=["WebSocket"], summary="Teleoperation (WebSocket)", status_code=426)
async def teleoperate_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/teleoperate/ws")
async def teleoperate_websocket(
    websocket: WebSocket,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    robot_manager: RobotConnectionManagerDep,
    calibration_service: RobotCalibrationServiceDep,
    scheduler: Annotated[Scheduler, Depends(get_scheduler_ws)],
) -> None:
    """Robot control websocket."""
    await websocket.accept()
    data = await websocket.receive_json("text")
    config = TeleoperationConfig.model_validate(data["data"])
    try:
        await dataset_service.get_dataset_by_id(config.dataset.id)
    except ResourceNotFoundError:
        dataset = await dataset_service.create_dataset(config.dataset)
        await websocket.send_json({"event": "dataset", "data": dataset.model_dump()})
    queue: mp.Queue = mp.Queue()
    process = TeleoperateWorker(
        stop_event=scheduler.mp_stop_event,
        robot_manager=robot_manager,
        calibration_service=calibration_service,
        config=config,
        queue=queue,
    )
    process.start()

    async def handle_incoming():
        try:
            while True:
                data = await websocket.receive_json("text")
                if data["event"] == "start_recording":
                    process.start_recording()
                if data["event"] == "cancel":
                    process.reset()
                if data["event"] == "save":
                    process.save()
                if data["event"] == "disconnect":
                    process.stop()
                    process.join(timeout=5)
                    break
        except Exception:
            logger.info("Except: disconnected!")

    async def handle_outgoing():
        try:
            while True:
                try:
                    message = to_python_primitive(queue.get_nowait())
                    await websocket.send_json(message)
                except Empty:
                    await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"Outgoing task stopped: {e}")

    incoming_task = asyncio.create_task(handle_incoming())
    outgoing_task = asyncio.create_task(handle_outgoing())

    _, pending = await asyncio.wait(
        {incoming_task, outgoing_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

    if process is not None:
        process.stop()
        process.join(10)

    queue.close()
    logger.info("websocket handling done...")


@router.get("/inference/ws", tags=["WebSocket"], summary="Inference (WebSocket)", status_code=426)
async def inference_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/inference/ws")
async def inference_websocket(
    websocket: WebSocket,
    robot_manager: RobotConnectionManagerDep,
    calibration_service: RobotCalibrationServiceDep,
    scheduler: Annotated[Scheduler, Depends(get_scheduler_ws)],
) -> None:
    """Robot control websocket."""
    await websocket.accept()
    queue: mp.Queue = mp.Queue()
    process = InferenceWorker(
        stop_event=scheduler.mp_stop_event,
        robot_client_factory=RobotClientFactory(
            robot_manager=robot_manager,
            calibration_service=calibration_service,
        ),
        queue=queue,
    )
    process.start()

    async def handle_incoming():
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

    async def handle_outgoing():
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

    incoming_task = asyncio.create_task(handle_incoming())
    outgoing_task = asyncio.create_task(handle_outgoing())

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
