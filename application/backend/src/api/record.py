import asyncio
import multiprocessing as mp
from queue import Empty
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from loguru import logger

from api.dependencies import (
    RobotCalibrationServiceDep,
    RobotConnectionManagerDep,
    get_dataset_service,
    get_scheduler_ws,
)
from core.scheduler import Scheduler
from exceptions import ResourceNotFoundError
from schemas import InferenceConfig, TeleoperationConfig
from services import DatasetService
from workers import InferenceWorker, TeleoperateWorker

router = APIRouter(prefix="/api/record")


@router.websocket("/teleoperate/ws")
async def teleoperate_websocket(  # noqa: C901
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
                    break
        except WebSocketDisconnect:
            logger.info("Except: disconnected!")
            if process is not None:
                process.stop()

    def to_python_primitive(obj: Any) -> Any:
        """Replace numpy values to primitive types."""
        if isinstance(obj, dict):
            return {k: to_python_primitive(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_python_primitive(v) for v in obj]
        if isinstance(obj, np.generic):  # catches np.float32, np.int64, etc.
            return obj.item()
        return obj

    async def handle_outgoing():
        try:
            while True:
                try:
                    message = to_python_primitive(queue.get_nowait())
                    await websocket.send_json(message)
                except Empty:
                    await asyncio.sleep(0.05)
        except Exception as e:
            raise e
            logger.info(f"Outgoing task stopped: {e}")

    incoming_task = asyncio.create_task(handle_incoming())
    outgoing_task = asyncio.create_task(handle_outgoing())

    _, pending = await asyncio.wait(
        {incoming_task, outgoing_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    # cancel whichever task is still running
    for task in pending:
        task.cancel()

    logger.info("websocket handling done...")


@router.websocket("/inference/ws")
async def inference_websocket(
    websocket: WebSocket,
    robot_manager: RobotConnectionManagerDep,
    calibration_service: RobotCalibrationServiceDep,
    scheduler: Annotated[Scheduler, Depends(get_scheduler_ws)],
) -> None:
    """Robot control websocket."""
    await websocket.accept()
    data = await websocket.receive_json("text")
    config = InferenceConfig.model_validate(data["data"])
    queue: mp.Queue = mp.Queue()
    process = InferenceWorker(
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
                if data["event"] == "start_task":
                    task_index = data["data"]["task_index"]
                    process.start_task(task_index)
                if data["event"] == "stop":
                    process.stop()
                if data["event"] == "disconnect":
                    process.disconnect()
                    break
        except WebSocketDisconnect:
            logger.info("Except: disconnected!")
            if process is not None:
                process.disconnect()

    async def handle_outgoing():
        try:
            while True:
                try:
                    message = queue.get_nowait()
                    await websocket.send_json(message)
                except Empty:
                    await asyncio.sleep(0.05)
        except Exception as e:
            raise e
            logger.info(f"Outgoing task stopped: {e}")

    incoming_task = asyncio.create_task(handle_incoming())
    outgoing_task = asyncio.create_task(handle_outgoing())

    _, pending = await asyncio.wait(
        {incoming_task, outgoing_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
    logger.info("websocket handling done...")
