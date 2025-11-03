import asyncio
import multiprocessing as mp
from queue import Empty
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from api.dependencies import get_dataset_service, get_scheduler_ws
from core.scheduler import Scheduler
from exceptions import ResourceNotFoundError
from schemas import TeleoperationConfig
from services import DatasetService
from workers import TeleoperateWorker

router = APIRouter(prefix="/api/record")


@router.websocket("/teleoperate/ws")
async def teleoperate_websocket(  # noqa: C901
    websocket: WebSocket,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
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
        await websocket.send_json({
            "event": "dataset",
            "data": dataset.model_dump()
        })
    queue: mp.Queue = mp.Queue()
    process = TeleoperateWorker(
        stop_event=scheduler.mp_stop_event,
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
            print("Except: disconnected!")
            if process is not None:
                process.stop()

    async def handle_outgoing():
        try:
            while True:
                try:
                    message = queue.get_nowait()
                    await websocket.send_json(message)
                except Empty:
                    await asyncio.sleep(0.05)
        except Exception as e:
            print(f"Outgoing task stopped: {e}")

    incoming_task = asyncio.create_task(handle_incoming())
    outgoing_task = asyncio.create_task(handle_outgoing())

    _, pending = await asyncio.wait(
        {incoming_task, outgoing_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    # cancel whichever task is still running
    for task in pending:
        task.cancel()

    print("websocket handling done...")
