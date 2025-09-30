from typing import Annotated
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import ValidationError
import asyncio

import multiprocessing as mp
from schemas import TeleoperationConfig
from core.scheduler import Scheduler
from api.dependencies import get_scheduler
from workers import TeleoperateWorker
router = APIRouter(prefix="/api/record")


@router.websocket("/teleoperate/ws")
async def teleoperate_websocket(
        websocket: WebSocket,
        scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> None:
    """Robot control websocket"""
    await websocket.accept()
    data = await websocket.receive_json("text")
    config = TeleoperationConfig.model_validate(data["data"])
    queue = mp.Queue()
    process = TeleoperateWorker(
        stop_event=scheduler.mp_stop_event,
        config=config,
        queue=queue,
    )
    process.start()

    print("got a connection")
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
                except mp.queues.Empty:
                    await asyncio.sleep(0.05)
        except Exception as e:
            print(f"Outgoing task stopped: {e}")

    incoming_task = asyncio.create_task(handle_incoming())
    outgoing_task = asyncio.create_task(handle_outgoing())

    done, pending = await asyncio.wait(
        {incoming_task, outgoing_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    # cancel whichever task is still running
    for task in pending:
        task.cancel()

    print("websocket handling done...")
